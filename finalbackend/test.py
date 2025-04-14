import os
import logging
from typing import List, Dict, Any
import traceback
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from werkzeug.utils import secure_filename
import requests
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import PyTorch
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available. Some functionality will be limited.")
    TORCH_AVAILABLE = False

# Try to import required dependencies
try:
    import fitz  # PyMuPDF
except ImportError:
    logger.error("PyMuPDF (fitz) not installed. Please install with: pip install pymupdf")
    raise

try:
    import chromadb
except ImportError:
    logger.error("ChromaDB not installed. Please install with: pip install chromadb")
    raise

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    logger.warning("spaCy not available. Using simpler text processing.")
    SPACY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    logger.warning("SentenceTransformer not available. Using simpler embedding method.")
    SENTENCE_TRANSFORMER_AVAILABLE = False

# Fallback sentence splitter
def simple_sentence_splitter(text: str) -> List[str]:
    sentences = []
    for potential_sentence in text.split('.'):
        potential_sentence = potential_sentence.strip()
        if potential_sentence:
            sentences.append(potential_sentence + '.')
    return sentences

class RAGSystem:
    def __init__(self,
                 embed_model_name: str = 'all-mpnet-base-v2',
                 device: str = "cuda",
                 db_path: str = "./chroma_db"):
        self.device = device
        try:
            if TORCH_AVAILABLE:
                if device == "cuda" and not torch.cuda.is_available():
                    logger.warning("CUDA not available. Falling back to CPU.")
                    self.device = "cpu"
            else:
                self.device = "cpu"

            # Initialize NLP
            if SPACY_AVAILABLE:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    logger.warning("spaCy model not found. Downloading...")
                    os.system("python -m spacy download en_core_web_sm")
                    self.nlp = spacy.load("en_core_web_sm")

            # Initialize embedding model
            if SENTENCE_TRANSFORMER_AVAILABLE:
                self.embed_model = SentenceTransformer(embed_model_name, device=self.device)
            else:
                logger.warning("Using fallback random embeddings")
                self.embed_dim = 384

            self.chroma_client = chromadb.PersistentClient(path=db_path)
            self.collection = self.chroma_client.get_or_create_collection(name="multi_doc_collection")
            self.doc_counter = 0  # To generate unique doc IDs

        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise

    def fallback_encode(self, texts: List[str]) -> List[np.ndarray]:
        return [np.random.rand(self.embed_dim).astype(np.float32) for _ in texts]

    @staticmethod
    def text_formatter(text: str) -> str:
        return text.replace("\n", " ").strip()

    def open_text_and_pages(self, pdf_path: str) -> List[Dict[str, Any]]:
        try:
            pdf = fitz.open(pdf_path)
            page_and_text = []
            for page_number, page in tqdm(enumerate(pdf, start=0)):
                text = self.text_formatter(page.get_text())
                if text.strip():
                    page_and_text.append({
                        "page_number": page_number,
                        "text": text
                    })
            pdf.close()
            return page_and_text
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            raise

    def process_sentences(self, page_and_text: List[Dict[str, Any]]) -> None:
        for item in tqdm(page_and_text):
            if SPACY_AVAILABLE:
                doc = self.nlp(item["text"])
                item["sentences"] = [str(sent).strip() for sent in doc.sents if str(sent).strip()]
            else:
                item["sentences"] = [sent.strip() for sent in simple_sentence_splitter(item["text"]) if sent.strip()]

    @staticmethod
    def split_sentences(sentences: List[str], chunk_size: int = 5) -> List[List[str]]:
        return [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]

    def extract_chunks(self, page_and_text: List[Dict[str, Any]], doc_id: str) -> List[Dict[str, Any]]:
        page_and_chunk = []
        for item in tqdm(page_and_text):
            chunks = self.split_sentences(item.get("sentences", []))
            for chunk in chunks:
                combined_chunk = " ".join(chunk).strip()
                if combined_chunk:
                    page_and_chunk.append({
                        "doc_id": doc_id,  # Track which document this chunk belongs to
                        "page_number": item["page_number"],
                        "sentence_chunk": combined_chunk,
                        "sentence_chunk_word_count": len(combined_chunk.split(" "))
                    })
        return page_and_chunk

    def embed_chunks(self, page_and_chunk: List[Dict[str, Any]]) -> None:
        for item in tqdm(page_and_chunk):
            if SENTENCE_TRANSFORMER_AVAILABLE:
                item["embeddings"] = self.embed_model.encode(
                    [item["sentence_chunk"]],
                    convert_to_tensor=False
                )[0].tolist()
            else:
                item["embeddings"] = self.fallback_encode([item["sentence_chunk"]])[0].tolist()

    def store_in_chromadb(self, page_and_chunk: List[Dict[str, Any]]) -> None:
        try:
            for i, row in enumerate(page_and_chunk):
                unique_id = f"{row['doc_id']}_{i}"  # Unique ID per chunk across all docs
                self.collection.add(
                    ids=[unique_id],
                    embeddings=[row["embeddings"]],
                    metadatas=[{
                        "doc_id": row["doc_id"],
                        "page_number": int(row["page_number"]),
                        "sentence_chunk": str(row["sentence_chunk"]),
                        "sentence_chunk_word_count": int(row["sentence_chunk_word_count"])
                    }]
                )
        except Exception as e:
            logger.error(f"ChromaDB storage error: {e}")
            raise

    def retrieve_data(self) -> pd.DataFrame:
        try:
            results = self.collection.get(include=["embeddings", "metadatas"])
            data = []
            for metadata, embedding in zip(results["metadatas"], results["embeddings"]):
                data.append({
                    "doc_id": metadata["doc_id"],
                    "page_number": metadata["page_number"],
                    "sentence_chunk": metadata["sentence_chunk"],
                    "sentence_chunk_word_count": metadata["sentence_chunk_word_count"],
                    "embeddings": np.array(embedding)
                })
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Data retrieval error: {e}")
            raise

    def relevant_search(self, query: str, embeddings: Any, k: int = 5):
        if TORCH_AVAILABLE and SENTENCE_TRANSFORMER_AVAILABLE:
            query_embedding = self.embed_model.encode([query], convert_to_tensor=True)
            if isinstance(embeddings, torch.Tensor):
                query_embedding = query_embedding.to(embeddings.device)
                cosine_sim = F.cosine_similarity(query_embedding, embeddings, dim=1)
                return torch.topk(cosine_sim, k=min(k, len(embeddings)))
            else:
                embeddings_tensor = torch.tensor(embeddings).to(query_embedding.device)
                cosine_sim = F.cosine_similarity(query_embedding, embeddings_tensor, dim=1)
                return torch.topk(cosine_sim, k=min(k, len(embeddings)))
        else:
            if SENTENCE_TRANSFORMER_AVAILABLE:
                query_embedding = self.embed_model.encode([query], convert_to_tensor=False)[0]
            else:
                query_embedding = self.fallback_encode([query])[0]
            
            embeddings_array = np.stack(embeddings) if not isinstance(embeddings, np.ndarray) else embeddings
            norm_query = np.linalg.norm(query_embedding)
            norm_embeddings = np.linalg.norm(embeddings_array, axis=1)
            cosine_sim = np.dot(embeddings_array, query_embedding) / (norm_embeddings * norm_query + 1e-10)
            top_indices = np.argsort(cosine_sim)[-k:][::-1]
            top_values = cosine_sim[top_indices]
            return top_values, top_indices

    @staticmethod
    def prompt_formatter(embed: List[Dict[str, Any]]) -> str:
        return "\n".join([f"From {item['doc_id']} (Page {item['page_number']}): {item['sentence_chunk']}" for item in embed])

# Flask app setup
app = Flask(__name__)
CORS(app)

# Initialize RAG system globally
try:
    rag_system = RAGSystem(device="cpu")
except Exception as e:
    logger.critical(f"Failed to initialize RAG system: {e}")
    rag_system = None

@app.route('/docUpload', methods=['POST', 'OPTIONS'])
def doc_upload():
    if request.method == 'OPTIONS':
        response = jsonify({"message": "Preflight check"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response

    if not rag_system:
        return jsonify({"error": "System not initialized"}), 500

    try:
        if 'files' not in request.files:
            logger.error("No file part in the request")
            return jsonify({"error": "No file part"}), 400
        
        files = request.files.getlist('files')  # Get multiple files
        if not files or all(file.filename == '' for file in files):
            logger.error("No selected files")
            return jsonify({"error": "No selected files"}), 400

        upload_dir = "./uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        for file in files:
            safe_filename = secure_filename(file.filename)
            filepath = os.path.join(upload_dir, safe_filename)
            file.save(filepath)
            logger.info(f"File saved: {filepath}")
            
            doc_id = f"doc_{rag_system.doc_counter}_{safe_filename}"  # Unique doc ID
            rag_system.doc_counter += 1
            
            page_and_text = rag_system.open_text_and_pages(filepath)
            rag_system.process_sentences(page_and_text)
            page_and_chunk = rag_system.extract_chunks(page_and_text, doc_id)
            rag_system.embed_chunks(page_and_chunk)
            rag_system.store_in_chromadb(page_and_chunk)
            
            os.remove(filepath)
            logger.info(f"Processed and removed: {filepath}")
        
        return jsonify({"message": f"Processed {len(files)} files successfully"}), 200
    
    except Exception as e:
        logger.error(f"Upload processing error: {e}")
        return jsonify({"error": "File upload failed", "details": str(e)}), 500

@app.route('/query', methods=['POST'])
def query():
    if not rag_system:
        return jsonify({"error": "System not initialized"}), 500

    try:
        data = request.json
        if not data or "query" not in data:
            return jsonify({"error": "No valid JSON data provided"}), 400
            
        query_text = data["query"].strip()
        if not query_text:
            return jsonify({"error": "Query cannot be empty"}), 400
        
        df = rag_system.retrieve_data()
        if df.empty:
            return jsonify({"error": "No data found in ChromaDB"}), 404

        embeddings = df['embeddings'].values
        if TORCH_AVAILABLE:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            embeddings_tensor = torch.tensor(np.stack(embeddings)).to(device)
            values, indices = rag_system.relevant_search(query_text, embeddings_tensor, k=10)  # Increase k for broader context
            indices_list = [i.item() for i in indices]
        else:
            values, indices = rag_system.relevant_search(query_text, embeddings, k=10)
            indices_list = indices.tolist()
        
        embed = [df.iloc[i].to_dict() for i in indices_list]
        context = rag_system.prompt_formatter(embed)
        
        prompt = f"""Based on the following information from multiple documents, please answer this question comprehensively: {query_text}
Context:
{context}
Answer:"""

        API_KEY = "YOUR_ACTUAL_API_KEY"  # Replace with your valid key
        API_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={API_KEY}"
        
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        headers = {"Content-Type": "application/json"}
        
        response = requests.post(API_ENDPOINT, headers=headers, json=payload)
        if response.status_code == 200:
            response_json = response.json()
            response_text = response_json.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
            return jsonify({"response": response_text})
        else:
            logger.error(f"Gemini API error: {response.text}")
            return jsonify({"response": f"Error getting response from AI model: {response.status_code}"}), 500
    
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    os.makedirs("./uploads", exist_ok=True)
    os.makedirs("./chroma_db", exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)