import React, { useState } from 'react';
import { Brain, Send, Upload, Bot, User, Loader2, Home, MessageCircle, User as UserIcon, Settings, FileText } from 'lucide-react';
import './App.css';


const ChatInterface = () => {
  // State for messages and input
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);
  
  // State for navigation
  const [activeTab, setActiveTab] = useState('chat');
  
  // State for file dropping in settings
  const [isDragging, setIsDragging] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState([]);

  const handleSendMessage = async () => {
    if (!inputMessage.trim()) return;
    
    setIsLoading(true);
    setMessages(prev => [...prev, { type: 'user', content: inputMessage }]);
    
    try {
      const response = await fetch("http://localhost:5000/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: inputMessage })
      });

      const result = await response.json();
      
      if (response.ok) {
        setMessages(prev => [...prev, { type: 'assistant', content: result.response }]);
      } else {
        setMessages(prev => [...prev, { type: 'error', content: "Error: " + (result.message || "Query failed") }]);
      }
    } catch (error) {
      setMessages(prev => [...prev, { type: 'error', content: "Network error: " + error.message }]);
    }
    
    setIsLoading(false);
    setInputMessage('');
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    handleFileProcessing(file);
  };
  
  const handleFileDrop = async (files) => {
    for (let i = 0; i < files.length; i++) {
      await handleFileProcessing(files[i]);
    }
  };
  
  const handleFileProcessing = async (file) => {
    if (file) {
      setUploadedFile(file);
      setUploadedFiles(prev => [...prev, file]);
      
      const formData = new FormData();
      formData.append("file", file);
      
      try {
        const response = await fetch("http://localhost:5000/docUpload", {
          method: "POST",
          body: formData
        });

        const result = await response.json().catch(() => null);
        
        if (response.ok) {
          setMessages(prev => [...prev, { type: 'system', content: `File "${file.name}" uploaded successfully!` }]);
        } else {
          setMessages(prev => [...prev, { type: 'error', content: `Error: ${result?.message || "Upload failed"}` }]);
        }
      } catch (error) {
        setMessages(prev => [...prev, { type: 'error', content: `Error uploading file: ${error.message}` }]);
      }
    }
  };
  
  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };
  
  const handleDragLeave = () => {
    setIsDragging(false);
  };
  
  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files.length > 0) {
      handleFileDrop(e.dataTransfer.files);
    }
  };

  // Render different content based on active tab
  const renderContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return (
          <div className="dashboard-container">
          <div className="text-center mb-12">
          <div className="flex items-center justify-center mb-6">
            <div className="bg-blue-100 p-3 rounded-2xl">
              <Brain className="h-8 w-8 text-blue-600" />
            </div>
          </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-4">RAG Fusion Search</h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Upload a file, provide a link, or directly enter your query to search through documents using RAG Fusion technology.
          </p>
        </div>
            <h2>Dashboard</h2>
            <p>Welcome to RagAI Analytics Dashboard</p>
            <div className="stats-grid">
              <div className="stat-card">
                <h3>Total Queries</h3>
                <p className="stat-value">1,245</p>
              </div>
              <div className="stat-card">
                <h3>Documents</h3>
                <p className="stat-value">{uploadedFiles.length}</p>
              </div>
              <div className="stat-card">
                <h3>Response Time</h3>
                <p className="stat-value">1.2s</p>
              </div>
            </div>
          </div>
        );
      case 'chat':
        return (
          <>
            <div className="messages-container">
              <div className="messages-wrapper">
                <div className="messages-list">
                  {messages.map((message, index) => (
                    <div key={index} className={`message-row ${message.type === 'user' ? 'user-message' : ''}`}>
                      {message.type !== 'user' && <div className="avatar assistant-avatar"><Bot /></div>}
                      <div className={`message-bubble ${message.type}`}>{message.content}</div>
                      {message.type === 'user' && <div className="avatar user-avatar"><User /></div>}
                    </div>
                  ))}
                  {isLoading && (
                    <div className="message-row">
                      <div className="avatar assistant-avatar"><Bot /></div>
                      <div className="message-bubble loading">
                        <Loader2 className="loading-icon" /> Thinking...
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
            <div className="input-container">
              <input
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                placeholder="Type your message here..."
                className="message-input"
              />
              <button onClick={handleSendMessage} disabled={!inputMessage.trim()} className="send-button">
                <Send />
              </button>
            </div>
          </>
        );
      case 'profile':
        return (
          <div className="profile-container">
            <h2>User Profile</h2>
            <div className="profile-info">
              <div className="profile-avatar">
                <UserIcon size={48} />
              </div>
              <div className="profile-details">
                <h3>John Doe</h3>
                <p>john.doe@example.com</p>
                <p>Account Type: Premium</p>
              </div>
            </div>
            <div className="profile-stats">
              <h3>Usage Statistics</h3>
              <ul>
                <li>Queries this month: 156</li>
                <li>Documents uploaded: {uploadedFiles.length}</li>
                <li>Account created: January 15, 2024</li>
              </ul>
            </div>
          </div>
        );
      case 'settings':
        return (
          <div className="settings-container">
            <h2>Settings</h2>
            <div className="settings-section">
              <h3>Document Management</h3>
              <div 
                className={`file-drop-area ${isDragging ? 'dragging' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                <FileText size={48} />
                <p>Drag and drop files here or</p>
                <input
                  type="file"
                  id="settings-file-upload"
                  className="file-input"
                  onChange={handleFileUpload}
                  accept=".pdf,.doc,.docx,.txt"
                  multiple
                />
                <label htmlFor="settings-file-upload" className="upload-button">
                  <Upload className="upload-icon" />
                  Browse Files
                </label>
              </div>
              
              <div className="uploaded-files-list">
                <h4>Uploaded Documents</h4>
                {uploadedFiles.length === 0 ? (
                  <p>No documents uploaded yet</p>
                ) : (
                  <ul>
                    {uploadedFiles.map((file, index) => (
                      <li key={index} className="file-item">
                        <FileText size={16} />
                        <span>{file.name}</span>
                        <span className="file-size">({Math.round(file.size / 1024)} KB)</span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
            
            <div className="settings-section">
              <h3>API Configuration</h3>
              <div className="form-group">
                <label>API Key</label>
                <input type="password" placeholder="Enter your API key" value="••••••••••••••••" />
              </div>
              <div className="form-group">
                <label>Model Selection</label>
                <select>
                  <option>GPT-4</option>
                  <option>Claude 3 Opus</option>
                  <option>Llama 3</option>
                </select>
              </div>
            </div>
          </div>
        );
      default:
        return <div>Select an option from the sidebar</div>;
    }
  };

  return (
    <div className="app-container">
      <div className="sidebar">
        <div className="company-logo">
          <h2>RagAI</h2>
        </div>
        <nav className="sidebar-nav">
          <button 
            className={`nav-item ${activeTab === 'dashboard' ? 'active' : ''}`}
            onClick={() => setActiveTab('dashboard')}
          >
            <Home /> Dashboard
          </button>
          <button 
            className={`nav-item ${activeTab === 'chat' ? 'active' : ''}`}
            onClick={() => setActiveTab('chat')}
          >
            <MessageCircle /> Chat
          </button>
          <button 
            className={`nav-item ${activeTab === 'profile' ? 'active' : ''}`}
            onClick={() => setActiveTab('profile')}
          >
            <UserIcon /> Profile
          </button>
          <button 
            className={`nav-item ${activeTab === 'settings' ? 'active' : ''}`}
            onClick={() => setActiveTab('settings')}
          >
            <Settings /> Settings
          </button>
        </nav>
      </div>
      
      <div className="content-area">
        <div className="content-header">
          <h1>{activeTab.charAt(0).toUpperCase() + activeTab.slice(1)}</h1>
        </div>
        
        <div className="content-body">
          {renderContent()}
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;