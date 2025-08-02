import React from 'react';
import { NextPage } from 'next';

const HomePage: NextPage = () => {
  return (
    <div style={{ 
      padding: '40px', 
      textAlign: 'center', 
      fontFamily: 'Arial, sans-serif',
      backgroundColor: '#f5f5f5',
      minHeight: '100vh',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'center'
    }}>
      <h1 style={{ color: '#1890ff', marginBottom: '20px' }}>
        RAG System Backend API
      </h1>
      <p style={{ fontSize: '18px', color: '#666', marginBottom: '30px' }}>
        后端API服务正在运行中
      </p>
      
      <div style={{ 
        backgroundColor: 'white', 
        padding: '20px', 
        borderRadius: '8px',
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        maxWidth: '600px',
        margin: '0 auto'
      }}>
        <h3 style={{ marginBottom: '20px' }}>可用的API端点:</h3>
        <ul style={{ textAlign: 'left', lineHeight: '2' }}>
          <li><strong>健康检查:</strong> <code>/api/health</code></li>
          <li><strong>API信息:</strong> <code>/api</code></li>
          <li><strong>文档上传:</strong> <code>/api/documents/upload</code></li>
          <li><strong>查询接口:</strong> <code>/api/query</code></li>
        </ul>
      </div>

      <div style={{ marginTop: '30px' }}>
        <p style={{ color: '#999' }}>
          前端页面请访问: <a href="http://localhost:3000" style={{ color: '#1890ff' }}>http://localhost:3000</a>
        </p>
      </div>
    </div>
  );
};

export default HomePage;