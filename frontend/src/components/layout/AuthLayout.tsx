import React from 'react'
import { Outlet } from 'react-router-dom'

export const AuthLayout: React.FC = () => {
  return (
    <div style={{ 
      minHeight: '100vh', 
      display: 'flex',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
    }}>
      {/* 左侧品牌区域 - PC端显示 */}
      <div 
        className="hidden lg:block"
        style={{
          width: '50%',
          background: 'linear-gradient(135deg, #007AFF 0%, #5856D6 100%)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '4rem 3rem',
          position: 'relative'
        }}
      >
        {/* 装饰性圆形背景 */}
        <div style={{
          position: 'absolute',
          top: '10%',
          right: '15%',
          width: '120px',
          height: '120px',
          borderRadius: '50%',
          background: 'rgba(255, 255, 255, 0.1)',
          backdropFilter: 'blur(10px)'
        }}></div>
        <div style={{
          position: 'absolute',
          bottom: '20%',
          left: '10%',
          width: '80px',
          height: '80px',
          borderRadius: '50%',
          background: 'rgba(255, 255, 255, 0.08)',
          backdropFilter: 'blur(10px)'
        }}></div>

        <div style={{ textAlign: 'center', color: 'white', zIndex: 1 }}>
          <div style={{
            width: '80px',
            height: '80px',
            borderRadius: '24px',
            background: 'rgba(255, 255, 255, 0.15)',
            backdropFilter: 'blur(20px)',
            margin: '0 auto 2rem',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '2rem'
          }}>
            🤖
          </div>
          <h1 style={{ 
            fontSize: '2.25rem', 
            fontWeight: '700', 
            marginBottom: '1rem',
            letterSpacing: '-0.02em',
            lineHeight: '1.2'
          }}>
            RAG智能问答
          </h1>
          <p style={{ 
            fontSize: '1.125rem', 
            opacity: 0.9, 
            marginBottom: '0',
            fontWeight: '400',
            lineHeight: '1.5'
          }}>
            让AI理解您的文档<br />智能问答，触手可及
          </p>
        </div>
      </div>

      {/* 右侧登录区域 */}
      <div 
        className="w-full lg:w-1/2"
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '3rem 2rem',
          backgroundColor: '#fbfbfd',
          background: 'linear-gradient(180deg, #fbfbfd 0%, #f7f7f9 100%)'
        }}
      >
        <div style={{ width: '100%', maxWidth: '380px' }}>
          {/* 移动端品牌信息 */}
          <div className="text-center lg:hidden mb-8">
            <div style={{
              width: '60px',
              height: '60px',
              borderRadius: '16px',
              background: 'linear-gradient(135deg, #007AFF 0%, #5856D6 100%)',
              margin: '0 auto 1rem',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '1.5rem'
            }}>
              🤖
            </div>
            <h1 style={{ 
              fontSize: '1.5rem', 
              fontWeight: '700', 
              color: '#1d1d1f', 
              marginBottom: '0.5rem',
              letterSpacing: '-0.01em'
            }}>
              RAG智能问答
            </h1>
            <p style={{ color: '#86868b', fontSize: '0.9rem' }}>让AI理解您的文档</p>
          </div>

          {/* 登录卡片 */}
          <div style={{
            backgroundColor: '#ffffff',
            borderRadius: '16px',
            padding: '2.5rem',
            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.02), 0 2px 4px -1px rgba(0, 0, 0, 0.03)',
            border: '1px solid rgba(0, 0, 0, 0.04)'
          }}>
            {/* 登录表单标题 */}
            <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
              <h2 style={{ 
                fontSize: '1.75rem', 
                fontWeight: '700', 
                color: '#1d1d1f', 
                marginBottom: '0.5rem',
                letterSpacing: '-0.02em'
              }}>
                欢迎回来
              </h2>
              <p style={{ 
                color: '#86868b', 
                fontSize: '0.9rem',
                fontWeight: '400'
              }}>
                使用您的账户信息登录系统
              </p>
            </div>

            {/* 登录表单 */}
            <Outlet />
          </div>

          {/* 底部信息 */}
          <div style={{ 
            textAlign: 'center', 
            marginTop: '2rem',
            fontSize: '0.8rem',
            color: '#86868b'
          }}>
            安全登录，数据加密传输
          </div>
        </div>
      </div>
    </div>
  )
}