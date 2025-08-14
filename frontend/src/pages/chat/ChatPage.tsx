import React, { useState, useRef, useEffect } from 'react'
import { Input, Button, Card, Spin, Typography, Tag, Empty } from 'antd'
import { SendOutlined, UserOutlined } from '@ant-design/icons'
import { MessageOutlined } from '@ant-design/icons'
import { useChatStore } from '@/store/chatStore'
import { ClearChatButton } from '@/components/chat/ClearChatButton'
import type { ChatMessage } from '@/types'
import { formatDate } from '@/utils/formatters'

const { TextArea } = Input
const { Text, Paragraph } = Typography

export const ChatPage: React.FC = () => {
  const [inputValue, setInputValue] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const { messages, isLoading, sendMessage, loadHistory } = useChatStore()

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    // 页面加载时加载用户历史记录
    loadHistory()
  }, [loadHistory])

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSend = async () => {
    if (!inputValue.trim() || isLoading) return

    const message = inputValue.trim()
    setInputValue('')
    
    try {
      await sendMessage(message)
    } catch (error) {
      console.error('Failed to send message:', error)
    }
  }


  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', background: '#f5f5f7' }}>
      {/* 页面标题 */}
      <div style={{
        padding: '16px 24px',
        background: 'rgba(255,255,255,0.95)',
        backdropFilter: 'blur(20px)',
        borderBottom: '1px solid rgba(0,0,0,0.04)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div>
          <h1 style={{
            fontSize: 28,
            fontWeight: 600,
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            marginBottom: 4
          }}>
            AI智能问答
          </h1>
          <p style={{ 
            color: '#6e6e73', 
            fontSize: 14,
            margin: 0
          }}>
            基于您上传的文档进行智能问答
          </p>
        </div>
        <ClearChatButton />
      </div>

      {/* 聊天消息区域 */}
      <div style={{
        flex: 1,
        padding: '24px',
        overflowY: 'auto',
        display: 'flex',
        flexDirection: 'column'
      }}>
        {messages.length === 0 ? (
          <div style={{
            flex: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <Empty 
              description={
                <div style={{ textAlign: 'center' }}>
                  <p style={{ fontSize: 18, fontWeight: 500, color: '#1d1d1f', marginBottom: 8 }}>
                    开始对话吧！
                  </p>
                  <p style={{ fontSize: 14, color: '#6e6e73' }}>
                    问我任何关于您文档的问题
                  </p>
                </div>
              }
              image={Empty.PRESENTED_IMAGE_SIMPLE}
            />
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            {messages.map((message, index) => (
              <MessageItem key={index} message={message} />
            ))}
            
            {isLoading && (
              <div style={{ display: 'flex', justifyContent: 'flex-start' }}>
                <div style={{
                  background: 'rgba(255,255,255,0.95)',
                  backdropFilter: 'blur(20px)',
                  padding: '16px 20px',
                  borderRadius: 16,
                  boxShadow: '0 4px 24px rgba(0,0,0,0.08)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 12
                }}>
                  <Spin size="small" />
                  <span style={{ color: '#6e6e73', fontSize: 14 }}>正在思考...</span>
                </div>
              </div>
            )}
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* 输入区域 */}
      <div style={{
        padding: '16px 24px',
        background: 'rgba(255,255,255,0.95)',
        backdropFilter: 'blur(20px)',
        borderTop: '1px solid rgba(0,0,0,0.04)'
      }}>
        <div style={{ display: 'flex', gap: 12, alignItems: 'end' }}>
          <Input.TextArea
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="输入您的问题..."
            autoSize={{ minRows: 1, maxRows: 3 }}
            onPressEnter={(e) => {
              if (!e.shiftKey) {
                e.preventDefault()
                handleSend()
              }
            }}
            disabled={isLoading}
            style={{
              borderRadius: 12,
              border: '1px solid #d0d0d3',
              background: '#ffffff',
              fontSize: 16,
              boxShadow: 'none'
            }}
          />
          <Button
            type="primary"
            icon={<SendOutlined />}
            onClick={handleSend}
            loading={isLoading}
            disabled={!inputValue.trim()}
            style={{
              borderRadius: 12,
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              border: 'none',
              height: 44,
              paddingLeft: 20,
              paddingRight: 20,
              fontWeight: 500,
              boxShadow: '0 4px 16px rgba(102, 126, 234, 0.3)'
            }}
          >
            发送
          </Button>
        </div>
        <div style={{ 
          fontSize: 12, 
          color: '#86868b',
          marginTop: 8,
          textAlign: 'center'
        }}>
          按 Enter 发送，Shift + Enter 换行
        </div>
      </div>
    </div>
  )
}

// 消息组件
const MessageItem: React.FC<{ message: ChatMessage }> = ({ message }) => {
  const isUser = message.role === 'user'

  return (
    <div style={{ 
      display: 'flex',
      justifyContent: isUser ? 'flex-end' : 'flex-start',
      width: '100%'
    }}>
      <div style={{ 
        maxWidth: '75%',
        minWidth: '200px'
      }}>
        <div style={{ 
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          marginBottom: 8,
          justifyContent: isUser ? 'flex-end' : 'flex-start'
        }}>
          {!isUser && (
            <div style={{
              width: 32,
              height: 32,
              borderRadius: 16,
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <MessageOutlined style={{ color: 'white', fontSize: 16 }} />
            </div>
          )}
          {isUser && (
            <div style={{
              width: 32,
              height: 32,
              borderRadius: 16,
              background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <UserOutlined style={{ color: 'white', fontSize: 16 }} />
            </div>
          )}
          <Text style={{ fontSize: 14, fontWeight: 500, color: '#1d1d1f' }}>
            {isUser ? '我' : 'AI助手'}
          </Text>
          <Text style={{ fontSize: 12, color: '#86868b' }}>
            {formatDate(message.timestamp)}
          </Text>
        </div>
        
        <div style={{
          background: isUser 
            ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
            : 'rgba(255,255,255,0.95)',
          backdropFilter: 'blur(20px)',
          borderRadius: 16,
          padding: '16px 20px',
          boxShadow: '0 4px 24px rgba(0,0,0,0.08)',
          color: isUser ? 'white' : '#1d1d1f'
        }}>
          <div style={{ 
            fontSize: 15,
            lineHeight: 1.5,
            whiteSpace: 'pre-wrap'
          }}>
            {message.content}
          </div>
          
          {message.sources && message.sources.length > 0 && (
            <div style={{ marginTop: 16, paddingTop: 16, borderTop: '1px solid rgba(0,0,0,0.08)' }}>
              <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 12, opacity: 0.8 }}>
                参考来源：
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {message.sources.map((source, index) => (
                  <div key={index} style={{
                    background: 'rgba(255,255,255,0.1)',
                    borderRadius: 8,
                    padding: 12,
                    border: '1px solid rgba(255,255,255,0.1)'
                  }}>
                    <div style={{ fontSize: 13, lineHeight: 1.4, marginBottom: 8, opacity: 0.9 }}>
                      {source?.content ? 
                        (source.content.substring(0, 100) + (source.content.length > 100 ? '...' : '')) :
                        '内容不可用'
                      }
                    </div>
                    <div style={{
                      fontSize: 11,
                      background: 'rgba(255,255,255,0.15)',
                      borderRadius: 4,
                      padding: '2px 8px',
                      display: 'inline-block'
                    }}>
                      相似度: {source?.score ? (source.score * 100).toFixed(1) : '0.0'}%
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}