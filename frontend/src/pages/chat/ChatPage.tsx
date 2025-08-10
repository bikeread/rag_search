import React, { useState, useRef, useEffect } from 'react'
import { Input, Button, Card, Spin, Typography, Tag } from 'antd'
import { SendOutlined, UserOutlined } from '@ant-design/icons'
import { MessageOutlined } from '@ant-design/icons'
import { useChatStore } from '@/store/chatStore'
import type { ChatMessage } from '@/types'
import { formatDate } from '@/utils/formatters'

const { TextArea } = Input
const { Text, Paragraph } = Typography

export const ChatPage: React.FC = () => {
  const [inputValue, setInputValue] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const { messages, isLoading, sendMessage, clearMessages } = useChatStore()

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

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

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="h-full flex flex-col">
      {/* 页面标题 */}
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">AI智能对话</h1>
        <Button onClick={clearMessages} disabled={isLoading}>
          清空对话
        </Button>
      </div>

      {/* 聊天消息区域 */}
      <div className="flex-1 bg-white rounded-lg shadow-sm border p-4 mb-4 overflow-y-auto">
        <div className="space-y-4">
          {messages.length === 0 ? (
            <div className="text-center text-gray-500 mt-8">
              <MessageOutlined className="text-4xl mb-4" />
              <p>你好！我是AI助手，有什么问题可以问我。</p>
            </div>
          ) : (
            messages.map((message, index) => (
              <MessageItem key={index} message={message} />
            ))
          )}
          
          {isLoading && (
            <div className="flex items-center space-x-2">
              <Spin size="small" />
              <Text className="text-gray-500">AI正在思考...</Text>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* 输入区域 */}
      <div className="bg-white rounded-lg shadow-sm border p-4">
        <div className="flex space-x-2">
          <TextArea
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="输入你的问题... (Shift+Enter换行，Enter发送)"
            autoSize={{ minRows: 1, maxRows: 4 }}
            disabled={isLoading}
          />
          <Button
            type="primary"
            icon={<SendOutlined />}
            onClick={handleSend}
            loading={isLoading}
            disabled={!inputValue.trim()}
          >
            发送
          </Button>
        </div>
      </div>
    </div>
  )
}

// 消息组件
const MessageItem: React.FC<{ message: ChatMessage }> = ({ message }) => {
  const isUser = message.role === 'user'

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-2xl ${isUser ? 'order-2' : 'order-1'}`}>
        <div className="flex items-center space-x-2 mb-1">
          {!isUser && <MessageOutlined className="text-blue-500" />}
          {isUser && <UserOutlined className="text-green-500" />}
          <Text className="text-sm text-gray-500">
            {isUser ? '我' : 'AI助手'}
          </Text>
          <Text className="text-xs text-gray-400">
            {formatDate(message.timestamp)}
          </Text>
        </div>
        
        <Card
          size="small"
          className={isUser ? 'bg-blue-50' : 'bg-gray-50'}
        >
          <Paragraph className="mb-0 whitespace-pre-wrap">
            {message.content}
          </Paragraph>
          
          {message.sources && message.sources.length > 0 && (
            <div className="mt-3 pt-3 border-t border-gray-200">
              <Text className="text-sm text-gray-600 mb-2 block">参考来源：</Text>
              <div className="space-y-2">
                {message.sources.map((source, index) => (
                  <div key={index} className="bg-white p-2 rounded border-l-4 border-blue-400">
                    <Text className="text-sm">
                      {source.content.substring(0, 100)}
                      {source.content.length > 100 && '...'}
                    </Text>
                    <div className="mt-1">
                      <Tag color="blue">
                        相似度: {(source.score * 100).toFixed(1)}%
                      </Tag>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </Card>
      </div>
    </div>
  )
}