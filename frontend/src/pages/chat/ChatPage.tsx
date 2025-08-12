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
    <div className="h-full flex flex-col">
      {/* 页面标题 */}
      <div className="p-4 border-b flex justify-between items-center">
        <div>
          <h1 className="text-xl font-bold">AI智能问答</h1>
          <p className="text-gray-500 text-sm mt-1">
            基于您上传的文档进行智能问答
          </p>
        </div>
        <ClearChatButton />
      </div>

      {/* 聊天消息区域 */}
      <div className="flex-1 p-4 overflow-y-auto">
        {messages.length === 0 ? (
          <Empty 
            description="开始对话吧！问我任何关于您文档的问题。"
            image={Empty.PRESENTED_IMAGE_SIMPLE}
          />
        ) : (
          <div className="space-y-4">
            {messages.map((message, index) => (
              <MessageItem key={index} message={message} />
            ))}
            
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-gray-100 p-3 rounded-lg">
                  <Spin size="small" />
                  <span className="ml-2">正在思考...</span>
                </div>
              </div>
            )}
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* 输入区域 */}
      <div className="p-4 border-t">
        <div className="flex gap-2">
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
        <div className="text-xs text-gray-500 mt-2">
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
                      {source?.content ? 
                        (source.content.substring(0, 100) + (source.content.length > 100 ? '...' : '')) :
                        '内容不可用'
                      }
                    </Text>
                    <div className="mt-1">
                      <Tag color="blue">
                        相似度: {source?.score ? (source.score * 100).toFixed(1) : '0.0'}%
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