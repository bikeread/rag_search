import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { ChatMessage } from '@/types'
import { queryService } from '@/services/queries'
import { useAuthStore } from './authStore'

interface ChatState {
  messages: ChatMessage[]
  isLoading: boolean
  sendMessage: (content: string) => Promise<void>
  clearMessages: () => void
  loadHistory: () => Promise<void>
  clearUserData: () => void
}

// 获取当前用户的存储key
const getUserStorageKey = () => {
  const user = useAuthStore.getState().user
  return user ? `chat-storage-${user.id}` : 'chat-storage-guest'
}

export const useChatStore = create<ChatState>((set, get) => ({
  messages: [],
  isLoading: false,

  sendMessage: async (content: string) => {
    const userMessage: ChatMessage = {
      role: 'user',
      content,
      timestamp: Date.now(),
    }

    const newMessages = [...get().messages, userMessage]
    set({ messages: newMessages, isLoading: true })
    
    // 手动保存到localStorage
    localStorage.setItem(getUserStorageKey(), JSON.stringify({ messages: newMessages }))

    try {
      const response = await queryService.submitQuery({ query: content })
      
      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: response.answer,
        timestamp: Date.now(),
        sources: response.sources,
      }

      const finalMessages = [...newMessages, assistantMessage]
      set({ messages: finalMessages, isLoading: false })
      
      // 手动保存到localStorage
      localStorage.setItem(getUserStorageKey(), JSON.stringify({ messages: finalMessages }))
    } catch (error) {
      set({ isLoading: false })
      throw error
    }
  },

  clearMessages: () => {
    set({ messages: [] })
    localStorage.setItem(getUserStorageKey(), JSON.stringify({ messages: [] }))
  },

  clearUserData: () => {
    const key = getUserStorageKey()
    set({ messages: [] })
    localStorage.removeItem(key)
  },

  loadHistory: async () => {
    try {
      // 首先从localStorage加载（快速显示）
      const key = getUserStorageKey()
      const stored = localStorage.getItem(key)
      if (stored) {
        const data = JSON.parse(stored)
        set({ messages: data.messages || [] })
      }

      // 然后从服务器加载最新历史记录
      const historyResponse = await queryService.getQueryHistory({ limit: 50 })
      
      // 转换后端查询历史为前端聊天消息格式
      const messages: ChatMessage[] = []
      for (const query of historyResponse.queries) {
        // 用户消息
        messages.push({
          role: 'user',
          content: query.text,
          timestamp: new Date(query.createdAt).getTime(),
        })
        
        // AI回复（如果有）
        if (query.response) {
          messages.push({
            role: 'assistant',
            content: query.response,
            timestamp: new Date(query.createdAt).getTime() + 1000, // 稍微晚一点
            // TODO: 添加sources支持
          })
        }
      }
      
      // 更新状态和localStorage
      set({ messages })
      localStorage.setItem(key, JSON.stringify({ messages }))
      
    } catch (error) {
      console.error('Failed to load chat history:', error)
      // 保持localStorage中的数据作为fallback
    }
  },
}))