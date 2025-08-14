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
  clearAllHistory: () => Promise<void>
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

  clearAllHistory: async () => {
    await queryService.clearAllHistory()
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
      const key = getUserStorageKey()
      
      // 首先从localStorage加载（快速显示）
      const stored = localStorage.getItem(key)
      if (stored) {
        const data = JSON.parse(stored)
        set({ messages: data.messages || [] })
      }

      // 从服务器加载历史记录（过滤已删除记录，默认逻辑已处理）
      const historyResponse = await queryService.getQueryHistory({ 
        limit: 50
        // 后端已过滤 DELETED 状态记录
      })
      
      // 如果后端返回空数据且本地有数据，说明被清空了
      if (!historyResponse.queries || historyResponse.queries.length === 0) {
        set({ messages: [] })
        localStorage.removeItem(key)
        return
      }
      
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