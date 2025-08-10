import { create } from 'zustand'
import type { ChatMessage } from '@/types'
import { queryService } from '@/services/queries'

interface ChatState {
  messages: ChatMessage[]
  isLoading: boolean
  sendMessage: (content: string) => Promise<void>
  clearMessages: () => void
  loadHistory: () => Promise<void>
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

    set({ 
      messages: [...get().messages, userMessage],
      isLoading: true 
    })

    try {
      const response = await queryService.submitQuery({ query: content })
      
      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: response.answer,
        timestamp: Date.now(),
        sources: response.sources,
      }

      set({ 
        messages: [...get().messages, assistantMessage],
        isLoading: false 
      })
    } catch (error) {
      set({ isLoading: false })
      throw error
    }
  },

  clearMessages: () => {
    set({ messages: [] })
  },

  loadHistory: async () => {
    // 实现历史消息加载
  },
}))