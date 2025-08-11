import { create } from 'zustand'
import type { Document } from '@/types'
import { documentService } from '@/services/documents'

interface DocumentState {
  documents: Document[]
  currentDocument: Document | null
  isLoading: boolean
  uploadProgress: number
  
  loadDocuments: (params?: any) => Promise<void>
  uploadDocument: (file: File) => Promise<void>
  deleteDocument: (id: string) => Promise<void>
  setCurrentDocument: (document: Document | null) => void
  updateDocument: (id: string, updates: Partial<Document>) => void
}

export const useDocumentStore = create<DocumentState>((set, get) => ({
  documents: [],
  currentDocument: null,
  isLoading: false,
  uploadProgress: 0,

  loadDocuments: async (params = {}) => {
    set({ isLoading: true })
    try {
      const response = await documentService.getDocuments(params)
      set({ documents: response.documents, isLoading: false })
    } catch (error) {
      set({ isLoading: false })
      throw error
    }
  },

  uploadDocument: async (file: File) => {
    set({ isLoading: true, uploadProgress: 0 })
    try {
      await documentService.uploadDocument(file)
      set({ isLoading: false, uploadProgress: 100 })
      // 重新加载文档列表
      get().loadDocuments()
    } catch (error) {
      set({ isLoading: false, uploadProgress: 0 })
      throw error
    }
  },

  deleteDocument: async (id: string) => {
    try {
      await documentService.deleteDocument(id)
      set({ 
        documents: get().documents.filter(doc => doc.id !== id) 
      })
    } catch (error) {
      throw error
    }
  },

  setCurrentDocument: (document: Document | null) => {
    set({ currentDocument: document })
  },

  updateDocument: (id: string, updates: Partial<Document>) => {
    set({
      documents: get().documents.map(doc => 
        doc.id === id ? { ...doc, ...updates } : doc
      )
    })
  },
}))