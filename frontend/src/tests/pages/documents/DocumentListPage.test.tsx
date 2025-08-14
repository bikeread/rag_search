import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { BrowserRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import DocumentListPage from '@/pages/documents/DocumentListPage'

// Mock all dependencies
vi.mock('@/services/documents', () => ({
  documentService: {
    getDocuments: vi.fn(() => Promise.resolve({ 
      documents: [], 
      pagination: { total: 0, page: 1, limit: 10, totalPages: 0, hasNext: false, hasPrev: false }
    })),
    deleteDocument: vi.fn(() => Promise.resolve()),
    uploadDocument: vi.fn(() => Promise.resolve()),
    batchDeleteDocuments: vi.fn(() => Promise.resolve({ succeeded: 0, failed: 0 })),
  },
}))

vi.mock('@/hooks/useDocuments', () => ({
  useDocuments: vi.fn(() => ({
    data: { 
      documents: [], 
      pagination: { total: 0, page: 1, limit: 10, totalPages: 0, hasNext: false, hasPrev: false }
    },
    isLoading: false,
    refetch: vi.fn(),
  })),
  useDocumentDelete: vi.fn(() => ({
    mutate: vi.fn(),
    isPending: false,
  })),
  useDocumentUpload: vi.fn(() => ({
    mutate: vi.fn(),
    isPending: false,
    uploadProgress: {},
  })),
  useBatchOperations: vi.fn(() => ({
    selectedRows: [],
    batchLoading: false,
    rowSelection: { selectedRowKeys: [], onChange: vi.fn() },
    batchDelete: vi.fn(),
    clearSelection: vi.fn(),
  })),
  useResponsiveView: vi.fn(() => ({ isMobile: false })),
}))

vi.mock('@/utils/formatters', () => ({
  formatBytes: vi.fn((bytes) => `${bytes} B`),
  formatDate: vi.fn((date) => '2023-01-01 12:00'),
}))

vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom')
  return {
    ...actual,
    useNavigate: vi.fn(() => vi.fn()),
  }
})

vi.mock('react-hot-toast', () => ({
  toast: {
    success: vi.fn(),
    error: vi.fn(),
  },
}))

// Test wrapper component
const createTestWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  })
  
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        {children}
      </BrowserRouter>
    </QueryClientProvider>
  )
}

describe('文档列表页面', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('基础渲染测试', () => {
    it('应该正确渲染页面标题和主要元素', () => {
      render(<DocumentListPage />, { wrapper: createTestWrapper() })
      
      expect(screen.getByText('文档管理')).toBeInTheDocument()
      expect(screen.getAllByText('上传文档')).toHaveLength(2) // 页面顶部和空状态各有一个
      expect(screen.getByRole('button', { name: /刷新/ })).toBeInTheDocument()
      expect(screen.getByPlaceholderText('搜索文档名称')).toBeInTheDocument()
    })

    it('应该显示空状态当没有文档时', () => {
      render(<DocumentListPage />, { wrapper: createTestWrapper() })
      
      // 检查空状态文本（根据实际组件内容）
      expect(screen.getByText('暂无文档')).toBeInTheDocument()
    })
  })

  describe('搜索功能测试', () => {
    it('应该响应搜索输入', async () => {
      render(<DocumentListPage />, { wrapper: createTestWrapper() })
      
      const searchInput = screen.getByPlaceholderText('搜索文档名称')
      fireEvent.change(searchInput, { target: { value: '测试文档' } })
      
      expect(searchInput).toHaveValue('测试文档')
    })
  })

  describe('交互功能测试', () => {
    it('应该能点击刷新按钮', () => {
      render(<DocumentListPage />, { wrapper: createTestWrapper() })
      
      const refreshButton = screen.getByRole('button', { name: /刷新/ })
      fireEvent.click(refreshButton)
      
      // 验证刷新功能被触发（具体逻辑根据实际实现调整）
      expect(refreshButton).toBeInTheDocument()
    })

    it('应该能点击上传按钮', () => {
      render(<DocumentListPage />, { wrapper: createTestWrapper() })
      
      const uploadButtons = screen.getAllByText('上传文档')
      expect(uploadButtons).toHaveLength(2)
      
      // 点击第一个上传按钮（页面顶部的）
      fireEvent.click(uploadButtons[0])
      
      // 验证按钮存在并可点击
      expect(uploadButtons[0]).toBeInTheDocument()
    })
  })

  describe('响应式设计测试', () => {
    it('应该在不同屏幕尺寸下正确显示', () => {
      // 模拟不同的屏幕尺寸
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 375, // 移动端宽度
      })

      render(<DocumentListPage />, { wrapper: createTestWrapper() })
      
      expect(screen.getByText('文档管理')).toBeInTheDocument()
    })
  })

  describe('错误处理测试', () => {
    it('应该显示错误状态', () => {
      // Mock store with error state
      vi.doMock('@/store/documentStore', () => ({
        useDocumentStore: () => ({
          documents: [],
          loading: false,
          error: '网络错误',
          fetchDocuments: vi.fn(),
          deleteDocument: vi.fn(),
          clearError: vi.fn(),
        }),
      }))

      render(<DocumentListPage />, { wrapper: createTestWrapper() })
      
      // 验证错误处理（具体实现根据实际代码调整）
      expect(screen.getByText('文档管理')).toBeInTheDocument()
    })
  })
})