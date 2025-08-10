# 阶段4执行指导文档：React前端开发

## 📋 总览信息

**目标**: 构建现代化用户界面，完成RAG系统的前端体验层
**预计时间**: 10-14天 (2-2.5周)
**执行策略**: 组件化开发 (基础组件→页面→集成→优化)

## 🎯 核心目标

- ✅ **React应用架构** - 完整的React 18 + TypeScript项目
- ✅ **Ant Design UI** - 现代化、响应式用户界面
- ✅ **状态管理** - Zustand全局状态 + React Query数据获取
- ✅ **路由系统** - React Router完整页面导航
- ✅ **认证集成** - 与NextAuth后端完整对接
- ✅ **核心功能** - 文档上传、RAG聊天、文档管理
- ✅ **响应式设计** - 支持桌面端和移动端
- ✅ **性能优化** - 代码分割、懒加载、缓存策略

---

## 🏗️ 子阶段拆分执行计划

### 子阶段4.1：项目架构搭建 (2-3天) 🚀

**状态**: 🔴 待开始  
**优先级**: 🔥 最高 (阻塞其他任务)  
**依赖**: 阶段3完成 (Next.js后端API就绪)

#### 📁 目录结构设计

```bash
frontend/                          # React前端项目
├── public/
│   ├── index.html
│   ├── favicon.ico
│   └── logo192.png
├── src/
│   ├── components/                # 通用UI组件
│   │   ├── layout/                # 布局组件
│   │   │   ├── Header.tsx
│   │   │   ├── Sidebar.tsx
│   │   │   ├── Footer.tsx
│   │   │   └── MainLayout.tsx
│   │   ├── common/                # 通用组件
│   │   │   ├── Loading.tsx
│   │   │   ├── ErrorBoundary.tsx
│   │   │   ├── ConfirmDialog.tsx
│   │   │   └── PageHeader.tsx
│   │   ├── forms/                 # 表单组件
│   │   │   ├── LoginForm.tsx
│   │   │   ├── RegisterForm.tsx
│   │   │   └── SearchForm.tsx
│   │   └── ui/                    # 基础UI组件
│   │       ├── Button.tsx
│   │       ├── Input.tsx
│   │       └── Card.tsx
│   ├── pages/                     # 页面组件
│   │   ├── auth/                  # 认证页面
│   │   │   ├── LoginPage.tsx
│   │   │   ├── RegisterPage.tsx
│   │   │   └── ProfilePage.tsx
│   │   ├── documents/             # 文档管理
│   │   │   ├── DocumentListPage.tsx
│   │   │   ├── DocumentUploadPage.tsx
│   │   │   ├── DocumentDetailPage.tsx
│   │   │   └── DocumentManagePage.tsx
│   │   ├── chat/                  # RAG聊天
│   │   │   ├── ChatPage.tsx
│   │   │   ├── ChatHistory.tsx
│   │   │   └── ChatInterface.tsx
│   │   ├── dashboard/             # 仪表板
│   │   │   ├── DashboardPage.tsx
│   │   │   ├── StatisticsCard.tsx
│   │   │   └── RecentActivity.tsx
│   │   ├── admin/                 # 管理员页面
│   │   │   ├── UserManagement.tsx
│   │   │   ├── SystemStatus.tsx
│   │   │   └── AdminDashboard.tsx
│   │   └── HomePage.tsx           # 首页
│   ├── hooks/                     # 自定义Hook
│   │   ├── useAuth.ts             # 认证相关
│   │   ├── useDocuments.ts        # 文档管理
│   │   ├── useQuery.ts            # RAG查询
│   │   ├── useLocalStorage.ts     # 本地存储
│   │   └── useWebSocket.ts        # 实时通信
│   ├── services/                  # API服务
│   │   ├── api.ts                 # API客户端
│   │   ├── auth.ts                # 认证服务
│   │   ├── documents.ts           # 文档服务
│   │   ├── queries.ts             # 查询服务
│   │   └── admin.ts               # 管理服务
│   ├── store/                     # 状态管理
│   │   ├── authStore.ts           # 认证状态
│   │   ├── documentStore.ts       # 文档状态
│   │   ├── chatStore.ts           # 聊天状态
│   │   └── appStore.ts            # 应用全局状态
│   ├── types/                     # TypeScript类型
│   │   ├── auth.ts                # 认证类型
│   │   ├── document.ts            # 文档类型
│   │   ├── query.ts               # 查询类型
│   │   └── api.ts                 # API类型
│   ├── utils/                     # 工具函数
│   │   ├── constants.ts           # 常量定义
│   │   ├── helpers.ts             # 通用工具
│   │   ├── validation.ts          # 表单验证
│   │   ├── formatters.ts          # 数据格式化
│   │   └── storage.ts             # 存储工具
│   ├── styles/                    # 样式文件
│   │   ├── global.css             # 全局样式
│   │   ├── variables.css          # CSS变量
│   │   ├── components.css         # 组件样式
│   │   └── responsive.css         # 响应式样式
│   ├── App.tsx                    # 应用根组件
│   ├── index.tsx                  # 应用入口
│   └── setupTests.ts              # 测试配置
├── package.json
├── tsconfig.json
├── tailwind.config.js
├── vite.config.ts
└── .env.example
```

#### 🎯 具体任务清单

**Task 4.1.1: React项目初始化**

```bash
# 创建React项目 (使用Vite获得更好的性能)
npm create vite@latest frontend -- --template react-ts

# 安装核心依赖
npm install antd @ant-design/icons
npm install react-router-dom react-query zustand
npm install axios @types/node
npm install @tanstack/react-query @tanstack/react-query-devtools

# 安装UI和工具库
npm install dayjs clsx tailwindcss
npm install react-helmet-async react-hot-toast
npm install framer-motion lucide-react

# 安装开发依赖
npm install -D @types/react @types/react-dom
npm install -D eslint @typescript-eslint/parser @typescript-eslint/eslint-plugin
npm install -D prettier eslint-plugin-prettier
npm install -D @vitejs/plugin-react
```

**Task 4.1.2: 基础配置文件**

```typescript
// vite.config.ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:3001',
        changeOrigin: true,
      },
    },
  },
})

// tailwind.config.js
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#eff6ff',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
        },
        gray: {
          50: '#f9fafb',
          100: '#f3f4f6',
          900: '#111827',
        },
      },
    },
  },
  plugins: [],
}

// .env.example
VITE_API_BASE_URL=http://localhost:3001
VITE_APP_TITLE=RAG智能问答系统
VITE_APP_VERSION=1.0.0
VITE_ENABLE_DEVTOOLS=true
```

**Task 4.1.3: TypeScript类型定义**

```typescript
// src/types/api.ts
export interface ApiResponse<T = any> {
  success: boolean
  data?: T
  error?: string
  message?: string
  timestamp: string
}

export interface PaginatedResponse<T> {
  data: T[]
  pagination: {
    page: number
    limit: number
    total: number
    totalPages: number
    hasNext: boolean
    hasPrev: boolean
  }
}

// src/types/auth.ts
export interface User {
  id: string
  email: string
  name?: string
  role: 'USER' | 'ADMIN' | 'SUPER_ADMIN'
  avatar?: string
  isActive: boolean
  createdAt: string
  updatedAt: string
}

export interface LoginRequest {
  email: string
  password: string
}

export interface RegisterRequest {
  email: string
  password: string
  name?: string
}

// src/types/document.ts
export interface Document {
  id: string
  filename: string
  originalName: string
  mimeType: string
  size: number
  status: 'PENDING' | 'PROCESSING' | 'COMPLETED' | 'FAILED' | 'DELETED'
  chunksCount: number
  createdAt: string
  updatedAt: string
  processingStartedAt?: string
  processingCompletedAt?: string
  errorMessage?: string
}

export interface DocumentUploadRequest {
  file: File
}

// src/types/query.ts
export interface QueryRequest {
  query: string
  topK?: number
  useCache?: boolean
}

export interface QueryResponse {
  queryId: string
  answer: string
  sources: QuerySource[]
  responseTime: number
  metadata: {
    topK: number
    searchTime: number
    sourcesCount: number
  }
  cached?: boolean
}

export interface QuerySource {
  id: string
  content: string
  score: number
  metadata?: Record<string, any>
}

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  timestamp: number
  sources?: QuerySource[]
}
```

#### ✅ 验收标准4.1

- [ ] React项目成功创建并可启动 (npm run dev)
- [ ] TypeScript配置正确，无编译错误
- [ ] Tailwind CSS和Ant Design正确集成
- [ ] 路径别名 (@/) 正常工作
- [ ] 基础目录结构创建完成
- [ ] 环境变量配置正确

---

### 子阶段4.2：核心服务层搭建 (2-3天) 🔧

**状态**: 🔴 待开始  
**优先级**: 🔥 高  
**依赖**: 子阶段4.1完成

#### 🎯 具体任务清单

**Task 4.2.1: API服务客户端**

```typescript
// src/services/api.ts
import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios'
import { toast } from 'react-hot-toast'

class ApiClient {
  private client: AxiosInstance

  constructor() {
    this.client = axios.create({
      baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:3001',
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    })

    this.setupInterceptors()
  }

  private setupInterceptors() {
    // 请求拦截器
    this.client.interceptors.request.use(
      (config) => {
        const token = localStorage.getItem('token')
        if (token) {
          config.headers.Authorization = `Bearer ${token}`
        }
        return config
      },
      (error) => Promise.reject(error)
    )

    // 响应拦截器
    this.client.interceptors.response.use(
      (response: AxiosResponse) => response,
      (error) => {
        if (error.response?.status === 401) {
          localStorage.removeItem('token')
          window.location.href = '/login'
        } else if (error.response?.status >= 500) {
          toast.error('服务器错误，请稍后重试')
        }
        return Promise.reject(error)
      }
    )
  }

  async get<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.get(url, config)
    return response.data
  }

  async post<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.post(url, data, config)
    return response.data
  }

  async put<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.put(url, data, config)
    return response.data
  }

  async delete<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.delete(url, config)
    return response.data
  }
}

export const apiClient = new ApiClient()

// src/services/auth.ts
import { ApiResponse, LoginRequest, RegisterRequest, User } from '@/types'
import { apiClient } from './api'

export const authService = {
  async login(data: LoginRequest): Promise<{ user: User; token: string }> {
    return apiClient.post('/api/auth/signin', data)
  },

  async register(data: RegisterRequest): Promise<{ user: User }> {
    return apiClient.post('/api/auth/register', data)
  },

  async getCurrentUser(): Promise<User> {
    return apiClient.get('/api/auth/user')
  },

  async logout(): Promise<void> {
    return apiClient.post('/api/auth/signout')
  },
}

// src/services/documents.ts
import { Document, DocumentUploadRequest, PaginatedResponse } from '@/types'
import { apiClient } from './api'

export const documentService = {
  async uploadDocument(data: DocumentUploadRequest): Promise<{ documentId: string }> {
    const formData = new FormData()
    formData.append('file', data.file)
    
    return apiClient.post('/api/documents/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
  },

  async getDocumentList(params: {
    page?: number
    limit?: number
    status?: string
    search?: string
  }): Promise<PaginatedResponse<Document>> {
    return apiClient.get('/api/documents/list', { params })
  },

  async getDocumentStatus(id: string): Promise<{ document: Document }> {
    return apiClient.get(`/api/documents/status/${id}`)
  },

  async deleteDocument(id: string): Promise<void> {
    return apiClient.delete(`/api/documents/delete/${id}`)
  },
}

// src/services/queries.ts
import { QueryRequest, QueryResponse, PaginatedResponse } from '@/types'
import { apiClient } from './api'

export const queryService = {
  async submitQuery(data: QueryRequest): Promise<QueryResponse> {
    return apiClient.post('/api/query', data)
  },

  async getQueryHistory(params: {
    page?: number
    limit?: number
    status?: string
  }): Promise<PaginatedResponse<any>> {
    return apiClient.get('/api/query/history', { params })
  },

  async chatQuery(messages: Array<{ role: string; content: string }>): Promise<any> {
    return apiClient.post('/api/query/chat', { messages })
  },
}
```

**Task 4.2.2: Zustand状态管理**

```typescript
// src/store/authStore.ts
import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { User } from '@/types'
import { authService } from '@/services/auth'

interface AuthState {
  user: User | null
  token: string | null
  isLoading: boolean
  isAuthenticated: boolean
  login: (email: string, password: string) => Promise<void>
  register: (data: { email: string; password: string; name?: string }) => Promise<void>
  logout: () => void
  loadUser: () => Promise<void>
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      isLoading: false,
      isAuthenticated: false,

      login: async (email: string, password: string) => {
        set({ isLoading: true })
        try {
          const response = await authService.login({ email, password })
          set({
            user: response.user,
            token: response.token,
            isAuthenticated: true,
            isLoading: false,
          })
          localStorage.setItem('token', response.token)
        } catch (error) {
          set({ isLoading: false })
          throw error
        }
      },

      register: async (data) => {
        set({ isLoading: true })
        try {
          await authService.register(data)
          set({ isLoading: false })
        } catch (error) {
          set({ isLoading: false })
          throw error
        }
      },

      logout: () => {
        localStorage.removeItem('token')
        set({
          user: null,
          token: null,
          isAuthenticated: false,
        })
      },

      loadUser: async () => {
        const token = localStorage.getItem('token')
        if (token) {
          try {
            const user = await authService.getCurrentUser()
            set({
              user,
              token,
              isAuthenticated: true,
            })
          } catch {
            get().logout()
          }
        }
      },
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({ token: state.token }),
    }
  )
)

// src/store/documentStore.ts
import { create } from 'zustand'
import { Document } from '@/types'
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
      const response = await documentService.getDocumentList(params)
      set({ documents: response.data, isLoading: false })
    } catch (error) {
      set({ isLoading: false })
      throw error
    }
  },

  uploadDocument: async (file: File) => {
    set({ isLoading: true, uploadProgress: 0 })
    try {
      const response = await documentService.uploadDocument({ file })
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

// src/store/chatStore.ts
import { create } from 'zustand'
import { ChatMessage, QueryResponse } from '@/types'
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
```

**Task 4.2.3: React Query集成**

```typescript
// src/hooks/useQuery.ts
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { queryService } from '@/services/queries'
import { QueryRequest } from '@/types'

export const useSubmitQuery = () => {
  return useMutation({
    mutationFn: (data: QueryRequest) => queryService.submitQuery(data),
    onSuccess: () => {
      // 可以在这里处理成功逻辑
    },
  })
}

export const useQueryHistory = (params: { page?: number; limit?: number } = {}) => {
  return useQuery({
    queryKey: ['queryHistory', params],
    queryFn: () => queryService.getQueryHistory(params),
    staleTime: 5 * 60 * 1000, // 5分钟
  })
}

// src/hooks/useDocuments.ts
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { documentService } from '@/services/documents'
import { toast } from 'react-hot-toast'

export const useDocuments = (params: any = {}) => {
  return useQuery({
    queryKey: ['documents', params],
    queryFn: () => documentService.getDocumentList(params),
    staleTime: 2 * 60 * 1000, // 2分钟
  })
}

export const useDocumentUpload = () => {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: documentService.uploadDocument,
    onSuccess: () => {
      toast.success('文档上传成功！')
      queryClient.invalidateQueries({ queryKey: ['documents'] })
    },
    onError: (error: any) => {
      toast.error(error.message || '上传失败')
    },
  })
}

export const useDocumentDelete = () => {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: documentService.deleteDocument,
    onSuccess: () => {
      toast.success('文档删除成功！')
      queryClient.invalidateQueries({ queryKey: ['documents'] })
    },
    onError: (error: any) => {
      toast.error(error.message || '删除失败')
    },
  })
}

// src/hooks/useAuth.ts
import { useAuthStore } from '@/store/authStore'
import { useMutation } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { toast } from 'react-hot-toast'

export const useAuth = () => {
  const navigate = useNavigate()
  const { user, isAuthenticated, login, register, logout } = useAuthStore()

  const loginMutation = useMutation({
    mutationFn: ({ email, password }: { email: string; password: string }) =>
      login(email, password),
    onSuccess: () => {
      toast.success('登录成功！')
      navigate('/dashboard')
    },
    onError: (error: any) => {
      toast.error(error.message || '登录失败')
    },
  })

  const registerMutation = useMutation({
    mutationFn: register,
    onSuccess: () => {
      toast.success('注册成功！请登录')
      navigate('/login')
    },
    onError: (error: any) => {
      toast.error(error.message || '注册失败')
    },
  })

  const logoutHandler = () => {
    logout()
    toast.success('已退出登录')
    navigate('/login')
  }

  return {
    user,
    isAuthenticated,
    login: loginMutation,
    register: registerMutation,
    logout: logoutHandler,
  }
}
```

#### ✅ 验收标准4.2

- [ ] API客户端正确配置，可与后端通信
- [ ] Zustand状态管理正常工作
- [ ] React Query数据获取和缓存正常
- [ ] 自定义Hooks功能完整
- [ ] 认证流程完整（登录、注册、退出）
- [ ] 错误处理和Toast通知正常

---

### 子阶段4.3：核心页面开发 (3-4天) 🎨

**状态**: 🔴 待开始  
**优先级**: ⚡ 中高  
**依赖**: 子阶段4.2完成

#### 🎯 具体任务清单

**Task 4.3.1: 布局和导航组件**

```typescript
// src/components/layout/MainLayout.tsx
import React from 'react'
import { Layout, Menu, Avatar, Dropdown, Space, Typography } from 'antd'
import { 
  UserOutlined, 
  FileTextOutlined, 
  MessageOutlined, 
  DashboardOutlined,
  SettingOutlined,
  LogoutOutlined 
} from '@ant-design/icons'
import { useNavigate, useLocation, Outlet } from 'react-router-dom'
import { useAuth } from '@/hooks/useAuth'

const { Header, Sider, Content } = Layout
const { Text } = Typography

export const MainLayout: React.FC = () => {
  const navigate = useNavigate()
  const location = useLocation()
  const { user, logout } = useAuth()

  const menuItems = [
    {
      key: '/dashboard',
      icon: <DashboardOutlined />,
      label: '仪表板',
    },
    {
      key: '/chat',
      icon: <MessageOutlined />,
      label: 'AI对话',
    },
    {
      key: '/documents',
      icon: <FileTextOutlined />,
      label: '文档管理',
    },
    {
      key: '/profile',
      icon: <UserOutlined />,
      label: '个人中心',
    },
  ]

  const userMenu = (
    <Menu
      items={[
        {
          key: 'profile',
          icon: <UserOutlined />,
          label: '个人设置',
          onClick: () => navigate('/profile'),
        },
        {
          type: 'divider',
        },
        {
          key: 'logout',
          icon: <LogoutOutlined />,
          label: '退出登录',
          onClick: logout,
        },
      ]}
    />
  )

  return (
    <Layout className="min-h-screen">
      <Sider width={240} className="bg-white shadow-sm">
        <div className="h-16 flex items-center justify-center border-b">
          <Text className="text-xl font-bold text-blue-600">
            RAG智能问答
          </Text>
        </div>
        <Menu
          mode="inline"
          selectedKeys={[location.pathname]}
          items={menuItems}
          className="border-none"
          onClick={({ key }) => navigate(key)}
        />
      </Sider>

      <Layout>
        <Header className="bg-white shadow-sm px-6 flex items-center justify-between">
          <div className="text-lg font-medium">
            {menuItems.find(item => item.key === location.pathname)?.label || '首页'}
          </div>
          
          <Dropdown overlay={userMenu} placement="bottomRight">
            <Space className="cursor-pointer">
              <Avatar src={user?.avatar} icon={<UserOutlined />} />
              <Text>{user?.name || user?.email}</Text>
            </Space>
          </Dropdown>
        </Header>

        <Content className="p-6 bg-gray-50">
          <Outlet />
        </Content>
      </Layout>
    </Layout>
  )
}

// src/components/layout/AuthLayout.tsx
import React from 'react'
import { Layout, Card } from 'antd'
import { Outlet } from 'react-router-dom'

const { Content } = Layout

export const AuthLayout: React.FC = () => {
  return (
    <Layout className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <Content className="flex items-center justify-center p-6">
        <Card className="w-full max-w-md shadow-lg">
          <div className="text-center mb-8">
            <h1 className="text-2xl font-bold text-gray-800">RAG智能问答系统</h1>
            <p className="text-gray-600 mt-2">智能文档问答，让知识触手可及</p>
          </div>
          <Outlet />
        </Card>
      </Content>
    </Layout>
  )
}
```

**Task 4.3.2: 认证页面**

```typescript
// src/pages/auth/LoginPage.tsx
import React from 'react'
import { Form, Input, Button, Checkbox, Divider } from 'antd'
import { UserOutlined, LockOutlined } from '@ant-design/icons'
import { Link } from 'react-router-dom'
import { useAuth } from '@/hooks/useAuth'

export const LoginPage: React.FC = () => {
  const { login } = useAuth()

  const onFinish = (values: { email: string; password: string }) => {
    login.mutate(values)
  }

  return (
    <Form
      name="login"
      onFinish={onFinish}
      autoComplete="off"
      size="large"
    >
      <Form.Item
        name="email"
        rules={[
          { required: true, message: '请输入邮箱地址!' },
          { type: 'email', message: '请输入有效的邮箱地址!' }
        ]}
      >
        <Input 
          prefix={<UserOutlined />} 
          placeholder="邮箱地址" 
        />
      </Form.Item>

      <Form.Item
        name="password"
        rules={[{ required: true, message: '请输入密码!' }]}
      >
        <Input.Password
          prefix={<LockOutlined />}
          placeholder="密码"
        />
      </Form.Item>

      <Form.Item name="remember" valuePropName="checked">
        <Checkbox>记住我</Checkbox>
      </Form.Item>

      <Form.Item>
        <Button
          type="primary"
          htmlType="submit"
          loading={login.isPending}
          className="w-full"
        >
          登录
        </Button>
      </Form.Item>

      <Divider>或</Divider>

      <div className="text-center">
        <span className="text-gray-600">还没有账户？</span>
        <Link to="/register" className="text-blue-600 hover:text-blue-800 ml-1">
          立即注册
        </Link>
      </div>
    </Form>
  )
}

// src/pages/auth/RegisterPage.tsx
import React from 'react'
import { Form, Input, Button, Divider } from 'antd'
import { UserOutlined, LockOutlined, MailOutlined } from '@ant-design/icons'
import { Link } from 'react-router-dom'
import { useAuth } from '@/hooks/useAuth'

export const RegisterPage: React.FC = () => {
  const { register } = useAuth()

  const onFinish = (values: { email: string; password: string; name?: string }) => {
    register.mutate(values)
  }

  return (
    <Form
      name="register"
      onFinish={onFinish}
      autoComplete="off"
      size="large"
    >
      <Form.Item
        name="name"
        rules={[{ required: true, message: '请输入姓名!' }]}
      >
        <Input 
          prefix={<UserOutlined />} 
          placeholder="姓名" 
        />
      </Form.Item>

      <Form.Item
        name="email"
        rules={[
          { required: true, message: '请输入邮箱地址!' },
          { type: 'email', message: '请输入有效的邮箱地址!' }
        ]}
      >
        <Input 
          prefix={<MailOutlined />} 
          placeholder="邮箱地址" 
        />
      </Form.Item>

      <Form.Item
        name="password"
        rules={[
          { required: true, message: '请输入密码!' },
          { min: 8, message: '密码至少8个字符!' }
        ]}
      >
        <Input.Password
          prefix={<LockOutlined />}
          placeholder="密码 (至少8个字符)"
        />
      </Form.Item>

      <Form.Item
        name="confirmPassword"
        dependencies={['password']}
        rules={[
          { required: true, message: '请确认密码!' },
          ({ getFieldValue }) => ({
            validator(_, value) {
              if (!value || getFieldValue('password') === value) {
                return Promise.resolve()
              }
              return Promise.reject(new Error('两次输入的密码不一致!'))
            },
          }),
        ]}
      >
        <Input.Password
          prefix={<LockOutlined />}
          placeholder="确认密码"
        />
      </Form.Item>

      <Form.Item>
        <Button
          type="primary"
          htmlType="submit"
          loading={register.isPending}
          className="w-full"
        >
          注册
        </Button>
      </Form.Item>

      <Divider>或</Divider>

      <div className="text-center">
        <span className="text-gray-600">已有账户？</span>
        <Link to="/login" className="text-blue-600 hover:text-blue-800 ml-1">
          立即登录
        </Link>
      </div>
    </Form>
  )
}
```

**Task 4.3.3: 文档管理页面**

```typescript
// src/pages/documents/DocumentListPage.tsx
import React, { useState } from 'react'
import { 
  Table, 
  Button, 
  Input, 
  Select, 
  Space, 
  Tag, 
  Modal,
  message,
  Upload,
  Progress
} from 'antd'
import { 
  UploadOutlined, 
  SearchOutlined, 
  DeleteOutlined,
  EyeOutlined,
  ReloadOutlined
} from '@ant-design/icons'
import { useDocuments, useDocumentDelete } from '@/hooks/useDocuments'
import { useDocumentStore } from '@/store/documentStore'
import { Document } from '@/types'
import { formatBytes, formatDate } from '@/utils/formatters'

const { Search } = Input
const { Option } = Select

export const DocumentListPage: React.FC = () => {
  const [searchText, setSearchText] = useState('')
  const [statusFilter, setStatusFilter] = useState<string | undefined>()
  const [page, setPage] = useState(1)
  const limit = 10

  const { data: documentsData, isLoading, refetch } = useDocuments({
    page,
    limit,
    search: searchText,
    status: statusFilter,
  })

  const deleteDocument = useDocumentDelete()
  const { uploadDocument, uploadProgress } = useDocumentStore()

  const columns = [
    {
      title: '文档名称',
      dataIndex: 'originalName',
      key: 'originalName',
      ellipsis: true,
    },
    {
      title: '文件大小',
      dataIndex: 'size',
      key: 'size',
      render: (size: number) => formatBytes(size),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const statusConfig = {
          PENDING: { color: 'orange', text: '等待处理' },
          PROCESSING: { color: 'blue', text: '处理中' },
          COMPLETED: { color: 'green', text: '已完成' },
          FAILED: { color: 'red', text: '处理失败' },
        }
        const config = statusConfig[status as keyof typeof statusConfig]
        return <Tag color={config?.color}>{config?.text}</Tag>
      },
    },
    {
      title: '文档块数',
      dataIndex: 'chunksCount',
      key: 'chunksCount',
    },
    {
      title: '上传时间',
      dataIndex: 'createdAt',
      key: 'createdAt',
      render: (date: string) => formatDate(date),
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record: Document) => (
        <Space>
          <Button
            type="link"
            icon={<EyeOutlined />}
            onClick={() => handleViewDocument(record)}
          >
            查看
          </Button>
          <Button
            type="link"
            danger
            icon={<DeleteOutlined />}
            onClick={() => handleDeleteDocument(record)}
          >
            删除
          </Button>
        </Space>
      ),
    },
  ]

  const handleViewDocument = (document: Document) => {
    // 实现查看文档详情
    console.log('View document:', document)
  }

  const handleDeleteDocument = (document: Document) => {
    Modal.confirm({
      title: '确认删除',
      content: `确定要删除文档"${document.originalName}"吗？`,
      onOk: () => deleteDocument.mutate(document.id),
    })
  }

  const handleUpload = async (file: File) => {
    try {
      await uploadDocument(file)
      refetch()
    } catch (error) {
      message.error('上传失败')
    }
  }

  return (
    <div className="space-y-6">
      {/* 页面标题和操作 */}
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">文档管理</h1>
        <Upload
          beforeUpload={(file) => {
            handleUpload(file)
            return false
          }}
          showUploadList={false}
          accept=".pdf,.docx,.doc,.txt,.md"
        >
          <Button type="primary" icon={<UploadOutlined />}>
            上传文档
          </Button>
        </Upload>
      </div>

      {/* 上传进度 */}
      {uploadProgress > 0 && (
        <Progress percent={uploadProgress} />
      )}

      {/* 搜索和过滤 */}
      <div className="flex gap-4">
        <Search
          placeholder="搜索文档名称"
          value={searchText}
          onChange={(e) => setSearchText(e.target.value)}
          onSearch={() => refetch()}
          style={{ width: 300 }}
        />
        <Select
          placeholder="筛选状态"
          value={statusFilter}
          onChange={setStatusFilter}
          allowClear
          style={{ width: 150 }}
        >
          <Option value="PENDING">等待处理</Option>
          <Option value="PROCESSING">处理中</Option>
          <Option value="COMPLETED">已完成</Option>
          <Option value="FAILED">处理失败</Option>
        </Select>
        <Button icon={<ReloadOutlined />} onClick={() => refetch()}>
          刷新
        </Button>
      </div>

      {/* 文档表格 */}
      <Table
        columns={columns}
        dataSource={documentsData?.data || []}
        rowKey="id"
        loading={isLoading}
        pagination={{
          current: page,
          pageSize: limit,
          total: documentsData?.pagination.total || 0,
          onChange: setPage,
        }}
      />
    </div>
  )
}
```

**Task 4.3.4: RAG聊天界面**

```typescript
// src/pages/chat/ChatPage.tsx
import React, { useState, useRef, useEffect } from 'react'
import { Input, Button, Card, Spin, Typography, Space, Tag } from 'antd'
import { SendOutlined, UserOutlined, RobotOutlined } from '@ant-design/icons'
import { useChatStore } from '@/store/chatStore'
import { ChatMessage } from '@/types'
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
              <RobotOutlined className="text-4xl mb-4" />
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
          {!isUser && <RobotOutlined className="text-blue-500" />}
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
                      <Tag color="blue" size="small">
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
```

#### ✅ 验收标准4.3

- [ ] 布局组件正确渲染，导航功能正常
- [ ] 认证页面完整，登录注册功能正常
- [ ] 文档管理页面功能完整（列表、上传、删除）
- [ ] RAG聊天界面交互流畅，消息显示正确
- [ ] 页面响应式设计，移动端适配良好
- [ ] 组件复用性好，代码结构清晰

---

### 子阶段4.4：用户体验优化 (2-3天) ✨

**状态**: 🔴 待开始  
**优先级**: ⚡ 中等  
**依赖**: 子阶段4.3完成

#### 🎯 具体任务清单

**Task 4.4.1: 仪表板页面**

```typescript
// src/pages/dashboard/DashboardPage.tsx
import React from 'react'
import { Row, Col, Card, Statistic, Progress, List, Avatar } from 'antd'
import { 
  FileTextOutlined, 
  MessageOutlined, 
  ClockCircleOutlined,
  CheckCircleOutlined 
} from '@ant-design/icons'
import { useDocuments } from '@/hooks/useDocuments'
import { useQueryHistory } from '@/hooks/useQuery'

export const DashboardPage: React.FC = () => {
  const { data: documentsData } = useDocuments({ limit: 5 })
  const { data: queryData } = useQueryHistory({ limit: 5 })

  const stats = [
    {
      title: '总文档数',
      value: documentsData?.pagination.total || 0,
      icon: <FileTextOutlined className="text-blue-500" />,
    },
    {
      title: '今日查询',
      value: 23,
      icon: <MessageOutlined className="text-green-500" />,
    },
    {
      title: '处理中文档',
      value: 2,
      icon: <ClockCircleOutlined className="text-orange-500" />,
    },
    {
      title: '完成率',
      value: 95,
      suffix: '%',
      icon: <CheckCircleOutlined className="text-purple-500" />,
    },
  ]

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">仪表板</h1>

      {/* 统计卡片 */}
      <Row gutter={[16, 16]}>
        {stats.map((stat, index) => (
          <Col xs={24} sm={12} lg={6} key={index}>
            <Card>
              <Statistic
                title={stat.title}
                value={stat.value}
                suffix={stat.suffix}
                prefix={stat.icon}
              />
            </Card>
          </Col>
        ))}
      </Row>

      <Row gutter={[16, 16]}>
        {/* 最近文档 */}
        <Col xs={24} lg={12}>
          <Card title="最近上传" extra={<a href="/documents">查看全部</a>}>
            <List
              itemLayout="horizontal"
              dataSource={documentsData?.data || []}
              renderItem={(item) => (
                <List.Item>
                  <List.Item.Meta
                    avatar={<Avatar icon={<FileTextOutlined />} />}
                    title={item.originalName}
                    description={`${item.status} • ${new Date(item.createdAt).toLocaleDateString()}`}
                  />
                </List.Item>
              )}
            />
          </Card>
        </Col>

        {/* 最近查询 */}
        <Col xs={24} lg={12}>
          <Card title="最近查询" extra={<a href="/chat">开始对话</a>}>
            <List
              itemLayout="horizontal"
              dataSource={queryData?.data || []}
              renderItem={(item) => (
                <List.Item>
                  <List.Item.Meta
                    avatar={<Avatar icon={<MessageOutlined />} />}
                    title={item.text.substring(0, 30) + '...'}
                    description={new Date(item.createdAt).toLocaleString()}
                  />
                </List.Item>
              )}
            />
          </Card>
        </Col>
      </Row>
    </div>
  )
}
```

**Task 4.4.2: 加载和错误状态**

```typescript
// src/components/common/Loading.tsx
import React from 'react'
import { Spin } from 'antd'

interface LoadingProps {
  size?: 'small' | 'default' | 'large'
  tip?: string
  className?: string
}

export const Loading: React.FC<LoadingProps> = ({ 
  size = 'default', 
  tip = '加载中...', 
  className = '' 
}) => {
  return (
    <div className={`flex items-center justify-center p-8 ${className}`}>
      <Spin size={size} tip={tip} />
    </div>
  )
}

// src/components/common/ErrorBoundary.tsx
import React, { Component, ErrorInfo, ReactNode } from 'react'
import { Result, Button } from 'antd'

interface Props {
  children: ReactNode
}

interface State {
  hasError: boolean
  error?: Error
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo)
  }

  render() {
    if (this.state.hasError) {
      return (
        <Result
          status="500"
          title="500"
          subTitle="抱歉，页面出现了错误。"
          extra={
            <Button type="primary" onClick={() => window.location.reload()}>
              刷新页面
            </Button>
          }
        />
      )
    }

    return this.props.children
  }
}

// src/components/common/EmptyState.tsx
import React from 'react'
import { Empty, Button } from 'antd'

interface EmptyStateProps {
  title?: string
  description?: string
  action?: {
    text: string
    onClick: () => void
  }
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  title = '暂无数据',
  description,
  action,
}) => {
  return (
    <div className="flex items-center justify-center p-8">
      <Empty
        description={
          <div>
            <p className="text-gray-500">{title}</p>
            {description && <p className="text-sm text-gray-400">{description}</p>}
          </div>
        }
      >
        {action && (
          <Button type="primary" onClick={action.onClick}>
            {action.text}
          </Button>
        )}
      </Empty>
    </div>
  )
}
```

**Task 4.4.3: 路由和权限控制**

```typescript
// src/App.tsx
import React, { useEffect } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'
import { ConfigProvider } from 'antd'
import { Toaster } from 'react-hot-toast'
import zhCN from 'antd/locale/zh_CN'

import { useAuthStore } from '@/store/authStore'
import { ErrorBoundary } from '@/components/common/ErrorBoundary'
import { MainLayout } from '@/components/layout/MainLayout'
import { AuthLayout } from '@/components/layout/AuthLayout'
import { ProtectedRoute } from '@/components/common/ProtectedRoute'

// Pages
import { LoginPage } from '@/pages/auth/LoginPage'
import { RegisterPage } from '@/pages/auth/RegisterPage'
import { DashboardPage } from '@/pages/dashboard/DashboardPage'
import { ChatPage } from '@/pages/chat/ChatPage'
import { DocumentListPage } from '@/pages/documents/DocumentListPage'
import { ProfilePage } from '@/pages/auth/ProfilePage'

import '@/styles/global.css'

// 创建React Query客户端
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      staleTime: 5 * 60 * 1000, // 5分钟
    },
  },
})

const App: React.FC = () => {
  const { loadUser } = useAuthStore()

  useEffect(() => {
    loadUser()
  }, [loadUser])

  return (
    <QueryClientProvider client={queryClient}>
      <ConfigProvider locale={zhCN}>
        <ErrorBoundary>
          <BrowserRouter>
            <Routes>
              {/* 认证相关路由 */}
              <Route path="/auth" element={<AuthLayout />}>
                <Route path="login" element={<LoginPage />} />
                <Route path="register" element={<RegisterPage />} />
              </Route>

              {/* 主应用路由 */}
              <Route path="/" element={
                <ProtectedRoute>
                  <MainLayout />
                </ProtectedRoute>
              }>
                <Route index element={<Navigate to="/dashboard" replace />} />
                <Route path="dashboard" element={<DashboardPage />} />
                <Route path="chat" element={<ChatPage />} />
                <Route path="documents" element={<DocumentListPage />} />
                <Route path="profile" element={<ProfilePage />} />
              </Route>

              {/* 默认路由重定向 */}
              <Route path="/login" element={<Navigate to="/auth/login" replace />} />
              <Route path="/register" element={<Navigate to="/auth/register" replace />} />
              <Route path="*" element={<Navigate to="/dashboard" replace />} />
            </Routes>
          </BrowserRouter>
          
          <Toaster 
            position="top-right" 
            toastOptions={{
              duration: 4000,
              className: 'text-sm',
            }} 
          />
        </ErrorBoundary>
      </ConfigProvider>
      
      {import.meta.env.VITE_ENABLE_DEVTOOLS === 'true' && (
        <ReactQueryDevtools initialIsOpen={false} />
      )}
    </QueryClientProvider>
  )
}

export default App

// src/components/common/ProtectedRoute.tsx
import React from 'react'
import { Navigate, useLocation } from 'react-router-dom'
import { useAuthStore } from '@/store/authStore'
import { Loading } from './Loading'

interface ProtectedRouteProps {
  children: React.ReactNode
}

export const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ children }) => {
  const { isAuthenticated, user } = useAuthStore()
  const location = useLocation()

  // 如果用户未认证，重定向到登录页
  if (!isAuthenticated) {
    return <Navigate to="/auth/login" state={{ from: location }} replace />
  }

  // 如果用户信息还在加载中
  if (!user) {
    return <Loading />
  }

  return <>{children}</>
}
```

**Task 4.4.4: 工具函数和常量**

```typescript
// src/utils/formatters.ts
import dayjs from 'dayjs'
import relativeTime from 'dayjs/plugin/relativeTime'
import 'dayjs/locale/zh-cn'

dayjs.extend(relativeTime)
dayjs.locale('zh-cn')

export const formatBytes = (bytes: number): string => {
  if (bytes === 0) return '0 B'
  
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i]
}

export const formatDate = (date: string | number): string => {
  return dayjs(date).format('YYYY-MM-DD HH:mm')
}

export const formatRelativeTime = (date: string | number): string => {
  return dayjs(date).fromNow()
}

export const truncateText = (text: string, length: number = 100): string => {
  return text.length > length ? text.substring(0, length) + '...' : text
}

// src/utils/constants.ts
export const APP_CONFIG = {
  name: 'RAG智能问答系统',
  version: '1.0.0',
  api: {
    baseUrl: import.meta.env.VITE_API_BASE_URL || 'http://localhost:3001',
    timeout: 30000,
  },
  upload: {
    maxSize: 10 * 1024 * 1024, // 10MB
    acceptedTypes: ['.pdf', '.docx', '.doc', '.txt', '.md'],
  },
  pagination: {
    defaultPageSize: 10,
    pageSizeOptions: ['10', '20', '50', '100'],
  },
}

export const DOCUMENT_STATUS = {
  PENDING: { color: 'orange', text: '等待处理' },
  PROCESSING: { color: 'blue', text: '处理中' },
  COMPLETED: { color: 'green', text: '已完成' },
  FAILED: { color: 'red', text: '处理失败' },
  DELETED: { color: 'gray', text: '已删除' },
} as const

export const QUERY_STATUS = {
  PENDING: { color: 'orange', text: '等待处理' },
  PROCESSING: { color: 'blue', text: '处理中' },
  COMPLETED: { color: 'green', text: '完成' },
  FAILED: { color: 'red', text: '失败' },
} as const

// src/utils/validation.ts
import { z } from 'zod'

export const loginSchema = z.object({
  email: z.string().email('请输入有效的邮箱地址'),
  password: z.string().min(1, '请输入密码'),
})

export const registerSchema = z.object({
  email: z.string().email('请输入有效的邮箱地址'),
  password: z.string().min(8, '密码至少8个字符'),
  name: z.string().min(1, '请输入姓名'),
})

export const querySchema = z.object({
  query: z.string().min(1, '请输入查询内容').max(1000, '查询内容不能超过1000个字符'),
  topK: z.number().min(1).max(20).optional(),
  useCache: z.boolean().optional(),
})
```

#### ✅ 验收标准4.4

- [ ] 仪表板页面数据展示完整
- [ ] 加载状态和错误处理用户体验良好
- [ ] 路由权限控制正确，未登录用户正确重定向
- [ ] 工具函数和常量定义完善
- [ ] 表单验证功能正常
- [ ] 整体界面响应式设计良好

---

### 子阶段4.5：集成测试与优化 (1-2天) 🧪

**状态**: 🔴 待开始  
**优先级**: ⚡ 中等  
**依赖**: 子阶段4.4完成

#### 🎯 具体任务清单

**Task 4.5.1: 端到端功能测试**

```typescript
// src/__tests__/integration/auth.test.tsx
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { BrowserRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { LoginPage } from '@/pages/auth/LoginPage'

const TestWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } }
  })
  
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        {children}
      </BrowserRouter>
    </QueryClientProvider>
  )
}

describe('Authentication Flow', () => {
  test('should login successfully with valid credentials', async () => {
    render(
      <TestWrapper>
        <LoginPage />
      </TestWrapper>
    )

    // 填写登录表单
    fireEvent.change(screen.getByPlaceholderText('邮箱地址'), {
      target: { value: 'test@example.com' }
    })
    fireEvent.change(screen.getByPlaceholderText('密码'), {
      target: { value: 'password123' }
    })

    // 提交表单
    fireEvent.click(screen.getByText('登录'))

    // 验证登录成功
    await waitFor(() => {
      expect(window.location.pathname).toBe('/dashboard')
    })
  })
})
```

**Task 4.5.2: 性能优化**

```typescript
// src/components/common/LazyWrapper.tsx
import React, { Suspense } from 'react'
import { Loading } from './Loading'

interface LazyWrapperProps {
  children: React.ReactNode
}

export const LazyWrapper: React.FC<LazyWrapperProps> = ({ children }) => {
  return (
    <Suspense fallback={<Loading />}>
      {children}
    </Suspense>
  )
}

// 路由懒加载
// src/routes/lazyRoutes.tsx
import { lazy } from 'react'

export const LazyDashboardPage = lazy(() => 
  import('@/pages/dashboard/DashboardPage').then(module => ({ 
    default: module.DashboardPage 
  }))
)

export const LazyChatPage = lazy(() => 
  import('@/pages/chat/ChatPage').then(module => ({ 
    default: module.ChatPage 
  }))
)

export const LazyDocumentListPage = lazy(() => 
  import('@/pages/documents/DocumentListPage').then(module => ({ 
    default: module.DocumentListPage 
  }))
)

// 在App.tsx中使用懒加载
import { LazyWrapper } from '@/components/common/LazyWrapper'
import { LazyDashboardPage, LazyChatPage, LazyDocumentListPage } from '@/routes/lazyRoutes'

// 在Routes中使用
<Route path="dashboard" element={
  <LazyWrapper>
    <LazyDashboardPage />
  </LazyWrapper>
} />
```

**Task 4.5.3: 缓存策略优化**

```typescript
// src/utils/cache.ts
import { QueryClient } from '@tanstack/react-query'

export const createQueryClient = () => {
  return new QueryClient({
    defaultOptions: {
      queries: {
        // 5分钟缓存
        staleTime: 5 * 60 * 1000,
        // 10分钟垃圾回收
        gcTime: 10 * 60 * 1000,
        // 失败重试1次
        retry: 1,
        // 窗口重新聚焦时不自动重新获取
        refetchOnWindowFocus: false,
      },
      mutations: {
        // 失败重试0次
        retry: 0,
      },
    },
  })
}

// 预加载关键数据
export const preloadData = (queryClient: QueryClient) => {
  // 预加载用户文档列表
  queryClient.prefetchQuery({
    queryKey: ['documents'],
    queryFn: () => import('@/services/documents').then(m => 
      m.documentService.getDocumentList({})
    ),
  })

  // 预加载查询历史
  queryClient.prefetchQuery({
    queryKey: ['queryHistory'],
    queryFn: () => import('@/services/queries').then(m => 
      m.queryService.getQueryHistory({})
    ),
  })
}
```

**Task 4.5.4: 构建配置优化**

```typescript
// vite.config.ts 优化配置
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  build: {
    target: 'es2015',
    rollupOptions: {
      output: {
        manualChunks: {
          // 将大型依赖分离到独立chunk
          'react-vendor': ['react', 'react-dom'],
          'antd-vendor': ['antd'],
          'router-vendor': ['react-router-dom'],
          'query-vendor': ['@tanstack/react-query'],
        },
      },
    },
    chunkSizeWarningLimit: 1000,
  },
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:3001',
        changeOrigin: true,
      },
    },
  },
})

// package.json scripts优化
{
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "lint": "eslint . --ext ts,tsx --report-unused-disable-directives --max-warnings 0",
    "lint:fix": "eslint . --ext ts,tsx --fix",
    "type-check": "tsc --noEmit",
    "test": "vitest",
    "test:coverage": "vitest --coverage",
    "analyze": "npx vite-bundle-analyzer dist"
  }
}
```

#### ✅ 验收标准4.5

- [ ] 端到端测试通过，主要功能流程正常
- [ ] 页面加载性能优化，首屏加载时间 < 3秒
- [ ] 代码分割和懒加载正确实现
- [ ] 缓存策略合理，减少不必要的网络请求
- [ ] 构建产物大小优化，chunk分离合理
- [ ] 无明显的性能瓶颈和内存泄漏

---

## 📊 执行进度跟踪

### 进度检查点

- [ ] **Day 3**: 子阶段4.1完成，项目架构搭建完成
- [ ] **Day 6**: 子阶段4.2完成，核心服务层就绪
- [ ] **Day 10**: 子阶段4.3完成，核心页面开发完成
- [ ] **Day 13**: 子阶段4.4完成，用户体验优化完成
- [ ] **Day 14**: 子阶段4.5完成，集成测试和优化完成

### 风险缓解计划

1. **UI组件复杂度**: 使用Ant Design降低开发复杂度
2. **状态管理复杂度**: Zustand简化状态管理逻辑
3. **API集成问题**: 完善的错误处理和重试机制
4. **性能优化风险**: 渐进式优化，优先保证功能完整性

---

## 🔧 技术规范

### 组件开发规范

```typescript
// 组件文件结构标准
// ComponentName.tsx
import React from 'react'
import { SomeAntdComponent } from 'antd'
import { SomeIcon } from '@ant-design/icons'
import { useCustomHook } from '@/hooks/useCustomHook'
import { ComponentProps, ComponentState } from '@/types'

interface Props {
  // 属性定义
}

export const ComponentName: React.FC<Props> = ({ 
  prop1, 
  prop2 
}) => {
  // 组件逻辑

  return (
    <div className="component-wrapper">
      {/* JSX内容 */}
    </div>
  )
}
```

### 样式规范

```css
/* 使用Tailwind CSS + Ant Design */
.custom-class {
  @apply flex items-center justify-between p-4 bg-white rounded-lg shadow-sm;
}

/* 响应式设计 */
.responsive-grid {
  @apply grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4;
}
```

### API集成规范

```typescript
// 统一的API调用方式
const { data, isLoading, error } = useQuery({
  queryKey: ['resourceName', params],
  queryFn: () => apiService.method(params),
  staleTime: 5 * 60 * 1000,
})

// 统一的错误处理
if (error) {
  return <ErrorComponent error={error} />
}

if (isLoading) {
  return <Loading />
}
```

---

## ✅ 最终交付清单

### 代码交付

- [ ] 完整的React前端应用
- [ ] 所有核心页面和组件
- [ ] 状态管理和数据获取逻辑
- [ ] TypeScript类型定义
- [ ] 路由和权限控制

### 功能交付

- [ ] 用户认证和授权
- [ ] 文档上传和管理
- [ ] RAG智能对话
- [ ] 仪表板和统计
- [ ] 响应式移动端支持

### 性能交付

- [ ] 首屏加载时间 < 3秒
- [ ] 代码分割和懒加载
- [ ] 缓存策略优化
- [ ] 构建产物优化

### 文档交付

- [ ] 组件使用文档
- [ ] 开发环境搭建指南
- [ ] 部署配置说明
- [ ] 用户操作手册

---

**执行原则**:

1. **组件化优先** - 所有UI元素都应该是可复用的组件
2. **用户体验至上** - 每个交互都要考虑用户感受
3. **性能意识** - 在开发过程中持续关注性能表现
4. **测试驱动** - 核心功能要有对应的测试用例
5. **渐进式增强** - 先保证基础功能，再添加高级特性

此文档将作为阶段4执行的最高指导，确保React前端开发按照规范和标准完成，为整个RAG系统提供优秀的用户界面体验！