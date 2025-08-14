# RAG前端应用 - 开发指导文档

## 📋 应用概览

**RAG Frontend** 是基于React 18的现代单页应用，提供直观的文档管理界面和智能问答体验。采用TypeScript + Vite构建，集成Ant Design组件库，支持响应式设计和实时数据交互。

### 🏗️ 技术栈
- **框架**: React 18 + TypeScript + Vite
- **UI组件**: Ant Design 5.x + Lucide Icons
- **路由**: React Router DOM 7.x
- **状态管理**: Zustand (轻量状态) + TanStack Query (服务状态)
- **HTTP客户端**: Axios + 拦截器
- **样式**: Tailwind CSS 4.x + CSS Modules
- **构建工具**: Vite + TypeScript
- **端口**: 3000

### 📁 项目结构
```
frontend/
├── src/
│   ├── components/         # 公共组件
│   │   ├── common/        # 通用UI组件
│   │   ├── layout/        # 布局组件  
│   │   └── ui/            # 业务UI组件
│   ├── pages/             # 页面组件
│   │   ├── auth/          # 认证页面
│   │   ├── dashboard/     # 仪表板
│   │   ├── documents/     # 文档管理
│   │   └── chat/          # AI聊天
│   ├── hooks/             # 自定义hooks
│   ├── services/          # API服务层
│   ├── stores/            # 状态管理
│   ├── types/             # TypeScript类型
│   ├── utils/             # 工具函数
│   └── styles/            # 样式文件
├── public/                # 静态资源
└── tests/                 # 测试用例
```

## 🎯 核心功能模块

### 🔐 1. 认证系统
- **登录页面**: `/login` - 用户登录界面
- **注册页面**: `/register` - 用户注册界面  
- **自动认证**: JWT令牌管理和自动续签
- **路由守卫**: 未认证用户自动跳转登录

### 📄 2. 文档管理
- **文档列表**: `/documents` - 文档管理主界面
- **文档上传**: 拖拽上传 + 进度显示
- **实时搜索**: 文档名称模糊搜索
- **状态筛选**: 按处理状态筛选文档
- **分页浏览**: 高性能分页加载

### 🤖 3. AI聊天系统
- **智能问答**: `/chat` - RAG聊天界面
- **聊天历史**: 对话记录保存和查看
- **来源显示**: 答案来源文档高亮
- **实时交互**: 流式响应显示

## 🧪 开发和调试指南

### 环境配置
```bash
# 环境变量 (.env)
VITE_API_BASE_URL=http://localhost:3001
VITE_APP_NAME="RAG智能问答系统"
VITE_APP_VERSION=1.0.0
VITE_ENABLE_DEVTOOLS=true

# 开发模式配置
VITE_DEV_MODE=true
VITE_MOCK_API=false
```

### 本地开发启动
```bash
# 1. 安装依赖
npm install

# 2. 启动开发服务器
npm run dev

# 3. 构建生产版本
npm run build

# 4. 预览构建结果
npm run preview

# 应用地址: http://localhost:3000
```

### 代码质量工具
```bash
# 代码检查
npm run lint

# 自动修复
npm run lint:fix

# 类型检查
npm run type-check

# 运行测试
npm run test

# 测试覆盖率
npm run test:coverage
```

## 🔧 核心组件架构

### 1. 路由配置
```tsx
// src/App.tsx - 主路由配置
import { createBrowserRouter, RouterProvider } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

const router = createBrowserRouter([
  {
    path: '/',
    element: <ProtectedLayout />,
    children: [
      { path: '/', element: <Navigate to="/dashboard" replace /> },
      { path: '/dashboard', element: <DashboardPage /> },
      { path: '/documents', element: <DocumentListPage /> },
      { path: '/chat', element: <ChatPage /> },
    ],
  },
  {
    path: '/login',
    element: <LoginPage />,
  },
  {
    path: '/register',
    element: <RegisterPage />,
  },
]);

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 3,
      staleTime: 5 * 60 * 1000, // 5分钟
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  );
}
```

### 2. 认证状态管理
```tsx
// src/stores/authStore.ts - 认证状态
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  login: (user: User, token: string) => void;
  logout: () => void;
  updateUser: (user: Partial<User>) => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      isAuthenticated: false,
      
      login: (user, token) => {
        set({ user, token, isAuthenticated: true });
        // 设置axios默认头
        axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      },
      
      logout: () => {
        set({ user: null, token: null, isAuthenticated: false });
        delete axios.defaults.headers.common['Authorization'];
        // 跳转到登录页
        window.location.href = '/login';
      },
      
      updateUser: (userData) => {
        const { user } = get();
        if (user) {
          set({ user: { ...user, ...userData } });
        }
      },
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({ 
        user: state.user, 
        token: state.token,
        isAuthenticated: state.isAuthenticated 
      }),
    }
  )
);
```

### 3. API服务层
```tsx
// src/services/api.ts - API客户端配置
import axios from 'axios';
import { message } from 'antd';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL,
  timeout: 30000,
});

// 请求拦截器
api.interceptors.request.use(
  (config) => {
    const token = useAuthStore.getState().token;
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// 响应拦截器
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      useAuthStore.getState().logout();
    } else if (error.response?.status >= 500) {
      message.error('服务器错误，请稍后重试');
    } else if (error.code === 'NETWORK_ERROR') {
      message.error('网络连接失败');
    }
    return Promise.reject(error);
  }
);

export default api;
```

### 4. 文档管理API
```tsx
// src/services/documentService.ts - 文档相关API
import api from './api';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

// 文档列表查询
export const useDocuments = (params: DocumentListParams) => {
  return useQuery({
    queryKey: ['documents', params],
    queryFn: async () => {
      const { data } = await api.get('/api/documents/list', { params });
      return data;
    },
    keepPreviousData: true,
  });
};

// 文档上传
export const useUploadDocument = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (file: File) => {
      const formData = new FormData();
      formData.append('file', file);
      
      const { data } = await api.post('/api/documents/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries(['documents']);
      message.success('文档上传成功');
    },
    onError: (error: any) => {
      message.error(error.response?.data?.error || '上传失败');
    },
  });
};

// 文档删除
export const useDeleteDocument = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (documentId: string) => {
      await api.delete(`/api/documents/delete/${documentId}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries(['documents']);
      message.success('文档删除成功');
    },
    onError: () => {
      message.error('删除失败');
    },
  });
};
```

## 🎨 核心页面组件

### 1. 文档管理页面
```tsx
// src/pages/documents/DocumentListPage.tsx
import { useState } from 'react';
import { Table, Button, Upload, Input, Select, Space, Modal } from 'antd';
import { useDocuments, useUploadDocument, useDeleteDocument } from '@/services/documentService';

const DocumentListPage: React.FC = () => {
  const [params, setParams] = useState({
    page: 1,
    limit: 10,
    search: '',
    status: undefined,
  });

  const { data, isLoading } = useDocuments(params);
  const uploadMutation = useUploadDocument();
  const deleteMutation = useDeleteDocument();

  const columns = [
    {
      title: '文档名称',
      dataIndex: 'originalName',
      key: 'name',
      ellipsis: true,
    },
    {
      title: '文件大小',
      dataIndex: 'size',
      key: 'size',
      render: (size: number) => formatFileSize(size),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={getStatusColor(status)}>
          {getStatusText(status)}
        </Tag>
      ),
    },
    {
      title: '文档块数',
      dataIndex: 'chunksCount',
      key: 'chunks',
    },
    {
      title: '上传时间',
      dataIndex: 'createdAt',
      key: 'createdAt',
      render: (date: string) => dayjs(date).format('YYYY-MM-DD HH:mm'),
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: Document) => (
        <Space>
          <Button 
            type="link" 
            onClick={() => handleView(record.id)}
          >
            查看
          </Button>
          <Button 
            type="link" 
            danger
            onClick={() => handleDelete(record.id)}
            loading={deleteMutation.isLoading}
          >
            删除
          </Button>
        </Space>
      ),
    },
  ];

  const handleUpload = async (file: File) => {
    try {
      await uploadMutation.mutateAsync(file);
    } catch (error) {
      // 错误处理已在mutation中处理
    }
    return false; // 阻止默认上传行为
  };

  const handleDelete = (id: string) => {
    Modal.confirm({
      title: '确认删除',
      content: '删除后无法恢复，确定要删除这个文档吗？',
      onOk: () => deleteMutation.mutate(id),
    });
  };

  return (
    <div className="p-6">
      <div className="mb-4 flex justify-between items-center">
        <h1 className="text-2xl font-bold">文档管理</h1>
        <Upload
          beforeUpload={handleUpload}
          showUploadList={false}
          accept=".pdf,.doc,.docx,.txt,.md"
        >
          <Button type="primary" loading={uploadMutation.isLoading}>
            上传文档
          </Button>
        </Upload>
      </div>

      <div className="mb-4 flex gap-4">
        <Input.Search
          placeholder="搜索文档名称"
          value={params.search}
          onChange={(e) => setParams({ ...params, search: e.target.value, page: 1 })}
          style={{ width: 300 }}
        />
        <Select
          placeholder="筛选状态"
          value={params.status}
          onChange={(status) => setParams({ ...params, status, page: 1 })}
          allowClear
          style={{ width: 150 }}
        >
          <Select.Option value="PENDING">等待处理</Select.Option>
          <Select.Option value="PROCESSING">处理中</Select.Option>
          <Select.Option value="COMPLETED">已完成</Select.Option>
          <Select.Option value="FAILED">处理失败</Select.Option>
        </Select>
        <Button onClick={() => window.location.reload()}>
          刷新
        </Button>
      </div>

      <Table
        columns={columns}
        dataSource={data?.documents || []}
        loading={isLoading}
        pagination={{
          current: params.page,
          pageSize: params.limit,
          total: data?.pagination?.total || 0,
          showSizeChanger: true,
          showQuickJumper: true,
          onChange: (page, limit) => setParams({ ...params, page, limit }),
        }}
        rowKey="id"
      />
    </div>
  );
};

export default DocumentListPage;
```

### 2. AI聊天页面
```tsx
// src/pages/chat/ChatPage.tsx
import { useState, useRef, useEffect } from 'react';
import { Input, Button, Card, Typography, Spin, Empty } from 'antd';
import { SendOutlined } from '@ant-design/icons';
import { useChatQuery } from '@/services/chatService';

const ChatPage: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  const chatMutation = useChatQuery();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!inputValue.trim()) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: inputValue,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');

    try {
      const response = await chatMutation.mutateAsync(inputValue);
      
      const aiMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: response.answer,
        sources: response.sources,
        timestamp: new Date(),
        queryTime: response.queryTime,
      };

      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'error',
        content: '抱歉，查询失败，请稍后重试。',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    }
  };

  return (
    <div className="h-full flex flex-col">
      <div className="p-4 border-b">
        <h1 className="text-xl font-bold">AI智能问答</h1>
        <p className="text-gray-500 text-sm mt-1">
          基于您上传的文档进行智能问答
        </p>
      </div>

      <div className="flex-1 p-4 overflow-y-auto">
        {messages.length === 0 ? (
          <Empty 
            description="开始对话吧！问我任何关于您文档的问题。"
            image={Empty.PRESENTED_IMAGE_SIMPLE}
          />
        ) : (
          <div className="space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${
                  message.type === 'user' ? 'justify-end' : 'justify-start'
                }`}
              >
                <div
                  className={`max-w-3/4 p-3 rounded-lg ${
                    message.type === 'user'
                      ? 'bg-blue-500 text-white ml-auto'
                      : message.type === 'error'
                      ? 'bg-red-100 border border-red-300'
                      : 'bg-gray-100'
                  }`}
                >
                  <div className="whitespace-pre-wrap">{message.content}</div>
                  
                  {/* 显示来源文档 */}
                  {message.sources && message.sources.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-gray-300">
                      <div className="text-xs text-gray-600 mb-2">参考来源:</div>
                      {message.sources.map((source, idx) => (
                        <div key={idx} className="text-xs bg-white p-2 rounded mb-1">
                          <div className="text-gray-500">
                            相似度: {(source.score * 100).toFixed(1)}%
                          </div>
                          <div className="mt-1">{source.content.substring(0, 100)}...</div>
                        </div>
                      ))}
                    </div>
                  )}
                  
                  {/* 响应时间 */}
                  {message.queryTime && (
                    <div className="text-xs text-gray-500 mt-2">
                      响应时间: {(message.queryTime / 1000).toFixed(2)}秒
                    </div>
                  )}
                  
                  <div className="text-xs opacity-70 mt-2">
                    {dayjs(message.timestamp).format('HH:mm:ss')}
                  </div>
                </div>
              </div>
            ))}
            
            {chatMutation.isLoading && (
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

      <div className="p-4 border-t">
        <div className="flex gap-2">
          <Input.TextArea
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="输入您的问题..."
            autoSize={{ minRows: 1, maxRows: 3 }}
            onPressEnter={(e) => {
              if (!e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
          />
          <Button
            type="primary"
            icon={<SendOutlined />}
            onClick={handleSend}
            loading={chatMutation.isLoading}
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
  );
};

export default ChatPage;
```

#### 3.1 清空对话功能设计

基于用户体验和系统架构考虑，清空对话功能采用**分层清空**的设计方案，提供两种清空选项：

##### **功能设计规范**
```tsx
// 清空对话功能接口定义
interface ClearChatOptions {
  // 轻量级: 清空当前会话显示 (无确认)
  clearCurrentSession: () => void
  
  // 重量级: 清空所有历史记录 (双重确认 + 软删除)
  clearAllHistory: () => Promise<void>
}

// 清空对话组件设计
const ClearChatButton: React.FC = () => {
  const [historyCount, setHistoryCount] = useState(0)
  
  const clearActions = {
    // 方案1: 仅清空UI界面 (0.1秒响应)
    clearSession: () => {
      setMessages([])        // 清空聊天界面
      setInputValue('')      // 清空输入框
      message.success('对话已清空')
    },
    
    // 方案2: 删除后端历史数据 (确认 + API调用)
    clearAllHistory: async () => {
      const count = await chatService.getHistoryCount()
      setHistoryCount(count)
      
      Modal.confirm({
        title: '⚠️ 确认清空全部历史？',
        content: (
          <div>
            <p>将删除您的 <strong>{count}</strong> 条查询记录</p>
            <p style={{color: '#ff4d4f'}}>此操作不可恢复！</p>
          </div>
        ),
        okText: '确认清空',
        okType: 'danger',
        cancelText: '取消',
        onOk: async () => {
          await chatService.clearAllHistory()
          setMessages([])
          message.success('历史记录已清空')
        }
      })
    }
  }
  
  return (
    <Dropdown
      menu={{
        items: [
          {
            key: 'clear-session',
            label: (
              <div>
                <MessageOutlined className="mr-2" />
                清空当前对话
                <div className="text-xs text-gray-500">仅清理界面显示</div>
              </div>
            ),
            onClick: clearActions.clearSession
          },
          { type: 'divider' },
          {
            key: 'clear-all',
            label: (
              <div>
                <DeleteOutlined className="mr-2" />
                清空全部历史
                <div className="text-xs text-red-500">永久删除所有记录</div>
              </div>
            ),
            danger: true,
            onClick: clearActions.clearAllHistory
          }
        ]
      }}
      trigger={['click']}
      placement="bottomRight"
    >
      <Button 
        icon={<ClearOutlined />}
        type="text"
        size="small"
        className="ml-2"
      >
        清空
      </Button>
    </Dropdown>
  )
}
```

##### **UI集成位置**
```tsx
// 在ChatPage头部区域添加清空按钮
<div className="p-4 border-b flex justify-between items-center">
  <div>
    <h1 className="text-xl font-bold">AI智能问答</h1>
    <p className="text-gray-500 text-sm mt-1">
      基于您上传的文档进行智能问答
    </p>
  </div>
  {/* 清空对话按钮 */}
  <ClearChatButton />
</div>
```

##### **后端API支持**
```tsx
// src/services/chatService.ts - 清空功能API
export const chatService = {
  // 获取历史记录统计
  getHistoryCount: async (): Promise<number> => {
    const response = await api.get('/api/query/history-stats')
    return response.data.total
  },
  
  // 清空所有历史记录 (软删除)
  clearAllHistory: async (): Promise<void> => {
    await api.delete('/api/query/history-clear', {
      data: { confirm: true }
    })
  }
}

// TanStack Query集成
export const useClearHistory = () => {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: chatService.clearAllHistory,
    onSuccess: () => {
      // 清空相关查询缓存
      queryClient.invalidateQueries(['chat-history'])
      queryClient.invalidateQueries(['recent-queries'])
      message.success('历史记录已清空')
    },
    onError: (error) => {
      message.error('清空失败，请稍后重试')
      console.error('Clear history failed:', error)
    }
  })
}
```

##### **完整实现示例**
```tsx
// 完整的ChatPage组件 (含清空功能)
const ChatPage: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [inputValue, setInputValue] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)
  
  const chatMutation = useChatQuery()
  const clearHistoryMutation = useClearHistory()
  
  // 清空当前会话
  const handleClearSession = () => {
    setMessages([])
    setInputValue('')
    message.success('对话已清空')
  }
  
  // 清空全部历史
  const handleClearAllHistory = () => {
    Modal.confirm({
      title: '⚠️ 确认清空全部历史？',
      content: '将删除您的所有查询记录，此操作不可恢复',
      okText: '确认清空',
      okType: 'danger',
      onOk: () => {
        clearHistoryMutation.mutate()
        setMessages([]) // 同时清空当前界面
      }
    })
  }
  
  // ... 其他组件逻辑
  
  return (
    <div className="h-full flex flex-col">
      {/* 头部区域 - 包含清空按钮 */}
      <div className="p-4 border-b flex justify-between items-center">
        <div>
          <h1 className="text-xl font-bold">AI智能问答</h1>
          <p className="text-gray-500 text-sm mt-1">
            基于您上传的文档进行智能问答
          </p>
        </div>
        
        <Dropdown
          menu={{
            items: [
              {
                key: 'clear-session',
                label: '清空当前对话',
                icon: <MessageOutlined />,
                onClick: handleClearSession
              },
              { type: 'divider' },
              {
                key: 'clear-all',
                label: '清空全部历史',
                icon: <DeleteOutlined />,
                danger: true,
                onClick: handleClearAllHistory
              }
            ]
          }}
        >
          <Button icon={<ClearOutlined />} type="text" size="small">
            清空
          </Button>
        </Dropdown>
      </div>
      
      {/* ... 聊天内容和输入区域 */}
    </div>
  )
}
```

##### **设计优势总结**
- **分层设计**: 两种清空选项满足不同使用场景
- **安全确认**: 危险操作有明确的确认机制  
- **性能友好**: 轻量级清空无网络请求
- **用户直观**: 清晰的功能说明和视觉区分
- **数据保护**: 采用软删除策略，可设置恢复机制

### 3. 仪表板页面
```tsx
// src/pages/dashboard/DashboardPage.tsx
import { Card, Statistic, Row, Col, List, Button } from 'antd';
import { FileTextOutlined, MessageOutlined, ClockCircleOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';

const DashboardPage: React.FC = () => {
  const navigate = useNavigate();
  const { data: stats } = useDashboardStats();
  const { data: recentQueries } = useRecentQueries({ limit: 5 });

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-6">仪表板</h1>
      
      <Row gutter={16} className="mb-6">
        <Col span={8}>
          <Card>
            <Statistic
              title="文档总数"
              value={stats?.totalDocuments || 0}
              prefix={<FileTextOutlined />}
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card>
            <Statistic
              title="今日查询"
              value={stats?.todayQueries || 0}
              prefix={<MessageOutlined />}
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card>
            <Statistic
              title="平均响应时间"
              value={stats?.avgResponseTime || 0}
              suffix="秒"
              precision={2}
              prefix={<ClockCircleOutlined />}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={16}>
        <Col span={12}>
          <Card title="最近查询" extra={<Button type="link" onClick={() => navigate('/chat')}>查看全部</Button>}>
            <List
              dataSource={recentQueries?.queries || []}
              renderItem={(query) => (
                <List.Item>
                  <List.Item.Meta
                    title={query.text}
                    description={dayjs(query.createdAt).fromNow()}
                  />
                </List.Item>
              )}
            />
          </Card>
        </Col>
        <Col span={12}>
          <Card title="快速操作">
            <div className="space-y-3">
              <Button
                type="primary"
                block
                icon={<FileTextOutlined />}
                onClick={() => navigate('/documents')}
              >
                管理文档
              </Button>
              <Button
                block
                icon={<MessageOutlined />}
                onClick={() => navigate('/chat')}
              >
                开始问答
              </Button>
            </div>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default DashboardPage;
```

## 🎨 样式和主题

### Tailwind配置
```javascript
// tailwind.config.js
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#f0f9ff',
          500: '#3b82f6',
          600: '#2563eb',
        },
      },
    },
  },
  plugins: [],
}
```

### Ant Design主题定制
```tsx
// src/styles/theme.ts
import type { ThemeConfig } from 'antd';

export const antdTheme: ThemeConfig = {
  token: {
    colorPrimary: '#3b82f6',
    borderRadius: 8,
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
  },
  components: {
    Layout: {
      headerBg: '#ffffff',
      siderBg: '#fafafa',
    },
    Table: {
      headerBg: '#f8fafc',
    },
  },
};
```

## 🧪 测试策略

### 单元测试
```tsx
// src/tests/components/DocumentList.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import DocumentListPage from '../pages/documents/DocumentListPage';

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  );
};

describe('DocumentListPage', () => {
  it('should render document list correctly', () => {
    render(<DocumentListPage />, { wrapper: createWrapper() });
    
    expect(screen.getByText('文档管理')).toBeInTheDocument();
    expect(screen.getByText('上传文档')).toBeInTheDocument();
  });

  it('should handle search input', () => {
    render(<DocumentListPage />, { wrapper: createWrapper() });
    
    const searchInput = screen.getByPlaceholderText('搜索文档名称');
    fireEvent.change(searchInput, { target: { value: 'test' } });
    
    expect(searchInput.value).toBe('test');
  });
});
```

### E2E测试
```tsx
// src/tests/e2e/document-management.spec.ts
import { test, expect } from '@playwright/test';

test.describe('Document Management', () => {
  test.beforeEach(async ({ page }) => {
    // 登录
    await page.goto('/login');
    await page.fill('input[type="email"]', 'test@example.com');
    await page.fill('input[type="password"]', 'testpassword123');
    await page.click('button[type="submit"]');
    await page.waitForURL('/dashboard');
    
    // 导航到文档页面
    await page.click('text=文档管理');
    await page.waitForURL('/documents');
  });

  test('should upload document successfully', async ({ page }) => {
    // 模拟文件上传
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles('test-document.pdf');
    
    // 验证上传成功
    await expect(page.locator('.ant-message-success')).toBeVisible();
  });

  test('should search documents', async ({ page }) => {
    const searchInput = page.locator('input[placeholder*="搜索"]');
    await searchInput.fill('test');
    await searchInput.press('Enter');
    
    // 验证搜索结果
    await expect(page.locator('table tbody tr')).toBeVisible();
  });
});
```

## 📱 响应式设计

### 断点配置
```tsx
// src/hooks/useBreakpoint.ts
import { useEffect, useState } from 'react';

const breakpoints = {
  xs: 480,
  sm: 576, 
  md: 768,
  lg: 992,
  xl: 1200,
  xxl: 1600,
};

export const useBreakpoint = () => {
  const [breakpoint, setBreakpoint] = useState<keyof typeof breakpoints>('xl');

  useEffect(() => {
    const handleResize = () => {
      const width = window.innerWidth;
      
      if (width >= breakpoints.xxl) setBreakpoint('xxl');
      else if (width >= breakpoints.xl) setBreakpoint('xl');
      else if (width >= breakpoints.lg) setBreakpoint('lg');
      else if (width >= breakpoints.md) setBreakpoint('md');
      else if (width >= breakpoints.sm) setBreakpoint('sm');
      else setBreakpoint('xs');
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return breakpoint;
};
```

### 移动端适配
```tsx
// src/components/layout/MobileLayout.tsx
const MobileLayout: React.FC = ({ children }) => {
  const [collapsed, setCollapsed] = useState(true);

  return (
    <Layout className="min-h-screen">
      <Layout.Header className="flex items-center justify-between px-4">
        <Button
          type="text"
          icon={collapsed ? <MenuOutlined /> : <CloseOutlined />}
          onClick={() => setCollapsed(!collapsed)}
        />
        <h1 className="text-lg font-bold">RAG问答</h1>
        <UserDropdown />
      </Layout.Header>
      
      <Drawer
        placement="left"
        closable={false}
        open={!collapsed}
        onClose={() => setCollapsed(true)}
        width={280}
        bodyStyle={{ padding: 0 }}
      >
        <NavigationMenu />
      </Drawer>
      
      <Layout.Content className="p-4">
        {children}
      </Layout.Content>
    </Layout>
  );
};
```

## 🚀 性能优化

### 代码分割
```tsx
// src/pages/index.ts - 懒加载页面
import { lazy } from 'react';

export const DashboardPage = lazy(() => import('./dashboard/DashboardPage'));
export const DocumentListPage = lazy(() => import('./documents/DocumentListPage'));
export const ChatPage = lazy(() => import('./chat/ChatPage'));
```

### 虚拟滚动
```tsx
// src/components/VirtualList.tsx - 长列表优化
import { FixedSizeList as List } from 'react-window';

const VirtualDocumentList: React.FC<{ items: Document[] }> = ({ items }) => {
  const Row = ({ index, style }: { index: number; style: React.CSSProperties }) => (
    <div style={style}>
      <DocumentItem document={items[index]} />
    </div>
  );

  return (
    <List
      height={600}
      itemCount={items.length}
      itemSize={80}
      width="100%"
    >
      {Row}
    </List>
  );
};
```

### 缓存策略
```tsx
// src/utils/cache.ts
export class LocalStorageCache {
  private prefix = 'rag-app:';
  
  set(key: string, value: any, ttl?: number) {
    const data = {
      value,
      expires: ttl ? Date.now() + ttl : null,
    };
    localStorage.setItem(this.prefix + key, JSON.stringify(data));
  }
  
  get<T>(key: string): T | null {
    const item = localStorage.getItem(this.prefix + key);
    if (!item) return null;
    
    const data = JSON.parse(item);
    if (data.expires && Date.now() > data.expires) {
      this.delete(key);
      return null;
    }
    
    return data.value;
  }
  
  delete(key: string) {
    localStorage.removeItem(this.prefix + key);
  }
}

export const cache = new LocalStorageCache();
```

## 📋 开发检查清单

### ✅ 组件开发规范
- [ ] 使用TypeScript类型定义
- [ ] 实现错误边界处理
- [ ] 添加loading和error状态
- [ ] 实现响应式设计
- [ ] 添加组件测试用例
- [ ] 遵循Accessibility规范

### ✅ 状态管理
- [ ] 使用Zustand管理全局状态
- [ ] 使用TanStack Query管理服务状态
- [ ] 实现状态持久化
- [ ] 优化重新渲染性能
- [ ] 添加状态调试工具

### ✅ 性能优化
- [ ] 实现代码分割
- [ ] 添加虚拟滚动
- [ ] 实现图片懒加载
- [ ] 优化包体积
- [ ] 添加性能监控

### ✅ 用户体验
- [ ] 实现loading骨架屏
- [ ] 添加错误提示机制
- [ ] 实现离线功能提示
- [ ] 添加快捷键支持
- [ ] 优化移动端体验

## 🔗 相关文档

- [后端API文档](../backend/CLAUDE.md)
- [Python服务文档](../python-services/CLAUDE.md)  
- [设计系统文档](./src/styles/README.md)
- [测试文档](./tests/README.md)
- [部署配置](../docker-compose.yml)

---

## 🚀 **RAG系统准确率优化方案**
**重要**: Frontend作为用户界面，正在实施[RAG系统准确率优化方案](../CLAUDE.md#rag系统准确率优化方案)，目标将准确率从53.4%-65%提升至90-95%。

## 📅 最新更新 (2025-08-14)

### 🎨 苹果设计系统升级完成
- **视觉重构**: 完成所有页面的苹果设计系统改造
- **毛玻璃效果**: 使用`backdrop-filter: blur(20px)`实现现代化界面
- **渐变色系**: 采用蓝紫渐变主色调和多色彩辅助渐变
- **交互动画**: 添加流畅的微动画和hover效果
- **主题系统**: 创建完整的`apple-theme.css`覆盖Ant Design组件

### 🌟 设计升级亮点
1. **仪表板页面**: 全新的渐变统计卡片和数据可视化
2. **聊天界面**: 重新设计的消息气泡和输入区域
3. **文档管理**: 现代化的文件列表和操作界面  
4. **个人中心**: 清爽的用户信息和统计展示
5. **全局布局**: 毛玻璃侧边栏和固定导航设计

### 🎯 技术实现
- **CSS技术**: 使用现代CSS3特性，包括渐变、阴影、变换
- **动画系统**: 采用`cubic-bezier(0.4, 0, 0.2, 1)`苹果标准缓动
- **响应式设计**: 完美适配桌面和移动设备
- **主题覆盖**: 全面自定义Ant Design组件样式

### 📊 前端性能指标 (设计升级后)
| 功能模块 | 响应时间 | 用户体验 | 升级效果 |
|---------|---------|---------|----------|
| 页面加载 | <300ms | 流畅 | ✅ 视觉效果显著提升 |
| 消息发送 | <100ms | 即时 | ✅ 苹果风格气泡界面 |
| UI更新 | <16ms | 60fps | ✅ 流畅动画和过渡 |
| 路由切换 | <200ms | 无感知 | ✅ 毛玻璃过渡效果 |

### 📋 Frontend优化重点

#### ✅ Phase 1: 用户体验优化 (进行中)
**智能查询建议** - 提升查询质量
```tsx
// src/components/QuerySuggestion.tsx
export const QuerySuggestion: React.FC = () => {
  const [suggestions, setSuggestions] = useState<string[]>([])
  
  const generateSuggestions = useCallback(async (input: string) => {
    if (input.length < 3) return
    
    // 基于查询历史和文档内容生成建议
    const response = await api.post('/api/query/suggestions', { input })
    setSuggestions(response.data.suggestions)
  }, [])
  
  return (
    <div className="suggestion-panel">
      <h4>💡 建议查询</h4>
      {suggestions.map((suggestion, index) => (
        <Button
          key={index}
          type="text"
          size="small"
          onClick={() => onSuggestionClick(suggestion)}
          className="suggestion-item"
        >
          {suggestion}
        </Button>
      ))}
    </div>
  )
}
```

**实时反馈系统** - 准确率追踪
```tsx
// src/components/FeedbackWidget.tsx
export const FeedbackWidget: React.FC<{ queryId: string }> = ({ queryId }) => {
  const [rating, setRating] = useState<number>(0)
  const feedbackMutation = useFeedbackMutation()
  
  const handleFeedback = (rating: number, comment?: string) => {
    feedbackMutation.mutate({
      queryId,
      rating,
      comment,
      timestamp: new Date().toISOString()
    })
  }
  
  return (
    <div className="feedback-widget">
      <Rate 
        value={rating} 
        onChange={setRating}
        character={<LikeOutlined />}
      />
      <Button 
        size="small" 
        type="link"
        onClick={() => handleFeedback(rating)}
      >
        提交反馈
      </Button>
    </div>
  )
}
```

**查询历史智能化** - 学习用户偏好
```tsx
// src/components/SmartHistory.tsx
export const SmartHistory: React.FC = () => {
  const { data: history } = useQueryHistory()
  const [groupedHistory, setGroupedHistory] = useState<GroupedHistory>({})
  
  useEffect(() => {
    if (history) {
      // 按主题和准确率分组
      const grouped = groupQueriesByTopic(history)
      setGroupedHistory(grouped)
    }
  }, [history])
  
  return (
    <div className="smart-history">
      {Object.entries(groupedHistory).map(([topic, queries]) => (
        <div key={topic} className="topic-group">
          <h4>{topic}</h4>
          {queries.map(query => (
            <HistoryItem 
              key={query.id}
              query={query}
              showAccuracy={true}
              onReuse={() => reuseQuery(query.text)}
            />
          ))}
        </div>
      ))}
    </div>
  )
}
```

#### 🔄 Phase 2: 交互优化 (计划中)
**多模态查询界面** - 支持图像和语音
```tsx
// src/components/MultiModalInput.tsx
export const MultiModalInput: React.FC = () => {
  const [inputMode, setInputMode] = useState<'text' | 'voice' | 'image'>('text')
  
  const VoiceInput = () => (
    <SpeechRecognition
      onResult={(transcript) => setQuery(transcript)}
      onError={(error) => message.error('语音识别失败')}
    />
  )
  
  const ImageInput = () => (
    <Upload
      accept="image/*"
      beforeUpload={(file) => {
        processImageQuery(file)
        return false
      }}
    >
      <Button icon={<CameraOutlined />}>上传图片查询</Button>
    </Upload>
  )
  
  return (
    <div className="multimodal-input">
      <Radio.Group value={inputMode} onChange={(e) => setInputMode(e.target.value)}>
        <Radio.Button value="text">文本</Radio.Button>
        <Radio.Button value="voice">语音</Radio.Button>
        <Radio.Button value="image">图像</Radio.Button>
      </Radio.Group>
      
      {inputMode === 'text' && <TextInput />}
      {inputMode === 'voice' && <VoiceInput />}
      {inputMode === 'image' && <ImageInput />}
    </div>
  )
}
```

**结果可视化增强** - 更好的信息展示
```tsx
// src/components/ResultVisualization.tsx
export const ResultVisualization: React.FC<{ result: QueryResult }> = ({ result }) => {
  const [viewMode, setViewMode] = useState<'text' | 'graph' | 'table'>('text')
  
  const GraphView = () => (
    <div className="graph-view">
      {result.entities && (
        <EntityGraph 
          entities={result.entities}
          relations={result.relations}
        />
      )}
    </div>
  )
  
  const TableView = () => (
    <Table
      columns={result.table_headers}
      dataSource={result.table_data}
      pagination={false}
      size="small"
    />
  )
  
  return (
    <div className="result-visualization">
      <div className="view-controls">
        <Radio.Group value={viewMode} onChange={(e) => setViewMode(e.target.value)}>
          <Radio.Button value="text">文本</Radio.Button>
          <Radio.Button value="graph">图谱</Radio.Button>
          <Radio.Button value="table">表格</Radio.Button>
        </Radio.Group>
      </div>
      
      {viewMode === 'text' && <TextResult result={result} />}
      {viewMode === 'graph' && <GraphView />}
      {viewMode === 'table' && <TableView />}
    </div>
  )
}
```

#### 📈 Phase 3: 性能和可用性 (规划中)
**离线模式支持** - 网络中断时的体验
```tsx
// src/hooks/useOfflineMode.ts
export const useOfflineMode = () => {
  const [isOnline, setIsOnline] = useState(navigator.onLine)
  const [cachedQueries, setCachedQueries] = useState<CachedQuery[]>([])
  
  useEffect(() => {
    const handleOnline = () => {
      setIsOnline(true)
      // 同步离线期间的查询
      syncOfflineQueries()
    }
    
    const handleOffline = () => {
      setIsOnline(false)
      message.warning('网络连接中断，启用离线模式')
    }
    
    window.addEventListener('online', handleOnline)
    window.addEventListener('offline', handleOffline)
    
    return () => {
      window.removeEventListener('online', handleOnline)
      window.removeEventListener('offline', handleOffline)
    }
  }, [])
  
  return { isOnline, cachedQueries }
}
```

**性能监控仪表板** - 实时系统状态
```tsx
// src/components/PerformanceDashboard.tsx
export const PerformanceDashboard: React.FC = () => {
  const { data: metrics } = usePerformanceMetrics()
  
  return (
    <div className="performance-dashboard">
      <Row gutter={16}>
        <Col span={6}>
          <Statistic
            title="准确率"
            value={metrics?.accuracy || 0}
            suffix="%"
            valueStyle={{ color: metrics?.accuracy > 90 ? '#3f8600' : '#cf1322' }}
          />
        </Col>
        <Col span={6}>
          <Statistic
            title="平均响应时间"
            value={metrics?.avgResponseTime || 0}
            suffix="秒"
            precision={2}
          />
        </Col>
        <Col span={6}>
          <Statistic
            title="今日查询"
            value={metrics?.todayQueries || 0}
          />
        </Col>
        <Col span={6}>
          <Statistic
            title="用户满意度"
            value={metrics?.satisfaction || 0}
            suffix="/5"
            precision={1}
          />
        </Col>
      </Row>
      
      {/* 实时准确率趋势图 */}
      <AccuracyTrendChart data={metrics?.accuracyTrend} />
    </div>
  )
}
```

### 🔧 开发优先级
1. **立即实施**: 智能查询建议和实时反馈系统
2. **本周实施**: 查询历史智能化和结果展示优化
3. **下周实施**: 多模态查询界面集成
4. **月内完成**: 离线模式和性能监控仪表板

### 📊 预期效果
- **Phase 1完成**: 用户查询质量提升25%，反馈收集率提升至80%
- **Phase 2完成**: 多模态支持，用户体验满意度提升40%
- **Phase 3完成**: 离线可用性，系统监控透明度提升100%

详细技术方案和实施路线图请参考[根目录RAG优化方案](../CLAUDE.md#rag系统准确率优化方案)。

---

---

## 🚀 快速启动指南 (2025-08-14)

### 启动前端开发服务器
```bash
cd /home/bikeread/dev/rag_search/frontend

# 安装依赖（如需要）
npm install

# 启动开发服务器
npm run dev
```

### 访问苹果设计系统界面
- **主页面**: http://localhost:3000
- **仪表板**: http://localhost:3000/dashboard
- **AI对话**: http://localhost:3000/chat
- **文档管理**: http://localhost:3000/documents
- **个人中心**: http://localhost:3000/profile

### 设计系统文件
- **主题文件**: `src/styles/apple-theme.css`
- **全局样式**: `src/index.css`
- **主入口**: `src/main.tsx`

### 特色功能验证
1. **毛玻璃效果**: 侧边栏和导航栏
2. **渐变卡片**: 仪表板统计卡片
3. **动画交互**: 按钮hover和页面切换
4. **响应式设计**: 移动端和桌面端适配

---

**维护说明**: 本文档是RAG前端应用的核心开发指导，包含完整的组件架构、开发流程和苹果设计系统实现。配合[后端API文档](../backend/CLAUDE.md)和[Python服务文档](../python-services/CLAUDE.md)，确保Frontend在RAG准确率优化过程中的用户体验作用得到充分发挥。