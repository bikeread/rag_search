import React from 'react'
import { Layout, Menu, Avatar, Dropdown, Space, Typography } from 'antd'
import { 
  UserOutlined, 
  FileTextOutlined, 
  MessageOutlined, 
  DashboardOutlined,
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

  const userMenuItems = [
    {
      key: 'profile',
      icon: <UserOutlined />,
      label: '个人设置',
      onClick: () => navigate('/profile'),
    },
    {
      type: 'divider' as const,
    },
    {
      key: 'logout',
      icon: <LogoutOutlined />,
      label: '退出登录',
      onClick: logout,
    },
  ]

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
          
          <Dropdown menu={{ items: userMenuItems }} placement="bottomRight">
            <Space className="cursor-pointer">
              <Avatar icon={<UserOutlined />} />
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