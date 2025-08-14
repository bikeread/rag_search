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
      icon: <DashboardOutlined style={{ fontSize: 18 }} />,
      label: <span style={{ fontSize: 15, fontWeight: 500 }}>仪表板</span>,
    },
    {
      key: '/chat',
      icon: <MessageOutlined style={{ fontSize: 18 }} />,
      label: <span style={{ fontSize: 15, fontWeight: 500 }}>AI对话</span>,
    },
    {
      key: '/documents',
      icon: <FileTextOutlined style={{ fontSize: 18 }} />,
      label: <span style={{ fontSize: 15, fontWeight: 500 }}>文档管理</span>,
    },
    {
      key: '/profile',
      icon: <UserOutlined style={{ fontSize: 18 }} />,
      label: <span style={{ fontSize: 15, fontWeight: 500 }}>个人中心</span>,
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
    <Layout style={{ minHeight: '100vh', background: '#f5f5f7' }}>
      <Sider 
        width={260} 
        style={{
          background: 'rgba(255,255,255,0.95)',
          backdropFilter: 'blur(20px)',
          borderRight: '1px solid rgba(0,0,0,0.04)',
          position: 'fixed',
          height: '100vh',
          left: 0,
          top: 0,
          zIndex: 100
        }}
      >
        <div style={{
          height: 80,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          borderBottom: '1px solid rgba(0,0,0,0.04)',
          marginBottom: 16
        }}>
          <Text style={{
            fontSize: 20,
            fontWeight: 600,
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent'
          }}>
            RAG智能问答
          </Text>
        </div>
        <Menu
          mode="inline"
          selectedKeys={[location.pathname]}
          items={menuItems}
          style={{
            border: 'none',
            background: 'transparent',
            padding: '0 16px'
          }}
          onClick={({ key }) => navigate(key)}
        />
      </Sider>

      <Layout style={{ marginLeft: 260, background: 'transparent' }}>
        <Header style={{
          background: 'rgba(255,255,255,0.95)',
          backdropFilter: 'blur(20px)',
          padding: '0 32px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          borderBottom: '1px solid rgba(0,0,0,0.04)',
          position: 'sticky',
          top: 0,
          zIndex: 50
        }}>
          <div style={{
            fontSize: 18,
            fontWeight: 500,
            color: '#1d1d1f'
          }}>
            {menuItems.find(item => item.key === location.pathname)?.label || '首页'}
          </div>
          
          <Dropdown menu={{ items: userMenuItems }} placement="bottomRight">
            <Space style={{ cursor: 'pointer' }}>
              <div style={{
                width: 36,
                height: 36,
                borderRadius: 18,
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}>
                <UserOutlined style={{ color: 'white', fontSize: 16 }} />
              </div>
              <Text style={{ 
                fontWeight: 500,
                color: '#1d1d1f'
              }}>
                {user?.name || user?.email}
              </Text>
            </Space>
          </Dropdown>
        </Header>

        <Content style={{ background: 'transparent' }}>
          <Outlet />
        </Content>
      </Layout>
    </Layout>
  )
}