import React, { useState } from 'react'
import { Card, Form, Input, Button, Avatar, Space, Divider, message, Spin } from 'antd'
import { UserOutlined, EditOutlined, SaveOutlined, FileTextOutlined, MessageOutlined } from '@ant-design/icons'
import { useAuth } from '@/hooks/useAuth'
import { useQuery } from '@tanstack/react-query'
import { documentService } from '@/services/documents'

export const ProfilePage: React.FC = () => {
  const { user } = useAuth()
  const [editing, setEditing] = useState(false)
  const [form] = Form.useForm()

  // 获取用户统计信息
  const { data: documentsData, isLoading: documentsLoading } = useQuery({
    queryKey: ['documents', { page: 1, limit: 1 }], // 只需要总数
    queryFn: () => documentService.getDocuments({ page: 1, limit: 1 }),
    staleTime: 5 * 60 * 1000, // 5分钟缓存
  })

  const handleEdit = () => {
    setEditing(true)
    form.setFieldsValue({
      name: user?.name || '',
      email: user?.email || '',
    })
  }

  const handleSave = async () => {
    try {
      const values = await form.validateFields()
      console.log('保存用户信息:', values)
      // TODO: 实现用户信息更新API调用
      message.success('用户信息更新成功')
      setEditing(false)
    } catch (error) {
      console.error('表单验证失败:', error)
    }
  }

  const handleCancel = () => {
    setEditing(false)
    form.resetFields()
  }

  return (
    <div style={{ 
      maxWidth: 800, 
      margin: '0 auto', 
      padding: '24px',
      background: '#f5f5f7',
      minHeight: '100vh'
    }}>
      <Card 
        title={
          <span style={{
            fontSize: 24,
            fontWeight: 600,
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent'
          }}>
            个人中心
          </span>
        }
        bordered={false}
        style={{
          borderRadius: 16,
          boxShadow: '0 4px 24px rgba(0,0,0,0.06)',
          marginBottom: 24
        }}
      >
        <div style={{ textAlign: 'center', marginBottom: 32 }}>
          <div style={{
            width: 100,
            height: 100,
            borderRadius: 50,
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: '0 auto 16px',
            fontSize: 40,
            color: 'white'
          }}>
            <UserOutlined />
          </div>
          <h2 style={{ 
            fontSize: 20, 
            fontWeight: 600, 
            color: '#1d1d1f',
            marginBottom: 8
          }}>
            {user?.name || user?.email}
          </h2>
          <p style={{ color: '#6e6e73', fontSize: 14 }}>
            用户角色: {user?.role || 'USER'}
          </p>
        </div>

        <Divider style={{ borderColor: '#f0f0f2' }} />

        {!editing ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
            <div>
              <label style={{ 
                fontSize: 13, 
                color: '#6e6e73', 
                display: 'block', 
                marginBottom: 8,
                fontWeight: 500
              }}>
                用户名
              </label>
              <div style={{ 
                fontSize: 16, 
                color: '#1d1d1f',
                fontWeight: 500
              }}>
                {user?.name || '未设置'}
              </div>
            </div>
            <div>
              <label style={{ 
                fontSize: 13, 
                color: '#6e6e73', 
                display: 'block', 
                marginBottom: 8,
                fontWeight: 500
              }}>
                邮箱
              </label>
              <div style={{ 
                fontSize: 16, 
                color: '#1d1d1f',
                fontWeight: 500
              }}>
                {user?.email}
              </div>
            </div>
            <div>
              <label style={{ 
                fontSize: 13, 
                color: '#6e6e73', 
                display: 'block', 
                marginBottom: 8,
                fontWeight: 500
              }}>
                用户ID
              </label>
              <div style={{ 
                fontSize: 14, 
                color: '#86868b',
                fontFamily: 'Monaco, Consolas, monospace'
              }}>
                {user?.id}
              </div>
            </div>
            
            <div style={{ paddingTop: 16 }}>
              <Button 
                type="primary" 
                icon={<EditOutlined />} 
                onClick={handleEdit}
                style={{
                  borderRadius: 10,
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  border: 'none',
                  height: 44,
                  paddingLeft: 24,
                  paddingRight: 24,
                  fontWeight: 500,
                  boxShadow: '0 4px 16px rgba(102, 126, 234, 0.3)'
                }}
              >
                编辑资料
              </Button>
            </div>
          </div>
        ) : (
          <Form form={form} layout="vertical">
            <Form.Item
              name="name"
              label={
                <span style={{ 
                  fontSize: 14, 
                  fontWeight: 500, 
                  color: '#1d1d1f' 
                }}>
                  用户名
                </span>
              }
              rules={[
                { required: true, message: '请输入用户名' },
                { min: 2, max: 50, message: '用户名长度应为2-50字符' }
              ]}
              style={{ marginBottom: 24 }}
            >
              <Input 
                placeholder="请输入用户名" 
                size="large"
                style={{ borderRadius: 8 }}
              />
            </Form.Item>

            <Form.Item
              name="email"
              label={
                <span style={{ 
                  fontSize: 14, 
                  fontWeight: 500, 
                  color: '#1d1d1f' 
                }}>
                  邮箱
                </span>
              }
              rules={[
                { required: true, message: '请输入邮箱' },
                { type: 'email', message: '请输入有效的邮箱地址' }
              ]}
              style={{ marginBottom: 32 }}
            >
              <Input 
                placeholder="请输入邮箱" 
                disabled 
                size="large"
                style={{ borderRadius: 8 }}
              />
            </Form.Item>

            <Form.Item style={{ marginBottom: 0 }}>
              <Space size={12}>
                <Button 
                  type="primary" 
                  icon={<SaveOutlined />} 
                  onClick={handleSave}
                  style={{
                    borderRadius: 8,
                    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                    border: 'none',
                    height: 40,
                    paddingLeft: 20,
                    paddingRight: 20,
                    fontWeight: 500,
                    boxShadow: '0 4px 16px rgba(102, 126, 234, 0.3)'
                  }}
                >
                  保存
                </Button>
                <Button 
                  onClick={handleCancel}
                  style={{
                    borderRadius: 8,
                    height: 40,
                    paddingLeft: 20,
                    paddingRight: 20,
                    fontWeight: 500
                  }}
                >
                  取消
                </Button>
              </Space>
            </Form.Item>
          </Form>
        )}
      </Card>

      <Card 
        title={
          <span style={{
            fontSize: 20,
            fontWeight: 600,
            color: '#1d1d1f'
          }}>
            账户统计
          </span>
        }
        bordered={false}
        style={{
          borderRadius: 16,
          boxShadow: '0 4px 24px rgba(0,0,0,0.06)'
        }}
      >
        <Spin spinning={documentsLoading}>
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(2, 1fr)', 
            gap: 20 
          }}>
            <div style={{
              textAlign: 'center',
              padding: 24,
              borderRadius: 16,
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              color: 'white'
            }}>
              <div style={{ 
                fontSize: 32, 
                marginBottom: 12,
                display: 'flex',
                justifyContent: 'center'
              }}>
                <FileTextOutlined />
              </div>
              <div style={{ 
                fontSize: 28, 
                fontWeight: 600, 
                marginBottom: 8
              }}>
                {documentsData?.pagination?.total || 0}
              </div>
              <div style={{ 
                fontSize: 14, 
                opacity: 0.9
              }}>
                上传文档数
              </div>
            </div>
            <div style={{
              textAlign: 'center',
              padding: 24,
              borderRadius: 16,
              background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
              color: 'white'
            }}>
              <div style={{ 
                fontSize: 32, 
                marginBottom: 12,
                display: 'flex',
                justifyContent: 'center'
              }}>
                <MessageOutlined />
              </div>
              <div style={{ 
                fontSize: 28, 
                fontWeight: 600, 
                marginBottom: 8
              }}>
                --
              </div>
              <div style={{ 
                fontSize: 14, 
                opacity: 0.9
              }}>
                AI对话次数
              </div>
            </div>
          </div>
        </Spin>
      </Card>
    </div>
  )
}