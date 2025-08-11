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
    <div className="max-w-2xl mx-auto">
      <Card title="个人中心" className="shadow-sm">
        <div className="text-center mb-6">
          <Avatar size={80} icon={<UserOutlined />} className="mb-4" />
          <h2 className="text-xl font-semibold">{user?.name || user?.email}</h2>
          <p className="text-gray-500">用户角色: {user?.role || 'USER'}</p>
        </div>

        <Divider />

        {!editing ? (
          <div className="space-y-4">
            <div>
              <label className="text-sm text-gray-600 block mb-1">用户名</label>
              <div className="text-base">{user?.name || '未设置'}</div>
            </div>
            <div>
              <label className="text-sm text-gray-600 block mb-1">邮箱</label>
              <div className="text-base">{user?.email}</div>
            </div>
            <div>
              <label className="text-sm text-gray-600 block mb-1">用户ID</label>
              <div className="text-base text-gray-500 font-mono text-sm">{user?.id}</div>
            </div>
            
            <div className="pt-4">
              <Button type="primary" icon={<EditOutlined />} onClick={handleEdit}>
                编辑资料
              </Button>
            </div>
          </div>
        ) : (
          <Form form={form} layout="vertical" className="space-y-4">
            <Form.Item
              name="name"
              label="用户名"
              rules={[
                { required: true, message: '请输入用户名' },
                { min: 2, max: 50, message: '用户名长度应为2-50字符' }
              ]}
            >
              <Input placeholder="请输入用户名" />
            </Form.Item>

            <Form.Item
              name="email"
              label="邮箱"
              rules={[
                { required: true, message: '请输入邮箱' },
                { type: 'email', message: '请输入有效的邮箱地址' }
              ]}
            >
              <Input placeholder="请输入邮箱" disabled />
            </Form.Item>

            <Form.Item>
              <Space>
                <Button 
                  type="primary" 
                  icon={<SaveOutlined />} 
                  onClick={handleSave}
                >
                  保存
                </Button>
                <Button onClick={handleCancel}>
                  取消
                </Button>
              </Space>
            </Form.Item>
          </Form>
        )}
      </Card>

      <Card title="账户统计" className="shadow-sm mt-6">
        <Spin spinning={documentsLoading}>
          <div className="grid grid-cols-2 gap-4">
            <div className="text-center p-4 bg-blue-50 rounded-lg">
              <div className="text-2xl font-bold text-blue-600 mb-2">
                <FileTextOutlined />
              </div>
              <div className="text-2xl font-bold text-blue-600">
                {documentsData?.pagination?.total || 0}
              </div>
              <div className="text-sm text-gray-600">上传文档数</div>
            </div>
            <div className="text-center p-4 bg-green-50 rounded-lg">
              <div className="text-2xl font-bold text-green-600 mb-2">
                <MessageOutlined />
              </div>
              <div className="text-2xl font-bold text-green-600">--</div>
              <div className="text-sm text-gray-600">AI对话次数</div>
            </div>
          </div>
        </Spin>
      </Card>
    </div>
  )
}