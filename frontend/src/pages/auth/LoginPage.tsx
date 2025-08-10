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