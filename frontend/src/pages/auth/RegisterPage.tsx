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