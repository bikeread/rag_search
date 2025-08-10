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