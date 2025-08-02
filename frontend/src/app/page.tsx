'use client'

import React from 'react'
import { Layout, Typography, Row, Col, Card } from 'antd'
import { FileTextOutlined, SearchOutlined, RobotOutlined } from '@ant-design/icons'

const { Header, Content } = Layout
const { Title, Paragraph } = Typography

export default function Home() {
  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ background: '#fff', padding: '0 50px', boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }}>
        <div style={{ display: 'flex', alignItems: 'center', height: '64px' }}>
          <RobotOutlined style={{ fontSize: '24px', marginRight: '12px', color: '#1890ff' }} />
          <Title level={3} style={{ margin: 0, color: '#1890ff' }}>
            RAG Document System
          </Title>
        </div>
      </Header>
      
      <Content style={{ padding: '50px' }}>
        <div className="container">
          <Row gutter={[24, 24]} justify="center">
            <Col span={24} style={{ textAlign: 'center', marginBottom: '40px' }}>
              <Title level={1}>智能文档检索系统</Title>
              <Paragraph style={{ fontSize: '18px', color: '#666' }}>
                上传文档，使用AI进行智能问答和内容检索
              </Paragraph>
            </Col>
          </Row>
          
          <Row gutter={[24, 24]} justify="center">
            <Col xs={24} md={8}>
              <Card
                hoverable
                style={{ textAlign: 'center', height: '200px' }}
                cover={<FileTextOutlined style={{ fontSize: '48px', color: '#52c41a', padding: '20px' }} />}
              >
                <Card.Meta
                  title="文档上传"
                  description="支持PDF、Word、TXT等多种格式文档上传和处理"
                />
              </Card>
            </Col>
            
            <Col xs={24} md={8}>
              <Card
                hoverable
                style={{ textAlign: 'center', height: '200px' }}
                cover={<SearchOutlined style={{ fontSize: '48px', color: '#1890ff', padding: '20px' }} />}
              >
                <Card.Meta
                  title="智能检索"
                  description="基于向量相似度的语义检索，快速找到相关内容"
                />
              </Card>
            </Col>
            
            <Col xs={24} md={8}>
              <Card
                hoverable
                style={{ textAlign: 'center', height: '200px' }}
                cover={<RobotOutlined style={{ fontSize: '48px', color: '#722ed1', padding: '20px' }} />}
              >
                <Card.Meta
                  title="AI问答"
                  description="结合检索结果，提供准确的AI问答服务"
                />
              </Card>
            </Col>
          </Row>
        </div>
      </Content>
    </Layout>
  )
}