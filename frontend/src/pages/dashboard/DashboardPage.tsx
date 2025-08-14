import React from 'react'
import { Row, Col, Card, Statistic, List, Avatar } from 'antd'
import { Link } from 'react-router-dom'
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

  // 计算统计数据
  const totalDocuments = documentsData?.pagination?.total || 0
  const todayQueries = queryData?.pagination?.total || 0
  const processingDocuments = documentsData?.documents?.filter(doc => doc.status === 'PROCESSING').length || 0
  const completionRate = totalDocuments > 0 ? Math.round((totalDocuments - processingDocuments) / totalDocuments * 100) : 0

  const stats = [
    {
      title: '总文档数',
      value: totalDocuments,
      icon: <FileTextOutlined style={{ fontSize: 24 }} />,
      gradient: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    },
    {
      title: '今日查询',
      value: todayQueries,
      icon: <MessageOutlined style={{ fontSize: 24 }} />,
      gradient: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
    },
    {
      title: '处理中文档',
      value: processingDocuments,
      icon: <ClockCircleOutlined style={{ fontSize: 24 }} />,
      gradient: 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)',
    },
    {
      title: '完成率',
      value: completionRate,
      suffix: '%',
      icon: <CheckCircleOutlined style={{ fontSize: 24 }} />,
      gradient: 'linear-gradient(135deg, #30cfd0 0%, #330867 100%)',
    },
  ]

  return (
    <div style={{ padding: '24px', background: '#f5f5f7', minHeight: '100vh' }}>
      {/* 标题区域 */}
      <div style={{ marginBottom: 32 }}>
        <h1 style={{ 
          fontSize: 34, 
          fontWeight: 600, 
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          marginBottom: 8
        }}>
          仪表板
        </h1>
        <p style={{ fontSize: 16, color: '#6e6e73' }}>欢迎回来，查看您的数据概览</p>
      </div>

      {/* 统计卡片 */}
      <Row gutter={[20, 20]}>
        {stats.map((stat, index) => (
          <Col xs={24} sm={12} lg={6} key={index}>
            <Card 
              bordered={false}
              style={{ 
                borderRadius: 16,
                boxShadow: '0 4px 24px rgba(0,0,0,0.06)',
                overflow: 'hidden',
                transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                cursor: 'pointer'
              }}
              bodyStyle={{ padding: 0 }}
              hoverable
            >
              <div style={{
                background: stat.gradient,
                padding: '20px 24px',
                color: 'white'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <div>
                    <div style={{ fontSize: 13, opacity: 0.9, marginBottom: 8 }}>{stat.title}</div>
                    <div style={{ fontSize: 32, fontWeight: 600, lineHeight: 1 }}>
                      {stat.value}{stat.suffix}
                    </div>
                  </div>
                  <div style={{ 
                    background: 'rgba(255,255,255,0.2)', 
                    borderRadius: 12,
                    padding: 12,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                  }}>
                    {React.cloneElement(stat.icon, { style: { ...stat.icon.props.style, color: 'white' } })}
                  </div>
                </div>
              </div>
            </Card>
          </Col>
        ))}
      </Row>

      <Row gutter={[20, 20]} style={{ marginTop: 32 }}>
        {/* 最近文档 */}
        <Col xs={24} lg={12}>
          <Card 
            bordered={false}
            style={{ 
              borderRadius: 16,
              boxShadow: '0 4px 24px rgba(0,0,0,0.06)',
              height: '100%'
            }}
            title={
              <div style={{ fontSize: 20, fontWeight: 600, color: '#1d1d1f' }}>最近上传</div>
            }
            extra={
              <Link to="/documents" style={{ 
                color: '#0071e3',
                fontSize: 14,
                fontWeight: 500,
                textDecoration: 'none'
              }}>
                查看全部 →
              </Link>
            }
          >
            <List
              itemLayout="horizontal"
              dataSource={documentsData?.documents || []}
              renderItem={(item) => (
                <List.Item style={{ 
                  borderBottom: '1px solid #f0f0f2',
                  padding: '16px 0'
                }}>
                  <List.Item.Meta
                    avatar={
                      <Avatar 
                        icon={<FileTextOutlined />} 
                        style={{ 
                          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                          borderRadius: 12
                        }}
                        size={44}
                      />
                    }
                    title={
                      <div style={{ 
                        fontSize: 15,
                        fontWeight: 500,
                        color: '#1d1d1f',
                        marginBottom: 4
                      }}>
                        {item.originalName}
                      </div>
                    }
                    description={
                      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                        <span style={{
                          padding: '2px 8px',
                          borderRadius: 6,
                          fontSize: 12,
                          background: item.status === 'COMPLETED' ? '#e8f5e9' : '#fff3e0',
                          color: item.status === 'COMPLETED' ? '#2e7d32' : '#f57c00'
                        }}>
                          {item.status === 'COMPLETED' ? '已完成' : '处理中'}
                        </span>
                        <span style={{ fontSize: 13, color: '#86868b' }}>
                          {new Date(item.createdAt).toLocaleDateString('zh-CN')}
                        </span>
                      </div>
                    }
                  />
                </List.Item>
              )}
              locale={{ emptyText: '暂无文档' }}
            />
          </Card>
        </Col>

        {/* 最近查询 */}
        <Col xs={24} lg={12}>
          <Card 
            bordered={false}
            style={{ 
              borderRadius: 16,
              boxShadow: '0 4px 24px rgba(0,0,0,0.06)',
              height: '100%'
            }}
            title={
              <div style={{ fontSize: 20, fontWeight: 600, color: '#1d1d1f' }}>最近查询</div>
            }
            extra={
              <Link to="/chat" style={{ 
                color: '#0071e3',
                fontSize: 14,
                fontWeight: 500,
                textDecoration: 'none'
              }}>
                开始对话 →
              </Link>
            }
          >
            <List
              itemLayout="horizontal"
              dataSource={queryData?.data || []}
              renderItem={(item) => (
                <List.Item style={{ 
                  borderBottom: '1px solid #f0f0f2',
                  padding: '16px 0'
                }}>
                  <List.Item.Meta
                    avatar={
                      <Avatar 
                        icon={<MessageOutlined />} 
                        style={{ 
                          background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
                          borderRadius: 12
                        }}
                        size={44}
                      />
                    }
                    title={
                      <div style={{ 
                        fontSize: 15,
                        fontWeight: 500,
                        color: '#1d1d1f',
                        marginBottom: 4
                      }}>
                        {item.text ? (item.text.length > 40 ? item.text.substring(0, 40) + '...' : item.text) : '查询内容'}
                      </div>
                    }
                    description={
                      <span style={{ fontSize: 13, color: '#86868b' }}>
                        {new Date(item.createdAt).toLocaleString('zh-CN')}
                      </span>
                    }
                  />
                </List.Item>
              )}
              locale={{ emptyText: '暂无查询记录' }}
            />
          </Card>
        </Col>
      </Row>
    </div>
  )
}