import React from 'react'
import { Row, Col, Card, Statistic, List, Avatar } from 'antd'
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

  const stats = [
    {
      title: '总文档数',
      value: documentsData?.pagination.total || 0,
      icon: <FileTextOutlined className="text-blue-500" />,
    },
    {
      title: '今日查询',
      value: 23,
      icon: <MessageOutlined className="text-green-500" />,
    },
    {
      title: '处理中文档',
      value: 2,
      icon: <ClockCircleOutlined className="text-orange-500" />,
    },
    {
      title: '完成率',
      value: 95,
      suffix: '%',
      icon: <CheckCircleOutlined className="text-purple-500" />,
    },
  ]

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">仪表板</h1>

      {/* 统计卡片 */}
      <Row gutter={[16, 16]}>
        {stats.map((stat, index) => (
          <Col xs={24} sm={12} lg={6} key={index}>
            <Card>
              <Statistic
                title={stat.title}
                value={stat.value}
                suffix={stat.suffix}
                prefix={stat.icon}
              />
            </Card>
          </Col>
        ))}
      </Row>

      <Row gutter={[16, 16]}>
        {/* 最近文档 */}
        <Col xs={24} lg={12}>
          <Card title="最近上传" extra={<a href="/documents">查看全部</a>}>
            <List
              itemLayout="horizontal"
              dataSource={documentsData?.data || []}
              renderItem={(item) => (
                <List.Item>
                  <List.Item.Meta
                    avatar={<Avatar icon={<FileTextOutlined />} />}
                    title={item.originalName}
                    description={`${item.status} • ${new Date(item.createdAt).toLocaleDateString()}`}
                  />
                </List.Item>
              )}
            />
          </Card>
        </Col>

        {/* 最近查询 */}
        <Col xs={24} lg={12}>
          <Card title="最近查询" extra={<a href="/chat">开始对话</a>}>
            <List
              itemLayout="horizontal"
              dataSource={queryData?.data || []}
              renderItem={(item) => (
                <List.Item>
                  <List.Item.Meta
                    avatar={<Avatar icon={<MessageOutlined />} />}
                    title={item.text.substring(0, 30) + '...'}
                    description={new Date(item.createdAt).toLocaleString()}
                  />
                </List.Item>
              )}
            />
          </Card>
        </Col>
      </Row>
    </div>
  )
}