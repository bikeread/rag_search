import React from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import {
  Card,
  Tabs,
  Button,
  Descriptions,
  Tag,
  Skeleton,
  Progress,
  Table,
  Alert,
  Divider,
  Space,
  Typography
} from 'antd'
import {
  ArrowLeftOutlined,
  FileTextOutlined,
} from '@ant-design/icons'
import { documentService } from '@/services/documents'
import type { Document, DocumentChunk } from '@/types'
import { formatBytes, formatDate } from '@/utils/formatters'

const { Title, Text } = Typography

const DocumentHeader: React.FC<{
  document: Document
  onBack: () => void
}> = ({ document, onBack }) => (
  <div className="mb-6">
    <Space className="mb-4">
      <Button icon={<ArrowLeftOutlined />} onClick={onBack}>
        返回列表
      </Button>
    </Space>
    
    <div className="flex items-start justify-between">
      <div className="flex items-center space-x-3">
        <FileTextOutlined className="text-2xl text-blue-500" />
        <div>
          <Title level={3} className="mb-1">{document.originalName}</Title>
          <Space split={<Divider type="vertical" />}>
            <Text type="secondary">{formatBytes(document.size)}</Text>
            <Text type="secondary">块数: {document.chunksCount}</Text>
            <Text type="secondary">{formatDate(document.createdAt)}</Text>
          </Space>
        </div>
      </div>
      
      <Tag color={getStatusColor(document.status)}>
        {getStatusText(document.status)}
      </Tag>
    </div>
  </div>
)

const DocumentInfo: React.FC<{ document: Document }> = ({ document }) => (
  <Descriptions bordered column={2}>
    <Descriptions.Item label="文档ID">{document.id}</Descriptions.Item>
    <Descriptions.Item label="文件名">{document.filename}</Descriptions.Item>
    <Descriptions.Item label="原始名称">{document.originalName}</Descriptions.Item>
    <Descriptions.Item label="MIME类型">{document.mimeType}</Descriptions.Item>
    <Descriptions.Item label="文件大小">{formatBytes(document.size)}</Descriptions.Item>
    <Descriptions.Item label="文档块数">{document.chunksCount}</Descriptions.Item>
    <Descriptions.Item label="创建时间">{formatDate(document.createdAt)}</Descriptions.Item>
    <Descriptions.Item label="更新时间">{formatDate(document.updatedAt)}</Descriptions.Item>
    {document.processingStartedAt && (
      <Descriptions.Item label="处理开始时间">
        {formatDate(document.processingStartedAt)}
      </Descriptions.Item>
    )}
    {document.processingCompletedAt && (
      <Descriptions.Item label="处理完成时间">
        {formatDate(document.processingCompletedAt)}
      </Descriptions.Item>
    )}
    {document.errorMessage && (
      <Descriptions.Item label="错误信息" span={2}>
        <Text type="danger">{document.errorMessage}</Text>
      </Descriptions.Item>
    )}
  </Descriptions>
)

const ProcessingStatus: React.FC<{
  document: Document
  processingStatus?: {
    currentStep: string
    progress: number
    estimatedTime?: number
  }
}> = ({ document, processingStatus }) => {
  if (document.status !== 'PROCESSING') {
    return (
      <Alert
        type="info"
        message="文档未在处理中"
        description="此文档当前不在处理状态"
      />
    )
  }

  return (
    <div className="space-y-4">
      {processingStatus && (
        <Card>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <Text strong>当前步骤：{processingStatus.currentStep}</Text>
              <Text type="secondary">
                {processingStatus.estimatedTime && 
                  `预计剩余: ${Math.ceil(processingStatus.estimatedTime / 60)}分钟`
                }
              </Text>
            </div>
            <Progress 
              percent={processingStatus.progress} 
              status="active"
              strokeColor="#1890ff"
            />
          </div>
        </Card>
      )}
      
      <Alert
        type="info"
        message="处理进行中"
        description="文档正在处理中，处理完成后可以进行检索查询"
        showIcon
      />
    </div>
  )
}

const DocumentChunks: React.FC<{ chunks: DocumentChunk[] }> = ({ chunks }) => {
  const columns = [
    {
      title: '块索引',
      dataIndex: 'chunkIndex',
      key: 'chunkIndex',
      width: 80,
    },
    {
      title: '内容预览',
      dataIndex: 'content',
      key: 'content',
      render: (content: string) => (
        <div className="max-w-md">
          <Text ellipsis={{ tooltip: content }}>
            {content.substring(0, 100)}
            {content.length > 100 && '...'}
          </Text>
        </div>
      ),
    },
    {
      title: '创建时间',
      dataIndex: 'createdAt',
      key: 'createdAt',
      render: (date: string) => formatDate(date),
      width: 150,
    },
  ]

  return (
    <Table
      columns={columns}
      dataSource={chunks}
      rowKey="id"
      pagination={{
        pageSize: 10,
        showSizeChanger: false,
        showQuickJumper: true,
      }}
      size="small"
    />
  )
}

const DocumentDetailSkeleton: React.FC = () => (
  <div className="space-y-6">
    <Skeleton.Button active size="large" style={{ width: 100 }} />
    <Skeleton active paragraph={{ rows: 3 }} />
    <Skeleton active paragraph={{ rows: 6 }} />
  </div>
)

const ErrorFallback: React.FC<{
  error: Error
  onRetry: () => void
}> = ({ error, onRetry }) => (
  <Card>
    <div className="text-center space-y-4">
      <Alert
        type="error"
        message="加载失败"
        description={error.message}
        showIcon
      />
      <Button type="primary" onClick={onRetry}>
        重新加载
      </Button>
    </div>
  </Card>
)

const NotFound: React.FC = () => (
  <Card>
    <div className="text-center">
      <Alert
        type="warning"
        message="文档未找到"
        description="请检查文档ID是否正确"
        showIcon
      />
    </div>
  </Card>
)

const getStatusColor = (status: string) => {
  const statusColors = {
    PENDING: 'orange',
    PROCESSING: 'blue',
    COMPLETED: 'green',
    FAILED: 'red',
    DELETED: 'gray',
  }
  return statusColors[status as keyof typeof statusColors] || 'default'
}

const getStatusText = (status: string) => {
  const statusText = {
    PENDING: '等待处理',
    PROCESSING: '处理中',
    COMPLETED: '已完成',
    FAILED: '处理失败',
    DELETED: '已删除',
  }
  return statusText[status as keyof typeof statusText] || status
}


export const DocumentDetailPage: React.FC = () => {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  
  const { data: documentDetail, isLoading, error, refetch } = useQuery({
    queryKey: ['document', id],
    queryFn: () => documentService.getDocumentStatus(id!),
    enabled: !!id,
    refetchInterval: (data) => {
      return data?.document.status === 'PROCESSING' ? 2000 : false
    },
  })

  if (isLoading) return <DocumentDetailSkeleton />
  if (error) return <ErrorFallback error={error as Error} onRetry={refetch} />
  if (!documentDetail) return <NotFound />

  const { document, processingStatus, chunks } = documentDetail.data

  return (
    <div className="max-w-4xl mx-auto p-6">
      <DocumentHeader 
        document={document} 
        onBack={() => navigate('/documents')}
      />
      
      <Tabs
        defaultActiveKey="info"
        items={[
          {
            key: 'info',
            label: '基本信息',
            children: <DocumentInfo document={document} />,
          },
          {
            key: 'processing',
            label: '处理状态',
            children: (
              <ProcessingStatus 
                document={document}
                processingStatus={processingStatus}
              />
            ),
          },
          {
            key: 'chunks',
            label: `文档块 (${chunks?.length || 0})`,
            children: <DocumentChunks chunks={chunks || []} />,
          },
        ]}
      />
    </div>
  )
}