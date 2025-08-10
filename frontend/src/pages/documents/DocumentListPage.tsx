import React, { useState } from 'react'
import { 
  Table, 
  Button, 
  Input, 
  Select, 
  Space, 
  Tag, 
  Modal,
  Upload,
  Progress
} from 'antd'
import { 
  UploadOutlined, 
  DeleteOutlined,
  EyeOutlined,
  ReloadOutlined
} from '@ant-design/icons'
import { useDocuments, useDocumentDelete, useDocumentUpload } from '@/hooks/useDocuments'
import type { Document } from '@/types'
import { formatBytes, formatDate } from '@/utils/formatters'

const { Search } = Input
const { Dragger } = Upload

export const DocumentListPage: React.FC = () => {
  const [searchText, setSearchText] = useState('')
  const [statusFilter, setStatusFilter] = useState<string | undefined>()
  const [page, setPage] = useState(1)
  const limit = 10

  const { data: documentsData, isLoading, refetch } = useDocuments({
    page,
    limit,
    search: searchText,
    status: statusFilter,
  })

  const deleteDocument = useDocumentDelete()
  const uploadDocument = useDocumentUpload()

  const columns = [
    {
      title: '文档名称',
      dataIndex: 'originalName',
      key: 'originalName',
      ellipsis: true,
    },
    {
      title: '文件大小',
      dataIndex: 'size',
      key: 'size',
      render: (size: number) => formatBytes(size),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const statusConfig = {
          PENDING: { color: 'orange', text: '等待处理' },
          PROCESSING: { color: 'blue', text: '处理中' },
          COMPLETED: { color: 'green', text: '已完成' },
          FAILED: { color: 'red', text: '处理失败' },
        }
        const config = statusConfig[status as keyof typeof statusConfig]
        return <Tag color={config?.color}>{config?.text}</Tag>
      },
    },
    {
      title: '文档块数',
      dataIndex: 'chunksCount',
      key: 'chunksCount',
    },
    {
      title: '上传时间',
      dataIndex: 'createdAt',
      key: 'createdAt',
      render: (date: string) => formatDate(date),
    },
    {
      title: '操作',
      key: 'actions',
      render: (_: any, record: Document) => (
        <Space>
          <Button
            type="link"
            icon={<EyeOutlined />}
            onClick={() => handleViewDocument(record)}
          >
            查看
          </Button>
          <Button
            type="link"
            danger
            icon={<DeleteOutlined />}
            onClick={() => handleDeleteDocument(record)}
          >
            删除
          </Button>
        </Space>
      ),
    },
  ]

  const handleViewDocument = (document: Document) => {
    // 实现查看文档详情
    console.log('View document:', document)
  }

  const handleDeleteDocument = (document: Document) => {
    Modal.confirm({
      title: '确认删除',
      content: `确定要删除文档"${document.originalName}"吗？`,
      onOk: () => deleteDocument.mutate(document.id),
    })
  }

  const handleUpload = (file: File) => {
    uploadDocument.mutate({ file })
    return false // 阻止默认上传行为
  }

  return (
    <div className="space-y-6">
      {/* 页面标题和操作 */}
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">文档管理</h1>
        <Dragger
          beforeUpload={handleUpload}
          showUploadList={false}
          accept=".pdf,.docx,.doc,.txt,.md"
          className="w-auto"
        >
          <Button type="primary" icon={<UploadOutlined />}>
            上传文档
          </Button>
        </Dragger>
      </div>

      {/* 上传进度 */}
      {uploadDocument.isPending && (
        <Progress percent={50} status="active" />
      )}

      {/* 搜索和过滤 */}
      <div className="flex gap-4">
        <Search
          placeholder="搜索文档名称"
          value={searchText}
          onChange={(e) => setSearchText(e.target.value)}
          onSearch={() => refetch()}
          style={{ width: 300 }}
        />
        <Select
          placeholder="筛选状态"
          value={statusFilter}
          onChange={setStatusFilter}
          allowClear
          style={{ width: 150 }}
        >
          <Select.Option value="PENDING">等待处理</Select.Option>
          <Select.Option value="PROCESSING">处理中</Select.Option>
          <Select.Option value="COMPLETED">已完成</Select.Option>
          <Select.Option value="FAILED">处理失败</Select.Option>
        </Select>
        <Button icon={<ReloadOutlined />} onClick={() => refetch()}>
          刷新
        </Button>
      </div>

      {/* 文档表格 */}
      <Table
        columns={columns}
        dataSource={documentsData?.data || []}
        rowKey="id"
        loading={isLoading}
        pagination={{
          current: page,
          pageSize: limit,
          total: documentsData?.pagination.total || 0,
          onChange: setPage,
        }}
      />
    </div>
  )
}