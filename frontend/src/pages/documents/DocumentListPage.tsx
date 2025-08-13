import React, { useState, useCallback, useMemo } from 'react'
import { 
  Table, 
  Button, 
  Input, 
  Select, 
  Space, 
  Tag, 
  Modal,
  Upload,
  Progress,
  Card,
  Empty,
  Tooltip,
} from 'antd'
import { 
  UploadOutlined, 
  DeleteOutlined,
  EyeOutlined,
  ReloadOutlined,
  FileTextOutlined,
  CloudUploadOutlined
} from '@ant-design/icons'
import { 
  useDocuments, 
  useDocumentDelete, 
  useDocumentUpload, 
  useBatchOperations,
  useResponsiveView
} from '@/hooks/useDocuments'
import type { Document, DocumentListParams } from '@/types'
import { formatBytes, formatDate } from '@/utils/formatters'
import { useNavigate } from 'react-router-dom'
// import { debounce } from 'lodash-es'
import { toast } from 'react-hot-toast'

const { Search } = Input
const { Dragger } = Upload

const DocumentListPage: React.FC = () => {
  const navigate = useNavigate()
  const { isMobile } = useResponsiveView()
  
  const [filters, setFilters] = useState<DocumentListParams>({
    search: '',
    status: undefined,
    page: 1,
    limit: 10,
  })

  const { data: documentsData, isLoading, refetch } = useDocuments(filters)
  const deleteDocument = useDocumentDelete()
  const uploadDocument = useDocumentUpload()
  const {
    selectedRows,
    batchLoading,
    rowSelection,
    batchDelete,
    clearSelection,
  } = useBatchOperations()
  
  // 简单搜索处理
  const handleSearch = useCallback((searchText: string) => {
    setFilters(prev => ({ ...prev, search: searchText, page: 1 }))
  }, [])

  const getColumns = useCallback(() => {
    const baseColumns = [
      {
        title: '文档名称',
        dataIndex: 'originalName',
        key: 'originalName',
        ellipsis: true,
        width: isMobile ? 200 : 300,
        render: (name: string, record: Document) => (
          <div className="flex items-center space-x-2">
            <FileTextOutlined className="text-blue-500" />
            <Tooltip title={name}>
              <span className="cursor-pointer hover:text-blue-500" 
                    onClick={() => handleViewDocument(record)}>
                {name}
              </span>
            </Tooltip>
          </div>
        ),
      },
      {
        title: '大小',
        dataIndex: 'size',
        key: 'size',
        render: (size: number) => formatBytes(size),
        width: 100,
        responsive: ['md'] as any,
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
        width: 120,
      },
      {
        title: '文档块',
        dataIndex: 'chunksCount',
        key: 'chunksCount',
        width: 80,
        responsive: ['lg'] as any,
      },
      {
        title: '上传时间',
        dataIndex: 'createdAt',
        key: 'createdAt',
        render: (date: string) => formatDate(date),
        width: 150,
        responsive: ['md'] as any,
      },
      {
        title: '操作',
        key: 'actions',
        render: (_: any, record: Document) => (
          <Space size="small">
            <Tooltip title="查看详情">
              <Button
                type="text"
                icon={<EyeOutlined />}
                onClick={() => handleViewDocument(record)}
              />
            </Tooltip>
            <Tooltip title="删除文档">
              <Button
                type="text"
                danger
                icon={<DeleteOutlined />}
                onClick={() => handleDeleteDocument(record)}
                disabled={record.status === 'PROCESSING'}
              />
            </Tooltip>
          </Space>
        ),
        width: 100,
        fixed: !isMobile ? ('right' as const) : undefined,
      },
    ]
    
    // 移动端列精简
    if (isMobile) {
      return baseColumns.filter(col => 
        ['originalName', 'status', 'actions'].includes(col.key)
      )
    }
    
    return baseColumns
  }, [isMobile])

  const handleViewDocument = (document: Document) => {
    navigate(`/documents/${document.id}`)
  }

  const handleDeleteDocument = (document: Document) => {
    Modal.confirm({
      title: '确认删除',
      content: `确定要删除文档"${document.originalName}"吗？`,
      onOk: () => deleteDocument.mutate(document.id),
    })
  }

  const handleUpload = (file: File) => {
    // 文件预验证
    const allowedTypes = [
      'application/pdf',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'application/msword',
      'text/plain',
      'text/markdown',
      'text/x-markdown'
    ]
    
    // 获取文件扩展名
    const fileName = file.name.toLowerCase()
    const fileExtension = fileName.substring(fileName.lastIndexOf('.') + 1)
    
    // 支持的文件扩展名
    const allowedExtensions = ['pdf', 'docx', 'doc', 'txt', 'md']
    
    // 检查文件扩展名（优先级更高，更可靠）
    if (allowedExtensions.includes(fileExtension)) {
      // 扩展名匹配，直接通过
      console.log(`文件 ${file.name} 通过扩展名验证: ${fileExtension}`)
    } else if (allowedTypes.includes(file.type)) {
      // MIME类型匹配作为备选方案
      console.log(`文件 ${file.name} 通过MIME类型验证: ${file.type}`)
    } else {
      console.error(`文件验证失败: 名称=${file.name}, 扩展名=${fileExtension}, MIME=${file.type}`)
      toast.error(`不支持的文件格式。支持的格式：PDF, Word文档, 文本文件, Markdown文件`)
      return false
    }
    
    if (file.size > 50 * 1024 * 1024) { // 50MB
      toast.error('文件大小超过限制（50MB）')
      return false
    }
    
    uploadDocument.mutate(file)
    return false
  }
  
  const handleBatchDelete = () => {
    Modal.confirm({
      title: '确认批量删除',
      content: `确定要删除选中的 ${selectedRows.length} 个文档吗？`,
      onOk: () => batchDelete(selectedRows),
    })
  }

  const uploadProgressEntries = Object.entries(uploadDocument.uploadProgress || {})
  
  return (
    <div className="space-y-6">
      {/* 页面标题和操作 */}
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">文档管理</h1>
        
        <Space>
          {selectedRows.length > 0 && (
            <Button 
              danger
              loading={batchLoading}
              onClick={handleBatchDelete}
              icon={<DeleteOutlined />}
            >
              批量删除 ({selectedRows.length})
            </Button>
          )}
          
          <Dragger
            beforeUpload={handleUpload}
            showUploadList={false}
            accept=".pdf,.docx,.doc,.txt,.md"
            className="inline-block"
            disabled={uploadDocument.isPending}
          >
            <Button 
              type="primary" 
              icon={<CloudUploadOutlined />}
              loading={uploadDocument.isPending}
            >
              上传文档
            </Button>
          </Dragger>
        </Space>
      </div>

      {/* 上传进度显示 */}
      {uploadProgressEntries.length > 0 && (
        <Card size="small" title="上传进度">
          {uploadProgressEntries.map(([fileId, progress]) => (
            <div key={fileId} className="mb-2">
              <div className="flex justify-between text-sm mb-1">
                <span>{progress.phase}</span>
                <span>{progress.progress}%</span>
              </div>
              <Progress 
                percent={progress.progress} 
                status={progress.status === 'failed' ? 'exception' : 'active'}
                size="small"
              />
            </div>
          ))}
        </Card>
      )}

      {/* 搜索和过滤 */}
      <div className={`flex gap-4 ${isMobile ? 'flex-col' : 'flex-row'}`}>
        <Search
          placeholder="搜索文档名称"
          onChange={(e) => handleSearch(e.target.value)}
          style={{ width: isMobile ? '100%' : 300 }}
          allowClear
        />
        <Select
          placeholder="筛选状态"
          value={filters.status}
          onChange={(status) => setFilters(prev => ({ ...prev, status, page: 1 }))}
          allowClear
          style={{ width: isMobile ? '100%' : 150 }}
        >
          <Select.Option value="PENDING">等待处理</Select.Option>
          <Select.Option value="PROCESSING">处理中</Select.Option>
          <Select.Option value="COMPLETED">已完成</Select.Option>
          <Select.Option value="FAILED">处理失败</Select.Option>
        </Select>
        
        <Space>
          <Button icon={<ReloadOutlined />} onClick={() => refetch()}>
            刷新
          </Button>
          {selectedRows.length > 0 && (
            <Button onClick={clearSelection}>
              取消选择
            </Button>
          )}
        </Space>
      </div>

      {/* 文档表格/卡片 */}
      {documentsData?.documents.length === 0 && !isLoading ? (
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description={
            <div className="text-center">
              {filters.search || filters.status ? (
                <>
                  <p className="text-lg mb-2">未找到匹配的文档</p>
                  <p className="text-sm text-gray-500">
                    尝试调整搜索条件或筛选器
                  </p>
                </>
              ) : (
                <>
                  <p className="text-lg mb-2">暂无文档</p>
                  <p className="text-sm text-gray-500">上传您的第一个文档开始使用</p>
                </>
              )}
            </div>
          }
        >
          {/* 只有在没有筛选条件时才显示上传按钮 */}
          {!filters.search && !filters.status && (
            <Dragger
              beforeUpload={handleUpload}
              showUploadList={false}
              accept=".pdf,.docx,.doc,.txt,.md"
            >
              <Button type="primary" icon={<UploadOutlined />}>
                上传文档
              </Button>
            </Dragger>
          )}
        </Empty>
      ) : (
        <Table
          columns={getColumns()}
          dataSource={documentsData?.documents || []}
          rowKey="id"
          loading={isLoading}
          rowSelection={rowSelection}
          pagination={{
            current: filters.page,
            pageSize: filters.limit,
            total: documentsData?.pagination.total || 0,
            onChange: (page) => setFilters(prev => ({ ...prev, page })),
            showSizeChanger: !isMobile,
            onShowSizeChange: (current, size) => 
              setFilters(prev => ({ ...prev, limit: size, page: 1 })),
            showQuickJumper: !isMobile,
            showTotal: (total, range) => 
              `第 ${range[0]}-${range[1]} 条，共 ${total} 条`,
          }}
          scroll={{ x: isMobile ? 600 : undefined }}
        />
      )}
    </div>
  )
}

export default DocumentListPage