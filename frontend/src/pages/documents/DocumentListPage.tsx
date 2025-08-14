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
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <div style={{
              width: 36,
              height: 36,
              borderRadius: 8,
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <FileTextOutlined style={{ color: 'white', fontSize: 16 }} />
            </div>
            <Tooltip title={name}>
              <span style={{
                cursor: 'pointer',
                color: '#1d1d1f',
                fontWeight: 500,
                fontSize: 14,
                transition: 'color 0.2s ease'
              }}
              onMouseEnter={(e) => e.target.style.color = '#0071e3'}
              onMouseLeave={(e) => e.target.style.color = '#1d1d1f'}
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
            PENDING: { 
              gradient: 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)', 
              text: '等待处理' 
            },
            PROCESSING: { 
              gradient: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 
              text: '处理中' 
            },
            COMPLETED: { 
              gradient: 'linear-gradient(135deg, #30cfd0 0%, #330867 100%)', 
              text: '已完成' 
            },
            FAILED: { 
              gradient: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)', 
              text: '处理失败' 
            },
          }
          const config = statusConfig[status as keyof typeof statusConfig]
          return (
            <span style={{
              background: config?.gradient,
              color: 'white',
              padding: '4px 12px',
              borderRadius: 12,
              fontSize: 12,
              fontWeight: 500
            }}>
              {config?.text}
            </span>
          )
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
          <Space size={8}>
            <Tooltip title="查看详情">
              <Button
                type="text"
                icon={<EyeOutlined />}
                onClick={() => handleViewDocument(record)}
                style={{
                  borderRadius: 8,
                  width: 32,
                  height: 32,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}
              />
            </Tooltip>
            <Tooltip title="删除文档">
              <Button
                type="text"
                danger
                icon={<DeleteOutlined />}
                onClick={() => handleDeleteDocument(record)}
                disabled={record.status === 'PROCESSING'}
                style={{
                  borderRadius: 8,
                  width: 32,
                  height: 32,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}
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
    <div style={{ padding: '24px', background: '#f5f5f7', minHeight: '100vh' }}>
      {/* 页面标题和操作 */}
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        marginBottom: 32
      }}>
        <div>
          <h1 style={{
            fontSize: 34,
            fontWeight: 600,
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            marginBottom: 8
          }}>
            文档管理
          </h1>
          <p style={{ fontSize: 16, color: '#6e6e73', margin: 0 }}>
            管理您的文档库，支持 PDF、Word、文本和 Markdown 格式
          </p>
        </div>
        
        <Space size={12}>
          {selectedRows.length > 0 && (
            <Button 
              danger
              loading={batchLoading}
              onClick={handleBatchDelete}
              icon={<DeleteOutlined />}
              style={{
                borderRadius: 10,
                fontWeight: 500,
                height: 40
              }}
            >
              批量删除 ({selectedRows.length})
            </Button>
          )}
          
          <Dragger
            beforeUpload={handleUpload}
            showUploadList={false}
            accept=".pdf,.docx,.doc,.txt,.md"
            style={{ display: 'inline-block' }}
            disabled={uploadDocument.isPending}
          >
            <Button 
              type="primary" 
              icon={<CloudUploadOutlined />}
              loading={uploadDocument.isPending}
              style={{
                borderRadius: 10,
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                border: 'none',
                height: 40,
                paddingLeft: 20,
                paddingRight: 20,
                fontWeight: 500,
                boxShadow: '0 4px 16px rgba(102, 126, 234, 0.3)'
              }}
            >
              上传文档
            </Button>
          </Dragger>
        </Space>
      </div>

      {/* 上传进度显示 */}
      {uploadProgressEntries.length > 0 && (
        <Card 
          size="small" 
          title="上传进度"
          bordered={false}
          style={{
            borderRadius: 16,
            boxShadow: '0 4px 24px rgba(0,0,0,0.06)',
            marginBottom: 20
          }}
        >
          {uploadProgressEntries.map(([fileId, progress]) => (
            <div key={fileId} style={{ marginBottom: 16 }}>
              <div style={{ 
                display: 'flex', 
                justifyContent: 'space-between', 
                fontSize: 14,
                fontWeight: 500,
                marginBottom: 8
              }}>
                <span style={{ color: '#1d1d1f' }}>{progress.phase}</span>
                <span style={{ color: '#0071e3' }}>{progress.progress}%</span>
              </div>
              <Progress 
                percent={progress.progress} 
                status={progress.status === 'failed' ? 'exception' : 'active'}
                strokeColor={{
                  '0%': '#667eea',
                  '100%': '#764ba2'
                }}
                style={{ marginBottom: 0 }}
              />
            </div>
          ))}
        </Card>
      )}

      {/* 搜索和过滤 */}
      <div style={{
        display: 'flex',
        gap: 16,
        flexDirection: isMobile ? 'column' : 'row',
        alignItems: isMobile ? 'stretch' : 'center',
        marginBottom: 24,
        padding: '20px 24px',
        background: 'rgba(255,255,255,0.95)',
        backdropFilter: 'blur(20px)',
        borderRadius: 16,
        boxShadow: '0 4px 24px rgba(0,0,0,0.06)'
      }}>
        <Search
          placeholder="搜索文档名称"
          onChange={(e) => handleSearch(e.target.value)}
          style={{ 
            width: isMobile ? '100%' : 300,
          }}
          allowClear
          size="large"
        />
        <Select
          placeholder="筛选状态"
          value={filters.status}
          onChange={(status) => setFilters(prev => ({ ...prev, status, page: 1 }))}
          allowClear
          size="large"
          style={{ width: isMobile ? '100%' : 160 }}
        >
          <Select.Option value="PENDING">等待处理</Select.Option>
          <Select.Option value="PROCESSING">处理中</Select.Option>
          <Select.Option value="COMPLETED">已完成</Select.Option>
          <Select.Option value="FAILED">处理失败</Select.Option>
        </Select>
        
        <Space size={12}>
          <Button 
            icon={<ReloadOutlined />} 
            onClick={() => refetch()}
            style={{
              borderRadius: 8,
              fontWeight: 500,
              height: 40
            }}
          >
            刷新
          </Button>
          {selectedRows.length > 0 && (
            <Button 
              onClick={clearSelection}
              style={{
                borderRadius: 8,
                fontWeight: 500,
                height: 40
              }}
            >
              取消选择
            </Button>
          )}
        </Space>
      </div>

      {/* 文档表格/卡片 */}
      {documentsData?.documents.length === 0 && !isLoading ? (
        <div style={{
          background: 'rgba(255,255,255,0.95)',
          backdropFilter: 'blur(20px)',
          borderRadius: 16,
          boxShadow: '0 4px 24px rgba(0,0,0,0.06)',
          padding: '48px 24px',
          textAlign: 'center'
        }}>
          <Empty
            image={Empty.PRESENTED_IMAGE_SIMPLE}
            description={
              <div>
                {filters.search || filters.status ? (
                  <>
                    <p style={{ fontSize: 18, fontWeight: 500, color: '#1d1d1f', marginBottom: 8 }}>
                      未找到匹配的文档
                    </p>
                    <p style={{ fontSize: 14, color: '#6e6e73' }}>
                      尝试调整搜索条件或筛选器
                    </p>
                  </>
                ) : (
                  <>
                    <p style={{ fontSize: 18, fontWeight: 500, color: '#1d1d1f', marginBottom: 8 }}>
                      暂无文档
                    </p>
                    <p style={{ fontSize: 14, color: '#6e6e73' }}>
                      上传您的第一个文档开始使用
                    </p>
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
                style={{ marginTop: 24 }}
              >
                <Button 
                  type="primary" 
                  icon={<UploadOutlined />}
                  style={{
                    borderRadius: 10,
                    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                    border: 'none',
                    height: 44,
                    paddingLeft: 24,
                    paddingRight: 24,
                    fontWeight: 500,
                    boxShadow: '0 4px 16px rgba(102, 126, 234, 0.3)'
                  }}
                >
                  上传文档
                </Button>
              </Dragger>
            )}
          </Empty>
        </div>
      ) : (
        <div style={{
          background: 'rgba(255,255,255,0.95)',
          backdropFilter: 'blur(20px)',
          borderRadius: 16,
          boxShadow: '0 4px 24px rgba(0,0,0,0.06)',
          overflow: 'hidden'
        }}>
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
              style: { padding: '0 24px 24px' }
            }}
            scroll={{ x: isMobile ? 600 : undefined }}
            style={{ 
              background: 'transparent'
            }}
          />
        </div>
      )}
    </div>
  )
}

export default DocumentListPage