import React, { useState, useEffect } from 'react'
import { Button, Dropdown, Modal, message } from 'antd'
import type { MenuProps } from 'antd'
import { 
  ClearOutlined, 
  MessageOutlined, 
  DeleteOutlined,
  ExclamationCircleOutlined
} from '@ant-design/icons'
import { useChatStore } from '@/store/chatStore'
import { queryService } from '@/services/queries'

interface ClearChatButtonProps {
  className?: string
}

export const ClearChatButton: React.FC<ClearChatButtonProps> = ({ 
  className = "" 
}) => {
  const [historyCount, setHistoryCount] = useState(0)
  const { messages, clearMessages, clearAllHistory } = useChatStore()

  // 获取历史记录统计
  useEffect(() => {
    const fetchHistoryStats = async () => {
      try {
        const response = await queryService.getHistoryStats()
        setHistoryCount(response.total || 0)
      } catch (error) {
        console.error('Failed to fetch history stats:', error)
      }
    }
    fetchHistoryStats()
  }, [messages])

  // 清空当前会话（仅UI界面）
  const handleClearSession = () => {
    clearMessages()
    message.success('当前对话已清空')
  }

  // 清空全部历史记录（后端数据）
  const handleClearAllHistory = () => {
    Modal.confirm({
      title: '⚠️ 确认清空全部历史？',
      icon: <ExclamationCircleOutlined />,
      content: (
        <div>
          <p>将删除您的 <strong>{historyCount}</strong> 条查询记录</p>
          <p style={{ color: '#ff4d4f', marginTop: 8 }}>
            此操作不可恢复！
          </p>
        </div>
      ),
      okText: '确认清空',
      okType: 'danger',
      cancelText: '取消',
      onOk: async () => {
        try {
          // 使用store中的clearAllHistory方法
          await clearAllHistory()
          setHistoryCount(0)
          message.success('历史记录已清空')
        } catch (error) {
          console.error('Failed to clear history:', error)
          message.error('清空失败，请稍后重试')
        }
      }
    })
  }

  const menuItems: MenuProps['items'] = [
    {
      key: 'clear-session',
      label: (
        <div>
          <MessageOutlined className="mr-2" />
          <span>清空当前对话</span>
          <div className="text-xs text-gray-500 mt-1">
            仅清理界面显示
          </div>
        </div>
      ),
      onClick: handleClearSession
    },
    {
      type: 'divider'
    },
    {
      key: 'clear-all',
      label: (
        <div>
          <DeleteOutlined className="mr-2" />
          <span>清空全部历史</span>
          <div className="text-xs text-red-500 mt-1">
            永久删除所有记录 ({historyCount}条)
          </div>
        </div>
      ),
      danger: true,
      onClick: handleClearAllHistory
    }
  ]

  return (
    <Dropdown
      menu={{ items: menuItems }}
      trigger={['click']}
      placement="bottomRight"
    >
      <Button 
        icon={<ClearOutlined />}
        type="text"
        size="small"
        className={`${className}`}
      >
        清空
      </Button>
    </Dropdown>
  )
}