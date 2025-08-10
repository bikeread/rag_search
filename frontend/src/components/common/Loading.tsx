import React from 'react'
import { Spin } from 'antd'

interface LoadingProps {
  size?: 'small' | 'default' | 'large'
  tip?: string
  className?: string
}

export const Loading: React.FC<LoadingProps> = ({ 
  size = 'default', 
  tip = '加载中...', 
  className = '' 
}) => {
  return (
    <div className={`flex items-center justify-center p-8 ${className}`}>
      <Spin size={size} tip={tip} />
    </div>
  )
}