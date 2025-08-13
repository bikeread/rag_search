import { describe, it, expect } from 'vitest'

describe('AI对话模块集成测试', () => {
  const API_BASE_URL = 'http://localhost:3001'
  
  // 获取有效的JWT token
  async function getAuthToken() {
    const response = await fetch(`${API_BASE_URL}/api/auth/signin`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        email: 'test@example.com',
        password: 'testpassword123'
      })
    })
    
    if (!response.ok) {
      throw new Error(`Failed to authenticate: ${response.status}`)
    }
    
    const data = await response.json()
    return data.token
  }
  
  describe('POST /api/query', () => {
    it('应该成功处理简单查询并返回结果', async () => {
      const token = await getAuthToken()
      
      const response = await fetch(`${API_BASE_URL}/api/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          query: 'test query',
          topK: 3
        })
      })
      
      expect(response.status).toBe(200)
      
      const data = await response.json()
      expect(data).toHaveProperty('queryId')
      expect(data).toHaveProperty('answer')
      expect(data).toHaveProperty('sources')
      expect(data).toHaveProperty('responseTime')
      expect(data).toHaveProperty('metadata')
      
      // 验证基本数据类型
      expect(typeof data.queryId).toBe('string')
      expect(typeof data.answer).toBe('string')
      expect(Array.isArray(data.sources)).toBe(true)
      expect(typeof data.responseTime).toBe('number')
      expect(typeof data.metadata).toBe('object')
      
      console.log('Query result:', JSON.stringify(data, null, 2))
    }, 30000) // 30秒超时，因为AI查询可能需要时间
    
    it('应该拒绝无效的查询参数', async () => {
      const token = await getAuthToken()
      
      const response = await fetch(`${API_BASE_URL}/api/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          query: '', // 空查询
          topK: 25  // 超出限制
        })
      })
      
      expect(response.status).toBe(400)
      
      const data = await response.json()
      expect(data).toHaveProperty('error')
      expect(data.error).toBe('Invalid query data')
    })
    
    it('应该拒绝未认证的请求', async () => {
      const response = await fetch(`${API_BASE_URL}/api/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          // 没有Authorization header
        },
        body: JSON.stringify({
          query: 'test query'
        })
      })
      
      expect(response.status).toBe(401)
    })
    
    it('应该只接受POST请求', async () => {
      const token = await getAuthToken()
      
      const response = await fetch(`${API_BASE_URL}/api/query`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })
      
      expect(response.status).toBe(405)
    })
  })
  
  describe('GET /api/query/history', () => {
    it('应该返回查询历史', async () => {
      const token = await getAuthToken()
      
      const response = await fetch(`${API_BASE_URL}/api/query/history`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })
      
      expect(response.status).toBe(200)
      
      const data = await response.json()
      expect(data).toHaveProperty('queries')
      expect(data).toHaveProperty('pagination')
      expect(Array.isArray(data.queries)).toBe(true)
      expect(typeof data.pagination).toBe('object')
    })
  })
})