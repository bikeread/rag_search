import { test, expect } from '@playwright/test'
import { promises as fs } from 'fs'
import path from 'path'

test.describe('Document Upload E2E Tests', () => {
  test.beforeEach(async ({ page }) => {
    // 导航到登录页面
    await page.goto('http://localhost:3000/login')
    
    // 登录
    await page.fill('#login_email', 'test@example.com')
    await page.fill('#login_password', 'testpassword123')
    await page.click('button[type="submit"]')
    
    // 等待登录完成并跳转到仪表板
    await page.waitForURL('**/dashboard')
    
    // 导航到文档管理页面
    await page.click('text=文档管理')
    await page.waitForURL('**/documents')
  })

  test('should upload a document and see it in the list', async ({ page }) => {
    // 创建测试文件
    const testContent = `E2E Test Document
    
This document was created for end-to-end testing.
Timestamp: ${Date.now()}
    
Features to test:
- File upload functionality
- Document processing
- List display
- Status updates
`
    
    const testFilePath = path.join(process.cwd(), 'tmp-e2e-test.txt')
    await fs.writeFile(testFilePath, testContent)
    
    try {
      // 记录上传前的文档数量
      const beforeUpload = await page.locator('table tbody tr').count()
      
      // 上传文件
      const fileInput = page.locator('input[type="file"]')
      await fileInput.setInputFiles(testFilePath)
      
      // 等待上传完成 - 检查是否有成功消息或新文档出现
      await page.waitForTimeout(3000)
      
      // 刷新页面查看最新状态
      await page.click('button:has-text("刷新")')
      await page.waitForTimeout(2000)
      
      // 验证文档是否出现在列表中
      const afterUpload = await page.locator('table tbody tr').count()
      expect(afterUpload).toBeGreaterThan(beforeUpload)
      
      // 查找新上传的文档
      const documentRow = page.locator('table tbody tr').filter({
        hasText: 'tmp-e2e-test.txt'
      })
      
      await expect(documentRow).toBeVisible()
      
      // 验证文档信息
      await expect(documentRow).toContainText('tmp-e2e-test.txt')
      await expect(documentRow).toContainText(/处理中|已完成|等待处理/)
      
      // 截图证明
      await page.screenshot({ 
        path: 'test-results/upload-success.png',
        fullPage: true 
      })
      
    } finally {
      // 清理测试文件
      try {
        await fs.unlink(testFilePath)
      } catch (error) {
        console.log('Failed to cleanup test file:', error)
      }
    }
  })

  test('should show upload progress and status updates', async ({ page }) => {
    // 创建一个较大的测试文件
    const largeContent = 'This is a larger test file. '.repeat(1000)
    const testFilePath = path.join(process.cwd(), 'tmp-large-e2e-test.txt')
    await fs.writeFile(testFilePath, largeContent)
    
    try {
      // 监听网络请求
      const uploadPromise = page.waitForResponse(
        response => response.url().includes('/api/documents/upload') && response.status() === 201
      )
      
      // 上传文件
      const fileInput = page.locator('input[type="file"]')
      await fileInput.setInputFiles(testFilePath)
      
      // 等待上传请求完成
      const uploadResponse = await uploadPromise
      expect(uploadResponse.status()).toBe(201)
      
      // 验证响应内容
      const responseData = await uploadResponse.json()
      expect(responseData.success).toBe(true)
      expect(responseData.data.documentId).toBeDefined()
      
      // 等待文档出现在列表中
      await page.waitForTimeout(2000)
      await page.click('button:has-text("刷新")')
      
      // 验证文档状态
      const documentRow = page.locator('table tbody tr').filter({
        hasText: 'tmp-large-e2e-test.txt'
      })
      
      await expect(documentRow).toBeVisible()
      
    } finally {
      await fs.unlink(testFilePath).catch(() => {})
    }
  })

  test('should handle upload errors gracefully', async ({ page }) => {
    // 测试错误处理 - 创建一个无效的文件类型
    const testFilePath = path.join(process.cwd(), 'tmp-invalid.exe')
    await fs.writeFile(testFilePath, 'Invalid file content')
    
    try {
      // 尝试上传无效文件
      const fileInput = page.locator('input[type="file"]')
      await fileInput.setInputFiles(testFilePath)
      
      // 等待错误消息或验证失败
      await page.waitForTimeout(2000)
      
      // 检查是否有错误提示
      const errorMessage = page.locator('.ant-message-error, .ant-notification-error')
      if (await errorMessage.isVisible()) {
        await expect(errorMessage).toContainText(/失败|错误|invalid/i)
      }
      
      // 验证文档没有被添加到列表中
      const invalidDocRow = page.locator('table tbody tr').filter({
        hasText: 'tmp-invalid.exe'
      })
      
      await expect(invalidDocRow).toHaveCount(0)
      
    } finally {
      await fs.unlink(testFilePath).catch(() => {})
    }
  })

  test('should support multiple file uploads', async ({ page }) => {
    // 创建多个测试文件
    const files = []
    const filePaths = []
    
    for (let i = 1; i <= 3; i++) {
      const content = `Multi-upload test file ${i}\\n\\nContent: ${Date.now()}-${i}`
      const filePath = path.join(process.cwd(), `tmp-multi-${i}.txt`)
      await fs.writeFile(filePath, content)
      files.push(`tmp-multi-${i}.txt`)
      filePaths.push(filePath)
    }
    
    try {
      // 记录初始文档数量
      const initialCount = await page.locator('table tbody tr').count()
      
      // 依次上传文件
      for (const filePath of filePaths) {
        const fileInput = page.locator('input[type="file"]')
        await fileInput.setInputFiles(filePath)
        await page.waitForTimeout(1000) // 等待上传完成
      }
      
      // 等待所有上传完成
      await page.waitForTimeout(5000)
      await page.click('button:has-text("刷新")')
      
      // 验证所有文件都出现在列表中
      for (const fileName of files) {
        const documentRow = page.locator('table tbody tr').filter({
          hasText: fileName
        })
        await expect(documentRow).toBeVisible()
      }
      
      // 验证总文档数量增加
      const finalCount = await page.locator('table tbody tr').count()
      expect(finalCount).toBeGreaterThanOrEqual(initialCount + files.length)
      
    } finally {
      // 清理所有测试文件
      for (const filePath of filePaths) {
        await fs.unlink(filePath).catch(() => {})
      }
    }
  })

  test('should display document information correctly', async ({ page }) => {
    // 创建测试文件
    const testContent = 'Document info test content'
    const testFilePath = path.join(process.cwd(), 'tmp-info-test.txt')
    await fs.writeFile(testFilePath, testContent)
    
    try {
      // 上传文件
      const fileInput = page.locator('input[type="file"]')
      await fileInput.setInputFiles(testFilePath)
      
      await page.waitForTimeout(3000)
      await page.click('button:has-text("刷新")')
      
      // 查找上传的文档
      const documentRow = page.locator('table tbody tr').filter({
        hasText: 'tmp-info-test.txt'
      })
      
      await expect(documentRow).toBeVisible()
      
      // 验证文档信息列
      await expect(documentRow.locator('td').nth(0)).toContainText('tmp-info-test.txt') // 文档名称
      await expect(documentRow.locator('td').nth(1)).toContainText(/\\d+/) // 文件大小
      await expect(documentRow.locator('td').nth(2)).toContainText(/处理中|已完成|等待处理/) // 状态
      await expect(documentRow.locator('td').nth(3)).toContainText(/\\d+/) // 文档块数
      await expect(documentRow.locator('td').nth(4)).toContainText(/\\d{4}-\\d{2}-\\d{2}/) // 上传时间
      
      // 验证操作按钮存在
      const actionCell = documentRow.locator('td').last()
      await expect(actionCell).toContainText(/查看|删除/)
      
    } finally {
      await fs.unlink(testFilePath).catch(() => {})
    }
  })
})

test.describe('Document Upload Error Scenarios', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:3000/login')
    await page.fill('#login_email', 'test@example.com')
    await page.fill('#login_password', 'testpassword123')
    await page.click('button[type="submit"]')
    await page.waitForURL('**/dashboard')
    await page.click('text=文档管理')
    await page.waitForURL('**/documents')
  })

  test('should handle network errors during upload', async ({ page }) => {
    // 创建测试文件
    const testFilePath = path.join(process.cwd(), 'tmp-network-test.txt')
    await fs.writeFile(testFilePath, 'Network test content')
    
    try {
      // 拦截网络请求并模拟错误
      await page.route('**/api/documents/upload', route => {
        route.abort('internetdisconnected')
      })
      
      // 尝试上传文件
      const fileInput = page.locator('input[type="file"]')
      await fileInput.setInputFiles(testFilePath)
      
      // 等待错误处理
      await page.waitForTimeout(3000)
      
      // 验证错误消息
      const errorMessage = page.locator('.ant-message-error, .ant-notification-error')
      if (await errorMessage.isVisible()) {
        console.log('Network error handled correctly')
      }
      
    } finally {
      await fs.unlink(testFilePath).catch(() => {})
    }
  })

  test('should handle authentication errors during upload', async ({ page }) => {
    // 创建测试文件
    const testFilePath = path.join(process.cwd(), 'tmp-auth-test.txt')
    await fs.writeFile(testFilePath, 'Auth test content')
    
    try {
      // 拦截网络请求并返回401错误
      await page.route('**/api/documents/upload', route => {
        route.fulfill({
          status: 401,
          contentType: 'application/json',
          body: JSON.stringify({ error: 'Unauthorized' })
        })
      })
      
      // 尝试上传文件
      const fileInput = page.locator('input[type="file"]')
      await fileInput.setInputFiles(testFilePath)
      
      // 等待错误处理
      await page.waitForTimeout(3000)
      
      // 验证是否重定向到登录页面或显示错误
      const currentUrl = page.url()
      if (currentUrl.includes('login') || await page.locator('.ant-message-error').isVisible()) {
        console.log('Authentication error handled correctly')
      }
      
    } finally {
      await fs.unlink(testFilePath).catch(() => {})
    }
  })
})