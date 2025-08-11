import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { chromium, Browser, Page } from 'playwright'
import path from 'path'

describe('文档管理 - 端到端测试', () => {
  let browser: Browser
  let page: Page
  
  const frontendUrl = 'http://localhost:3000'
  const backendUrl = 'http://localhost:3001'
  
  const testCredentials = {
    email: 'test@example.com',
    password: 'testpassword123',
  }

  beforeEach(async () => {
    browser = await chromium.launch({ headless: false })
    page = await browser.newPage()
    
    await page.goto(frontendUrl)
    
    await page.fill('input[type="email"]', testCredentials.email)
    await page.fill('input[type="password"]', testCredentials.password)
    await page.click('button[type="submit"]')
    
    await page.waitForURL(`${frontendUrl}/dashboard`)
    
    await page.click('text=文档管理')
    await page.waitForURL(`${frontendUrl}/documents`)
  })

  afterEach(async () => {
    if (browser) {
      await browser.close()
    }
  })

  describe('文档列表功能', () => {
    it('应该正确显示文档列表页面', async () => {
      await expect(page.locator('h1:text("文档管理")')).toBeVisible()
      
      await expect(page.locator('table')).toBeVisible()
      
      await expect(page.locator('button:text("上传文档")')).toBeVisible()
      
      await expect(page.locator('input[placeholder="搜索文档名称"]')).toBeVisible()
      
      await expect(page.locator('text=筛选状态')).toBeVisible()
    })

    it('应该显示正确的表格列', async () => {
      const tableHeaders = await page.locator('th').allTextContents()
      
      expect(tableHeaders).toContain('文档名称')
      expect(tableHeaders).toContain('文件大小')
      expect(tableHeaders).toContain('状态')
      expect(tableHeaders).toContain('文档块数')
      expect(tableHeaders).toContain('上传时间')
      expect(tableHeaders).toContain('操作')
    })

    it('应该加载并显示文档数据', async () => {
      await page.waitForSelector('table tbody tr')
      
      const rows = await page.locator('table tbody tr').count()
      expect(rows).toBeGreaterThan(0)
      
      const firstRow = page.locator('table tbody tr').first()
      await expect(firstRow.locator('td').first()).toContainText('.pdf')
    })

    it('应该正确显示文档状态标签', async () => {
      await page.waitForSelector('.ant-tag')
      
      const statusTags = await page.locator('.ant-tag').allTextContents()
      const validStatuses = ['等待处理', '处理中', '已完成', '处理失败']
      
      statusTags.forEach(status => {
        expect(validStatuses).toContain(status)
      })
    })
  })

  describe('搜索功能', () => {
    it('应该能够搜索文档', async () => {
      const searchInput = page.locator('input[placeholder="搜索文档名称"]')
      await searchInput.fill('测试')
      
      await page.click('.ant-input-search-button')
      
      await page.waitForResponse(response => 
        response.url().includes('/api/documents/list') && 
        response.url().includes('search=测试')
      )
      
      const tableRows = await page.locator('table tbody tr')
      const rowCount = await tableRows.count()
      
      if (rowCount > 0) {
        const firstRowText = await tableRows.first().textContent()
        expect(firstRowText).toContain('测试')
      }
    })

    it('应该清空搜索结果', async () => {
      const searchInput = page.locator('input[placeholder="搜索文档名称"]')
      await searchInput.fill('不存在的文档')
      await page.click('.ant-input-search-button')
      
      await page.waitForSelector('text=暂无数据')
      
      await searchInput.clear()
      await page.click('.ant-input-search-button')
      
      await page.waitForSelector('table tbody tr')
    })
  })

  describe('状态筛选功能', () => {
    it('应该能够按状态筛选文档', async () => {
      await page.click('.ant-select-selector:has-text("筛选状态")')
      
      await page.click('text=已完成')
      
      await page.waitForResponse(response => 
        response.url().includes('/api/documents/list') && 
        response.url().includes('status=COMPLETED')
      )
      
      const statusTags = await page.locator('.ant-tag').allTextContents()
      statusTags.forEach(status => {
        expect(status).toBe('已完成')
      })
    })

    it('应该能够清除状态筛选', async () => {
      await page.click('.ant-select-selector')
      await page.click('text=已完成')
      
      await page.click('.ant-select-clear')
      
      await page.waitForResponse(response => 
        response.url().includes('/api/documents/list') && 
        !response.url().includes('status=')
      )
    })
  })

  describe('分页功能', () => {
    it('应该显示分页组件', async () => {
      await expect(page.locator('.ant-pagination')).toBeVisible()
    })

    it('应该能够切换页面', async () => {
      const totalPages = await page.locator('.ant-pagination-total-text').textContent()
      
      if (totalPages && totalPages.includes('共') && parseInt(totalPages.match(/\d+/)?.[0] || '0') > 10) {
        await page.click('.ant-pagination-next')
        
        await page.waitForResponse(response => 
          response.url().includes('/api/documents/list') && 
          response.url().includes('page=2')
        )
        
        const currentPage = await page.locator('.ant-pagination-item-active').textContent()
        expect(currentPage).toBe('2')
      }
    })
  })

  describe('文档上传功能', () => {
    it('应该显示上传区域', async () => {
      await expect(page.locator('button:text("上传文档")')).toBeVisible()
    })

    it('应该能够上传PDF文件', async () => {
      const testFilePath = path.join(__dirname, '../fixtures/test.pdf')
      
      const fileChooserPromise = page.waitForEvent('filechooser')
      await page.click('button:text("上传文档")')
      const fileChooser = await fileChooserPromise
      await fileChooser.setFiles(testFilePath)
      
      await page.waitForResponse(response => 
        response.url().includes('/api/documents/upload') && 
        response.request().method() === 'POST'
      )
      
      await expect(page.locator('.ant-progress')).toBeVisible()
      
      await page.waitForSelector('.ant-progress', { state: 'hidden', timeout: 30000 })
      
      await page.waitForSelector('table tbody tr:has-text("test.pdf")')
    })

    it('应该拒绝不支持的文件类型', async () => {
      const invalidFilePath = path.join(__dirname, '../fixtures/test.jpg')
      
      const fileChooserPromise = page.waitForEvent('filechooser')
      await page.click('button:text("上传文档")')
      const fileChooser = await fileChooserPromise
      await fileChooser.setFiles(invalidFilePath)
      
      await expect(page.locator('.ant-message-error')).toBeVisible()
    })

    it('应该显示上传进度', async () => {
      const testFilePath = path.join(__dirname, '../fixtures/large.pdf')
      
      const fileChooserPromise = page.waitForEvent('filechooser')
      await page.click('button:text("上传文档")')
      const fileChooser = await fileChooserPromise
      await fileChooser.setFiles(testFilePath)
      
      await expect(page.locator('.ant-progress')).toBeVisible()
      
      const progress = await page.locator('.ant-progress-text').textContent()
      expect(progress).toContain('%')
    })
  })

  describe('文档删除功能', () => {
    it('应该显示删除确认对话框', async () => {
      await page.waitForSelector('table tbody tr')
      
      await page.click('button:text("删除")').first()
      
      await expect(page.locator('.ant-modal')).toBeVisible()
      await expect(page.locator('text=确认删除')).toBeVisible()
      await expect(page.locator('text=确定要删除文档')).toBeVisible()
    })

    it('应该能够取消删除', async () => {
      await page.waitForSelector('table tbody tr')
      const initialRowCount = await page.locator('table tbody tr').count()
      
      await page.click('button:text("删除")').first()
      await page.click('button:text("取消")')
      
      await expect(page.locator('.ant-modal')).not.toBeVisible()
      
      const currentRowCount = await page.locator('table tbody tr').count()
      expect(currentRowCount).toBe(initialRowCount)
    })

    it('应该能够确认删除', async () => {
      await page.waitForSelector('table tbody tr')
      const initialRowCount = await page.locator('table tbody tr').count()
      
      await page.click('button:text("删除")').first()
      await page.click('button:text("确定")')
      
      await page.waitForResponse(response => 
        response.url().includes('/api/documents/delete/') && 
        response.request().method() === 'DELETE'
      )
      
      await expect(page.locator('.ant-message-success')).toBeVisible()
      
      await page.waitForTimeout(1000)
      const newRowCount = await page.locator('table tbody tr').count()
      expect(newRowCount).toBeLessThanOrEqual(initialRowCount)
    })
  })

  describe('文档查看功能', () => {
    it('应该能够查看文档详情', async () => {
      await page.waitForSelector('table tbody tr')
      
      await page.click('button:text("查看")').first()
      
      const logs = []
      page.on('console', msg => logs.push(msg.text()))
      
      await page.waitForTimeout(500)
      
      expect(logs.some(log => log.includes('View document:'))).toBe(true)
    })
  })

  describe('刷新功能', () => {
    it('应该能够手动刷新列表', async () => {
      await page.click('button:text("刷新")')
      
      await page.waitForResponse(response => 
        response.url().includes('/api/documents/list')
      )
      
      await expect(page.locator('table')).toBeVisible()
    })

    it('应该在操作后自动刷新', async () => {
      await page.waitForSelector('table tbody tr')
      
      const testFilePath = path.join(__dirname, '../fixtures/test.pdf')
      const fileChooserPromise = page.waitForEvent('filechooser')
      await page.click('button:text("上传文档")')
      const fileChooser = await fileChooserPromise
      await fileChooser.setFiles(testFilePath)
      
      await page.waitForResponse(response => 
        response.url().includes('/api/documents/upload')
      )
      
      await page.waitForResponse(response => 
        response.url().includes('/api/documents/list')
      )
    })
  })

  describe('错误处理', () => {
    it('应该处理网络错误', async () => {
      await page.route('**/api/documents/list', route => {
        route.fulfill({
          status: 500,
          contentType: 'application/json',
          body: JSON.stringify({ error: 'Internal Server Error' }),
        })
      })
      
      await page.reload()
      
      await expect(page.locator('.ant-message-error')).toBeVisible()
    })

    it('应该处理认证过期', async () => {
      await page.route('**/api/documents/list', route => {
        route.fulfill({
          status: 401,
          contentType: 'application/json',
          body: JSON.stringify({ error: 'Unauthorized' }),
        })
      })
      
      await page.reload()
      
      await page.waitForURL(`${frontendUrl}/login`)
    })

    it('应该处理部分失败的操作', async () => {
      await page.route('**/api/documents/delete/*', route => {
        route.fulfill({
          status: 403,
          contentType: 'application/json',
          body: JSON.stringify({ error: 'Forbidden' }),
        })
      })
      
      await page.waitForSelector('table tbody tr')
      await page.click('button:text("删除")').first()
      await page.click('button:text("确定")')
      
      await expect(page.locator('.ant-message-error')).toBeVisible()
    })
  })

  describe('用户体验测试', () => {
    it('应该有合理的加载时间', async () => {
      const startTime = Date.now()
      
      await page.reload()
      await page.waitForSelector('table tbody tr')
      
      const loadTime = Date.now() - startTime
      expect(loadTime).toBeLessThan(3000)
    })

    it('应该支持键盘导航', async () => {
      await page.keyboard.press('Tab')
      
      const focused = await page.evaluate(() => document.activeElement?.tagName)
      expect(['INPUT', 'BUTTON', 'SELECT']).toContain(focused)
    })

    it('应该有无障碍支持', async () => {
      const table = page.locator('table')
      await expect(table).toHaveAttribute('role', 'table')
      
      const searchInput = page.locator('input[placeholder="搜索文档名称"]')
      await expect(searchInput).toHaveAttribute('aria-label')
    })
  })

  describe('完整工作流测试', () => {
    it('应该完成完整的文档管理流程', async () => {
      const initialRowCount = await page.locator('table tbody tr').count()
      
      const testFilePath = path.join(__dirname, '../fixtures/workflow-test.pdf')
      const fileChooserPromise = page.waitForEvent('filechooser')
      await page.click('button:text("上传文档")')
      const fileChooser = await fileChooserPromise
      await fileChooser.setFiles(testFilePath)
      
      await page.waitForResponse(response => 
        response.url().includes('/api/documents/upload')
      )
      
      await expect(page.locator('.ant-message-success')).toBeVisible()
      
      await page.waitForSelector('table tbody tr:has-text("workflow-test.pdf")')
      
      const searchInput = page.locator('input[placeholder="搜索文档名称"]')
      await searchInput.fill('workflow-test')
      await page.click('.ant-input-search-button')
      
      await page.waitForSelector('table tbody tr:has-text("workflow-test.pdf")')
      
      await page.click('button:text("查看")').first()
      
      await page.waitForTimeout(500)
      
      await page.click('button:text("删除")').first()
      await page.click('button:text("确定")')
      
      await page.waitForResponse(response => 
        response.url().includes('/api/documents/delete/')
      )
      
      await expect(page.locator('.ant-message-success')).toBeVisible()
      
      await searchInput.clear()
      await page.click('.ant-input-search-button')
      
      const finalRowCount = await page.locator('table tbody tr').count()
      expect(finalRowCount).toBeLessThanOrEqual(initialRowCount)
    })

    it('应该处理并发操作', async () => {
      const uploadFiles = [
        '../fixtures/concurrent1.pdf',
        '../fixtures/concurrent2.pdf',
        '../fixtures/concurrent3.pdf',
      ]

      for (const filePath of uploadFiles) {
        const fullPath = path.join(__dirname, filePath)
        const fileChooserPromise = page.waitForEvent('filechooser')
        await page.click('button:text("上传文档")')
        const fileChooser = await fileChooserPromise
        await fileChooser.setFiles(fullPath)
        
        await page.waitForTimeout(500)
      }
      
      await page.waitForTimeout(2000)
      
      const successMessages = await page.locator('.ant-message-success').count()
      expect(successMessages).toBeGreaterThan(0)
    })
  })

  describe('移动端适配测试', () => {
    it('应该在移动设备上正确显示', async () => {
      await page.setViewportSize({ width: 375, height: 667 })
      
      await page.reload()
      await page.waitForSelector('table')
      
      await expect(page.locator('table')).toBeVisible()
      
      const tableContainer = page.locator('.ant-table-container')
      const hasHorizontalScroll = await tableContainer.evaluate(el => 
        el.scrollWidth > el.clientWidth
      )
      expect(hasHorizontalScroll).toBe(true)
    })

    it('应该支持触摸操作', async () => {
      await page.setViewportSize({ width: 375, height: 667 })
      
      await page.touchscreen.tap(100, 100)
      
      const searchInput = page.locator('input[placeholder="搜索文档名称"]')
      await searchInput.tap()
      
      await expect(searchInput).toBeFocused()
    })
  })

  describe('性能测试', () => {
    it('应该在合理时间内加载大量文档', async () => {
      await page.route('**/api/documents/list', route => {
        const mockLargeDataset = {
          data: Array.from({ length: 100 }, (_, i) => ({
            id: `doc-${i}`,
            filename: `test${i}.pdf`,
            originalName: `测试文档${i}.pdf`,
            mimeType: 'application/pdf',
            size: 1048576,
            status: 'COMPLETED',
            chunksCount: 25,
            createdAt: '2025-08-10T14:00:00.000Z',
            updatedAt: '2025-08-10T14:05:00.000Z',
          })),
          pagination: {
            page: 1,
            limit: 10,
            total: 100,
            totalPages: 10,
            hasNext: true,
            hasPrev: false,
          },
        }
        
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(mockLargeDataset),
        })
      })
      
      const startTime = Date.now()
      await page.reload()
      await page.waitForSelector('table tbody tr')
      const renderTime = Date.now() - startTime
      
      expect(renderTime).toBeLessThan(2000)
      
      const rows = await page.locator('table tbody tr').count()
      expect(rows).toBe(10)
    })

    it('应该处理大文件上传', async () => {
      const largeFilePath = path.join(__dirname, '../fixtures/large.pdf')
      
      const fileChooserPromise = page.waitForEvent('filechooser')
      await page.click('button:text("上传文档")')
      const fileChooser = await fileChooserPromise
      await fileChooser.setFiles(largeFilePath)
      
      await expect(page.locator('.ant-progress')).toBeVisible()
      
      await page.waitForResponse(
        response => response.url().includes('/api/documents/upload'),
        { timeout: 60000 }
      )
    })
  })

  describe('数据一致性测试', () => {
    it('应该保持前后端数据一致', async () => {
      await page.waitForSelector('table tbody tr')
      
      const frontendData = await page.locator('table tbody tr').first().textContent()
      
      const response = await page.request.get(`${backendUrl}/api/documents/list`, {
        headers: {
          'Authorization': `Bearer ${await page.evaluate(() => localStorage.getItem('token'))}`,
        },
      })
      
      const backendData = await response.json()
      const firstDocument = backendData.documents[0]
      
      expect(frontendData).toContain(firstDocument.originalName)
      expect(frontendData).toContain(firstDocument.status === 'COMPLETED' ? '已完成' : firstDocument.status)
    })

    it('应该实时更新文档状态', async () => {
      const processingDoc = page.locator('table tbody tr:has(.ant-tag:text("处理中"))')
      
      if (await processingDoc.count() > 0) {
        await page.waitForTimeout(5000)
        
        await page.click('button:text("刷新")')
        await page.waitForResponse(response => 
          response.url().includes('/api/documents/list')
        )
        
        const statusAfterRefresh = await processingDoc.locator('.ant-tag').textContent()
        expect(['处理中', '已完成', '处理失败']).toContain(statusAfterRefresh || '')
      }
    })
  })
})