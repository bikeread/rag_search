import { test, expect, type Page } from '@playwright/test'

// 测试用户凭据
const TEST_USER = {
  email: 'test@example.com',
  password: 'testpassword123'
}

// 登录辅助函数
async function loginUser(page: Page) {
  // 导航到登录页面
  await page.goto('/login')
  
  // 填写登录表单
  await page.fill('input[placeholder*="邮箱"], input[name="email"]', TEST_USER.email)
  await page.fill('input[placeholder*="密码"], input[type="password"]', TEST_USER.password)
  
  // 点击登录按钮
  await page.click('button[type="submit"], button:has-text("登录")')
  
  // 等待登录成功并跳转
  await page.waitForURL(/\/(dashboard|chat|documents)/, { timeout: 10000 })
}

// 导航到聊天页面
async function navigateToChat(page: Page) {
  // 尝试多种方式导航到聊天页面
  try {
    await page.click('nav a[href="/chat"], a:has-text("AI问答"), a:has-text("Chat"), a:has-text("聊天")')
  } catch {
    // 如果找不到导航链接，直接访问聊天页面
    await page.goto('/chat')
  }
  
  await page.waitForLoadState('networkidle')
}

test.describe('AI对话功能测试', () => {
  test.beforeEach(async ({ page }) => {
    // 每个测试前都先登录
    await loginUser(page)
    await navigateToChat(page)
  })

  test('应该显示AI聊天界面', async ({ page }) => {
    // 验证页面标题
    await expect(page).toHaveTitle(/RAG|Chat|AI|问答/)
    
    // 验证聊天界面元素存在
    const chatContainer = page.locator('.chat-container, .chat-interface, .messages-container').first()
    await expect(chatContainer).toBeVisible()
    
    // 验证输入框存在
    const messageInput = page.locator('input[placeholder*="问题"], input[placeholder*="消息"], textarea[placeholder*="问题"], textarea[placeholder*="消息"]').first()
    await expect(messageInput).toBeVisible()
    
    // 验证发送按钮存在
    const sendButton = page.locator('button:has-text("发送"), button:has-text("Send"), button[type="submit"]').first()
    await expect(sendButton).toBeVisible()
  })

  test('应该能够发送消息并接收AI回复', async ({ page }) => {
    const testMessage = '什么是RAG？'
    
    // 定位输入框和发送按钮
    const messageInput = page.locator('input[placeholder*="问题"], input[placeholder*="消息"], textarea[placeholder*="问题"], textarea[placeholder*="消息"]').first()
    const sendButton = page.locator('button:has-text("发送"), button:has-text("Send"), button[type="submit"]').first()
    
    // 发送消息
    await messageInput.fill(testMessage)
    await sendButton.click()
    
    // 验证用户消息显示
    const userMessage = page.locator('.message-user, .user-message, [data-role="user"]').last()
    await expect(userMessage).toContainText(testMessage)
    
    // 等待AI回复（可能需要较长时间）
    const aiReply = page.locator('.message-assistant, .ai-message, .bot-message, [data-role="assistant"]').last()
    await expect(aiReply).toBeVisible({ timeout: 60000 })
    
    // 验证AI回复包含内容
    const replyText = await aiReply.textContent()
    expect(replyText).toBeTruthy()
    expect(replyText!.length).toBeGreaterThan(10)
    
    console.log('AI回复:', replyText)
  })

  test('应该显示消息时间戳', async ({ page }) => {
    const testMessage = '显示时间戳测试'
    
    const messageInput = page.locator('input[placeholder*="问题"], input[placeholder*="消息"], textarea[placeholder*="问题"], textarea[placeholder*="消息"]').first()
    const sendButton = page.locator('button:has-text("发送"), button:has-text("Send"), button[type="submit"]').first()
    
    await messageInput.fill(testMessage)
    await sendButton.click()
    
    // 检查时间戳
    const timestamp = page.locator('.timestamp, .message-time, .time').last()
    await expect(timestamp).toBeVisible()
  })

  test('应该能够处理长消息', async ({ page }) => {
    const longMessage = '这是一个很长的测试消息，用来验证AI聊天系统是否能够正确处理较长的输入文本。' + 
                       '这个消息包含了多个句子，用于测试系统的处理能力和响应质量。' +
                       '我们希望AI能够理解并给出相应的回复。'
    
    const messageInput = page.locator('input[placeholder*="问题"], input[placeholder*="消息"], textarea[placeholder*="问题"], textarea[placeholder*="消息"]').first()
    const sendButton = page.locator('button:has-text("发送"), button:has-text("Send"), button[type="submit"]').first()
    
    await messageInput.fill(longMessage)
    await sendButton.click()
    
    // 验证长消息正确显示
    const userMessage = page.locator('.message-user, .user-message, [data-role="user"]').last()
    await expect(userMessage).toContainText(longMessage.substring(0, 50))
    
    // 等待AI回复
    const aiReply = page.locator('.message-assistant, .ai-message, .bot-message, [data-role="assistant"]').last()
    await expect(aiReply).toBeVisible({ timeout: 60000 })
  })

  test('应该支持快速连续发送多条消息', async ({ page }) => {
    const messages = [
      '第一条测试消息',
      '第二条测试消息', 
      '第三条测试消息'
    ]
    
    const messageInput = page.locator('input[placeholder*="问题"], input[placeholder*="消息"], textarea[placeholder*="问题"], textarea[placeholder*="消息"]').first()
    const sendButton = page.locator('button:has-text("发送"), button:has-text("Send"), button[type="submit"]').first()
    
    for (const message of messages) {
      await messageInput.fill(message)
      await sendButton.click()
      
      // 等待消息出现
      await expect(page.locator('.message-user, .user-message, [data-role="user"]').last()).toContainText(message)
      
      // 短暂延迟
      await page.waitForTimeout(1000)
    }
    
    // 验证所有消息都显示了
    for (const message of messages) {
      await expect(page.locator(`text=${message}`).first()).toBeVisible()
    }
  })

  test('应该显示加载状态', async ({ page }) => {
    const testMessage = '测试加载状态'
    
    const messageInput = page.locator('input[placeholder*="问题"], input[placeholder*="消息"], textarea[placeholder*="问题"], textarea[placeholder*="消息"]').first()
    const sendButton = page.locator('button:has-text("发送"), button:has-text("Send"), button[type="submit"]').first()
    
    await messageInput.fill(testMessage)
    await sendButton.click()
    
    // 检查加载指示器
    const loadingIndicator = page.locator('.loading, .spinner, .thinking, .processing, [data-loading="true"]').first()
    
    // 在AI回复之前应该显示加载状态
    try {
      await expect(loadingIndicator).toBeVisible({ timeout: 5000 })
    } catch {
      // 如果没有找到加载指示器，至少应该有发送按钮被禁用或显示加载状态
      const disabledButton = page.locator('button[disabled], button:has-text("发送中"), button:has-text("处理中")')
      await expect(disabledButton).toBeVisible({ timeout: 2000 })
    }
  })

  test('应该正确处理空消息输入', async ({ page }) => {
    const sendButton = page.locator('button:has-text("发送"), button:has-text("Send"), button[type="submit"]').first()
    
    // 尝试发送空消息
    await sendButton.click()
    
    // 发送按钮应该被禁用或显示错误提示
    const isButtonDisabled = await sendButton.isDisabled()
    if (!isButtonDisabled) {
      // 如果按钮未被禁用，应该显示错误提示
      const errorMessage = page.locator('.error-message, .warning, .alert').first()
      await expect(errorMessage).toBeVisible({ timeout: 2000 })
    }
  })

  test('应该能够清空对话历史', async ({ page }) => {
    // 首先发送一条消息
    const testMessage = '测试清空功能'
    
    const messageInput = page.locator('input[placeholder*="问题"], input[placeholder*="消息"], textarea[placeholder*="问题"], textarea[placeholder*="消息"]').first()
    const sendButton = page.locator('button:has-text("发送"), button:has-text("Send"), button[type="submit"]').first()
    
    await messageInput.fill(testMessage)
    await sendButton.click()
    
    // 等待消息显示
    await expect(page.locator('.message-user, .user-message, [data-role="user"]').last()).toBeVisible()
    
    // 寻找清空按钮
    const clearButton = page.locator('button:has-text("清空"), button:has-text("Clear"), button:has-text("重置"), .clear-chat').first()
    
    if (await clearButton.isVisible()) {
      await clearButton.click()
      
      // 可能需要确认
      const confirmButton = page.locator('button:has-text("确认"), button:has-text("确定"), button:has-text("OK")').first()
      if (await confirmButton.isVisible()) {
        await confirmButton.click()
      }
      
      // 验证消息被清空
      await expect(page.locator('.message-user, .user-message, [data-role="user"]')).toHaveCount(0)
    }
  })
})

test.describe('聊天历史功能测试', () => {
  test.beforeEach(async ({ page }) => {
    await loginUser(page)
  })

  test('应该能够查看聊天历史', async ({ page }) => {
    // 导航到历史页面或在聊天页面查看历史
    try {
      await page.click('a[href="/history"], a:has-text("历史"), a:has-text("History")')
    } catch {
      // 如果没有历史页面，在聊天页面查看侧边栏或历史按钮
      await navigateToChat(page)
      await page.click('.history-toggle, .sidebar-toggle, button:has-text("历史")')
    }
    
    // 验证历史记录存在
    const historyItems = page.locator('.history-item, .chat-history-item, .conversation-item')
    
    if (await historyItems.count() > 0) {
      await expect(historyItems.first()).toBeVisible()
    } else {
      // 如果没有历史记录，应该显示空状态
      const emptyState = page.locator('.empty-history, .no-history, text=没有历史记录')
      await expect(emptyState).toBeVisible()
    }
  })
})

test.describe('错误处理测试', () => {
  test.beforeEach(async ({ page }) => {
    await loginUser(page)
    await navigateToChat(page)
  })

  test('应该正确处理网络错误', async ({ page }) => {
    // 模拟网络离线
    await page.context().setOffline(true)
    
    const testMessage = '网络错误测试'
    
    const messageInput = page.locator('input[placeholder*="问题"], input[placeholder*="消息"], textarea[placeholder*="问题"], textarea[placeholder*="消息"]').first()
    const sendButton = page.locator('button:has-text("发送"), button:has-text("Send"), button[type="submit"]').first()
    
    await messageInput.fill(testMessage)
    await sendButton.click()
    
    // 应该显示错误提示
    const errorMessage = page.locator('.error, .network-error, .offline-message, text=网络错误').first()
    await expect(errorMessage).toBeVisible({ timeout: 10000 })
    
    // 恢复网络连接
    await page.context().setOffline(false)
  })

  test('应该在AI服务不可用时显示错误提示', async ({ page }) => {
    // 这个测试依赖于实际的错误情况，
    // 如果AI回复"生成回答时出错"，应该显示友好的错误提示
    const testMessage = '服务错误测试'
    
    const messageInput = page.locator('input[placeholder*="问题"], input[placeholder*="消息"], textarea[placeholder*="问题"], textarea[placeholder*="消息"]').first()
    const sendButton = page.locator('button:has-text("发送"), button:has-text("Send"), button[type="submit"]').first()
    
    await messageInput.fill(testMessage)
    await sendButton.click()
    
    // 等待AI回复
    const aiReply = page.locator('.message-assistant, .ai-message, .bot-message, [data-role="assistant"]').last()
    await expect(aiReply).toBeVisible({ timeout: 60000 })
    
    // 检查是否包含错误信息
    const replyText = await aiReply.textContent()
    if (replyText && replyText.includes('错误')) {
      // 应该有重试按钮或错误提示
      const retryButton = page.locator('button:has-text("重试"), button:has-text("Retry")').first()
      const errorIndicator = page.locator('.error-indicator, .alert, .warning').first()
      
      const hasRetry = await retryButton.isVisible()
      const hasError = await errorIndicator.isVisible()
      
      expect(hasRetry || hasError).toBeTruthy()
    }
  })
})