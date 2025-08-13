import { test, expect, type Page } from '@playwright/test'

const TEST_USER = {
  email: 'test@example.com',
  password: 'testpassword123'
}

test.describe('AI聊天认证测试', () => {
  test('应该重定向未认证用户到登录页面', async ({ page }) => {
    // 直接访问聊天页面
    await page.goto('/chat')
    
    // 应该被重定向到登录页面
    await page.waitForURL(/\/login/, { timeout: 5000 })
    
    // 验证登录表单存在
    const emailInput = page.locator('input[placeholder*="邮箱"], input[name="email"]')
    const passwordInput = page.locator('input[placeholder*="密码"], input[type="password"]')
    const submitButton = page.locator('button[type="submit"], button:has-text("登录")')
    
    await expect(emailInput).toBeVisible()
    await expect(passwordInput).toBeVisible()  
    await expect(submitButton).toBeVisible()
  })

  test('应该能够成功登录并访问聊天页面', async ({ page }) => {
    // 访问登录页面
    await page.goto('/login')
    
    // 填写登录表单
    await page.fill('input[placeholder*="邮箱"], input[name="email"]', TEST_USER.email)
    await page.fill('input[placeholder*="密码"], input[type="password"]', TEST_USER.password)
    
    // 点击登录
    await page.click('button[type="submit"], button:has-text("登录")')
    
    // 等待登录成功
    await page.waitForURL(/\/(dashboard|chat|documents)/, { timeout: 10000 })
    
    // 现在应该能访问聊天页面
    await page.goto('/chat')
    
    // 验证聊天页面加载成功
    await expect(page).toHaveTitle(/RAG|Chat|AI|问答/)
    
    // 验证聊天界面元素存在
    const messageInput = page.locator('input[placeholder*="问题"], input[placeholder*="消息"], textarea[placeholder*="问题"], textarea[placeholder*="消息"]').first()
    await expect(messageInput).toBeVisible()
  })

  test('应该拒绝无效的登录凭据', async ({ page }) => {
    await page.goto('/login')
    
    // 使用错误的密码
    await page.fill('input[placeholder*="邮箱"], input[name="email"]', TEST_USER.email)
    await page.fill('input[placeholder*="密码"], input[type="password"]', 'wrongpassword')
    
    await page.click('button[type="submit"], button:has-text("登录")')
    
    // 应该显示错误消息
    const errorMessage = page.locator('.error, .alert-error, .error-message, text=错误, text=失败').first()
    await expect(errorMessage).toBeVisible({ timeout: 5000 })
    
    // 应该还在登录页面
    await expect(page).toHaveURL(/\/login/)
  })

  test('登录后应该显示用户信息', async ({ page }) => {
    // 登录
    await page.goto('/login')
    await page.fill('input[placeholder*="邮箱"], input[name="email"]', TEST_USER.email)
    await page.fill('input[placeholder*="密码"], input[type="password"]', TEST_USER.password)
    await page.click('button[type="submit"], button:has-text("登录")')
    
    await page.waitForURL(/\/(dashboard|chat|documents)/, { timeout: 10000 })
    
    // 检查用户信息是否显示
    const userInfo = page.locator('.user-info, .profile, .user-name, .avatar').first()
    
    // 或者检查是否有退出登录按钮
    const logoutButton = page.locator('button:has-text("退出"), button:has-text("登出"), button:has-text("Logout"), a:has-text("退出")').first()
    
    const hasUserInfo = await userInfo.isVisible()
    const hasLogout = await logoutButton.isVisible()
    
    expect(hasUserInfo || hasLogout).toBeTruthy()
  })

  test('应该能够退出登录', async ({ page }) => {
    // 先登录
    await page.goto('/login')
    await page.fill('input[type="email"], input[name="email"]', TEST_USER.email)
    await page.fill('input[type="password"], input[name="password"]', TEST_USER.password)
    await page.click('button[type="submit"], button:has-text("登录"), button:has-text("Login")')
    
    await page.waitForURL(/\/(dashboard|chat|documents)/, { timeout: 10000 })
    
    // 查找退出按钮
    const logoutButton = page.locator('button:has-text("退出"), button:has-text("登出"), button:has-text("Logout"), a:has-text("退出")').first()
    
    if (await logoutButton.isVisible()) {
      await logoutButton.click()
      
      // 应该被重定向到登录页面
      await page.waitForURL(/\/login/, { timeout: 5000 })
      
      // 现在访问聊天页面应该被重定向
      await page.goto('/chat')
      await page.waitForURL(/\/login/, { timeout: 5000 })
    }
  })

  test('应该正确处理会话过期', async ({ page }) => {
    // 登录
    await page.goto('/login')
    await page.fill('input[placeholder*="邮箱"], input[name="email"]', TEST_USER.email)
    await page.fill('input[placeholder*="密码"], input[type="password"]', TEST_USER.password)
    await page.click('button[type="submit"], button:has-text("登录")')
    
    await page.waitForURL(/\/(dashboard|chat|documents)/, { timeout: 10000 })
    
    // 清除localStorage中的token来模拟会话过期
    await page.evaluate(() => {
      localStorage.removeItem('token')
      localStorage.removeItem('auth-storage')
    })
    
    // 尝试访问聊天页面
    await page.goto('/chat')
    
    // 应该被重定向到登录页面
    await page.waitForURL(/\/login/, { timeout: 5000 })
  })
})

test.describe('保护路由测试', () => {
  const protectedRoutes = ['/chat', '/dashboard', '/documents']
  
  protectedRoutes.forEach(route => {
    test(`应该保护路由 ${route}`, async ({ page }) => {
      await page.goto(route)
      
      // 应该被重定向到登录页面
      await page.waitForURL(/\/login/, { timeout: 5000 })
      
      // 验证登录表单存在
      const loginForm = page.locator('form, .login-form').first()
      const emailInput = page.locator('input[placeholder*="邮箱"], input[name="email"]')
      
      await expect(emailInput).toBeVisible()
    })
  })
})