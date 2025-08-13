import { test, expect } from '@playwright/test';

const TEST_USER = {
  email: 'test@example.com',
  password: 'testpassword123'
};

test('直接测试消息发送功能', async ({ page }) => {
  console.log('开始直接消息发送测试');

  // 1. 先正常登录
  await page.goto('/login');
  
  // 等待登录页面加载
  await page.waitForLoadState('networkidle');
  
  // 填写登录信息
  await page.fill('input[name="email"], input[placeholder*="邮箱"]', TEST_USER.email);
  await page.fill('input[name="password"], input[type="password"]', TEST_USER.password);
  
  // 点击登录
  await page.click('button[type="submit"], .ant-btn-primary');
  
  // 等待重定向到首页或聊天页面
  await page.waitForTimeout(3000);
  
  console.log('登录后URL:', page.url());

  // 2. 导航到聊天页面
  await page.goto('/chat');
  await page.waitForLoadState('networkidle');
  await page.waitForTimeout(2000);
  
  console.log('聊天页面URL:', page.url());

  // 3. 查找并填写消息 - 使用更精确的选择器
  const messageInput = page.locator('textarea[placeholder*="问题"], .ant-input[placeholder*="问题"]');
  await messageInput.waitFor({ state: 'visible' });
  
  const testMessage = '这是一个测试消息';
  await messageInput.fill(testMessage);
  
  console.log('已输入测试消息');

  // 4. 监听网络请求
  let queryRequest = null;
  let queryResponse = null;

  page.on('request', request => {
    if (request.url().includes('/api/query')) {
      queryRequest = request;
      console.log('🔍 检测到查询请求:', request.url());
      console.log('请求方法:', request.method());
      console.log('请求头Authorization:', request.headers()['authorization'] ? '✅ 有token' : '❌ 无token');
    }
  });

  page.on('response', response => {
    if (response.url().includes('/api/query')) {
      queryResponse = response;
      console.log('📡 收到查询响应:', response.status(), response.url());
    }
  });

  // 5. 点击发送按钮 - 使用更精确的选择器
  const sendButton = page.locator('button:has-text("发送"), button[type="submit"]');
  await sendButton.waitFor({ state: 'visible' });
  await sendButton.click();
  
  console.log('点击发送按钮');

  // 6. 等待API请求和响应
  await page.waitForTimeout(5000);

  // 7. 检查结果
  if (queryRequest) {
    console.log('✅ 成功检测到API请求');
    console.log('请求URL:', queryRequest.url());
    
    if (queryResponse) {
      console.log('✅ 收到API响应，状态:', queryResponse.status());
      
      // 检查用户消息是否显示在界面上
      const userMessage = page.locator(`text=${testMessage}`);
      const isMessageVisible = await userMessage.isVisible({ timeout: 2000 });
      
      if (isMessageVisible) {
        console.log('✅ 用户消息已显示在界面上');
      } else {
        console.log('⚠️  用户消息未在界面上显示');
      }
      
      // 检查是否有AI回复
      await page.waitForTimeout(3000);
      const aiMessages = page.locator('.ant-card, [class*="message"]').filter({
        hasNotText: testMessage
      });
      const aiMessageCount = await aiMessages.count();
      
      if (aiMessageCount > 0) {
        console.log(`✅ 检测到 ${aiMessageCount} 个可能的AI回复`);
      } else {
        console.log('⚠️  未检测到AI回复');
      }
      
    } else {
      console.log('❌ API请求发送但未收到响应');
    }
  } else {
    console.log('❌ 未检测到API请求 - 可能存在问题：');
    console.log('1. 前端JavaScript错误');
    console.log('2. 事件监听器未正确绑定');
    console.log('3. 认证状态异常');
    
    // 检查控制台错误
    const logs = await page.evaluate(() => {
      return console.error.toString();
    });
    console.log('控制台信息:', logs);
  }

  // 8. 最终截图
  await page.screenshot({ path: 'direct-message-test.png', fullPage: true });
  console.log('测试完成，已保存截图');
});