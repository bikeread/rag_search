import { test, expect } from '@playwright/test';

const TEST_USER = {
  email: 'test@example.com',
  password: 'testpassword123'
};

test('绕过前端登录直接测试消息发送', async ({ page }) => {
  console.log('开始绕过前端登录的消息发送测试');

  // 1. 通过API直接获取token
  const response = await fetch('http://localhost:3001/api/auth/signin', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(TEST_USER),
  });

  const authData = await response.json();
  console.log('API登录状态:', response.status);
  console.log('获取到token:', authData.token ? '✅' : '❌');

  if (!authData.token) {
    console.log('❌ 无法获取token，跳过认证测试聊天界面');
    // 继续测试，但不期望认证成功
  }

  // 2. 访问聊天页面并注入认证信息
  await page.goto('/chat');
  
  if (authData.token) {
    // 注入认证信息到localStorage
    await page.evaluate((authInfo) => {
      localStorage.setItem('token', authInfo.token);
      localStorage.setItem('auth-storage', JSON.stringify({
        state: {
          user: authInfo.user,
          token: authInfo.token,
          isAuthenticated: true,
          isLoading: false
        },
        version: 0
      }));
    }, authData);

    console.log('✅ 已注入认证信息到localStorage');

    // 刷新页面让认证生效
    await page.reload();
    await page.waitForLoadState('networkidle');
  }

  console.log('当前URL:', page.url());

  // 3. 查找聊天界面元素
  console.log('查找聊天界面元素...');

  // 等待页面完全加载
  await page.waitForTimeout(2000);

  // 查找消息输入框的多种可能选择器
  const inputSelectors = [
    'input[placeholder*="问题"]',
    'input[placeholder*="消息"]', 
    'textarea[placeholder*="问题"]',
    'textarea[placeholder*="消息"]',
    '.ant-input',
    'input[type="text"]',
    'textarea'
  ];

  let messageInput = null;
  for (const selector of inputSelectors) {
    try {
      const element = page.locator(selector).last(); // 使用last()获取最后一个，通常是聊天输入框
      if (await element.isVisible({ timeout: 1000 })) {
        messageInput = element;
        console.log(`✅ 找到消息输入框: ${selector}`);
        break;
      }
    } catch (e) {
      // 继续查找
    }
  }

  // 查找发送按钮
  const buttonSelectors = [
    'button:has-text("发送")',
    'button:has-text("Send")',
    'button[type="submit"]',
    '.ant-btn-primary',
    'button:last-child'
  ];

  let sendButton = null;
  for (const selector of buttonSelectors) {
    try {
      const element = page.locator(selector).last();
      if (await element.isVisible({ timeout: 1000 })) {
        sendButton = element;
        console.log(`✅ 找到发送按钮: ${selector}`);
        break;
      }
    } catch (e) {
      // 继续查找
    }
  }

  // 4. 测试消息发送功能
  if (messageInput && sendButton) {
    console.log('✅ 找到聊天界面元素，开始测试消息发送');

    const testMessage = '你好，这是一个测试消息。请回复确认收到。';
    
    // 清空输入框并输入消息
    await messageInput.clear();
    await messageInput.fill(testMessage);
    console.log(`输入测试消息: ${testMessage}`);

    // 确认输入框内容
    const inputValue = await messageInput.inputValue();
    console.log(`输入框当前值: ${inputValue}`);

    // 监听API请求
    let apiRequestDetected = false;
    let apiResponse = null;

    page.on('request', request => {
      if (request.url().includes('/api/query') || request.url().includes('/chat')) {
        console.log(`🔍 检测到API请求: ${request.method()} ${request.url()}`);
        apiRequestDetected = true;
      }
    });

    page.on('response', response => {
      if (response.url().includes('/api/query') || response.url().includes('/chat')) {
        console.log(`📡 收到API响应: ${response.status()} ${response.url()}`);
        apiResponse = response;
      }
    });

    // 点击发送按钮
    await sendButton.click();
    console.log('🖱️  点击发送按钮');

    // 等待可能的API请求和页面更新
    await page.waitForTimeout(3000);

    if (apiRequestDetected) {
      console.log('✅ 检测到API请求');
    } else {
      console.log('⚠️  未检测到API请求');
    }

    // 检查用户消息是否显示在界面上
    const userMessageSelectors = [
      `.message-user:has-text("${testMessage}")`,
      `.user-message:has-text("${testMessage}")`,
      `[data-role="user"]:has-text("${testMessage}")`,
      `text=${testMessage}`,
      `.ant-message:has-text("${testMessage}")`
    ];

    let userMessageFound = false;
    for (const selector of userMessageSelectors) {
      try {
        const element = page.locator(selector);
        if (await element.isVisible({ timeout: 1000 })) {
          console.log(`✅ 找到用户消息显示: ${selector}`);
          userMessageFound = true;
          break;
        }
      } catch (e) {
        // 继续查找
      }
    }

    if (!userMessageFound) {
      console.log('⚠️  未在界面上找到用户消息的显示');
    }

    // 等待AI回复
    console.log('等待AI回复...');
    await page.waitForTimeout(5000);

    // 查找AI回复消息
    const aiMessageSelectors = [
      '.message-assistant',
      '.ai-message', 
      '.bot-message',
      '[data-role="assistant"]',
      '.ant-message-content'
    ];

    let aiMessageFound = false;
    for (const selector of aiMessageSelectors) {
      try {
        const elements = page.locator(selector);
        const count = await elements.count();
        if (count > 0) {
          console.log(`✅ 找到AI回复消息 (${count}条): ${selector}`);
          const lastMessage = elements.last();
          const content = await lastMessage.textContent();
          console.log(`AI回复内容: ${content?.substring(0, 200)}...`);
          aiMessageFound = true;
          break;
        }
      } catch (e) {
        // 继续查找
      }
    }

    if (!aiMessageFound) {
      console.log('⚠️  未找到AI回复消息');
    }

    // 检查输入框是否已清空
    const finalInputValue = await messageInput.inputValue();
    console.log(`发送后输入框值: "${finalInputValue}"`);

    // 输出最终测试结果
    console.log('\n📊 测试结果总结:');
    console.log(`- API请求检测: ${apiRequestDetected ? '✅' : '❌'}`);
    console.log(`- 用户消息显示: ${userMessageFound ? '✅' : '❌'}`);
    console.log(`- AI回复检测: ${aiMessageFound ? '✅' : '❌'}`);
    console.log(`- 输入框清空: ${finalInputValue === '' ? '✅' : '❌'}`);

  } else {
    console.log('❌ 无法找到聊天界面元素');
    if (!messageInput) console.log('  - 未找到消息输入框');
    if (!sendButton) console.log('  - 未找到发送按钮');

    // 输出调试信息
    console.log('\n🔍 页面调试信息:');
    console.log('页面标题:', await page.title());
    console.log('页面URL:', page.url());
    
    // 列出页面上所有的输入元素
    const allInputs = page.locator('input, textarea');
    const inputCount = await allInputs.count();
    console.log(`页面上的输入元素数量: ${inputCount}`);
    
    for (let i = 0; i < Math.min(inputCount, 3); i++) {
      const input = allInputs.nth(i);
      const tagName = await input.evaluate(el => el.tagName);
      const placeholder = await input.getAttribute('placeholder');
      const className = await input.getAttribute('class');
      console.log(`${tagName} ${i + 1}: placeholder="${placeholder}", class="${className}"`);
    }
  }

  // 最终截图
  await page.screenshot({ path: 'bypass-auth-chat-test.png', fullPage: true });
  console.log('已保存测试截图: bypass-auth-chat-test.png');
});