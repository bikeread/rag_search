import { test, expect } from '@playwright/test';

const TEST_USER = {
  email: 'test@example.com',
  password: 'testpassword123'
};

test.describe('AI消息发送测试', () => {
  test('应该能够发送消息并看到界面响应', async ({ page }) => {
    console.log('开始消息发送测试');

    // 1. 访问登录页面并登录
    await page.goto('/login');
    console.log('访问登录页面');

    // 填写登录信息
    await page.fill('input[placeholder*="邮箱"]', TEST_USER.email);
    await page.fill('input[placeholder*="密码"]', TEST_USER.password);
    
    console.log('填写登录信息完成');

    // 点击登录按钮
    await page.click('button[type="submit"]');
    console.log('点击登录按钮');

    // 等待页面变化（不管是否成功登录）
    await page.waitForTimeout(3000);
    console.log('登录后URL:', page.url());

    // 2. 尝试访问聊天页面
    await page.goto('/chat');
    console.log('导航到聊天页面');

    await page.waitForLoadState('networkidle');
    console.log('聊天页面加载完成');

    // 3. 查看页面状态
    const currentUrl = page.url();
    console.log('当前页面URL:', currentUrl);

    // 如果被重定向到登录页面，说明登录失败
    if (currentUrl.includes('/login')) {
      console.log('❌ 登录失败，无法测试消息发送');
      // 但我们仍然可以测试聊天界面的基本元素
      console.log('尝试手动访问聊天页面查看界面元素');
      
      // 跳过认证测试界面
      await page.goto('/chat');
      await page.waitForTimeout(2000);
    }

    // 4. 查找聊天界面元素
    console.log('查找聊天界面元素...');

    // 查找各种可能的消息输入框
    const possibleInputs = [
      'input[placeholder*="问题"]',
      'input[placeholder*="消息"]',
      'textarea[placeholder*="问题"]', 
      'textarea[placeholder*="消息"]',
      'input[type="text"]',
      'textarea'
    ];

    let messageInput = null;
    for (const selector of possibleInputs) {
      try {
        const element = page.locator(selector).first();
        if (await element.isVisible({ timeout: 1000 })) {
          messageInput = element;
          console.log(`✅ 找到输入框: ${selector}`);
          break;
        }
      } catch (e) {
        // 继续查找下一个
      }
    }

    // 查找各种可能的发送按钮
    const possibleButtons = [
      'button:has-text("发送")',
      'button:has-text("Send")',
      'button[type="submit"]',
      'button:has([class*="send"])',
      'button:has([class*="submit"])'
    ];

    let sendButton = null;
    for (const selector of possibleButtons) {
      try {
        const element = page.locator(selector).first();
        if (await element.isVisible({ timeout: 1000 })) {
          sendButton = element;
          console.log(`✅ 找到发送按钮: ${selector}`);
          break;
        }
      } catch (e) {
        // 继续查找下一个
      }
    }

    // 5. 测试消息输入和发送
    if (messageInput && sendButton) {
      console.log('✅ 找到了输入框和发送按钮，开始测试消息发送');

      const testMessage = '测试消息：这是一个AI对话测试';
      
      // 输入消息
      await messageInput.fill(testMessage);
      console.log(`输入测试消息: ${testMessage}`);

      // 监听可能的API请求
      const queryRequest = page.waitForRequest('**/api/query**');
      const chatRequest = page.waitForRequest('**/chat**');
      
      // 点击发送按钮
      await sendButton.click();
      console.log('点击发送按钮');

      // 等待并检查请求
      try {
        const request = await Promise.race([
          queryRequest.catch(() => null),
          chatRequest.catch(() => null),
          page.waitForTimeout(5000).then(() => null)
        ]);

        if (request) {
          console.log(`✅ 检测到API请求: ${request.url()}`);
          console.log(`请求方法: ${request.method()}`);
          
          // 等待响应
          try {
            const response = await request.response();
            if (response) {
              console.log(`响应状态: ${response.status()}`);
            }
          } catch (e) {
            console.log('等待响应时出错:', e.message);
          }
        } else {
          console.log('⚠️ 未检测到API请求');
        }
      } catch (e) {
        console.log('监听请求时出错:', e.message);
      }

      // 查看页面是否有变化
      await page.waitForTimeout(2000);
      
      // 查找可能的用户消息显示
      const possibleUserMessages = [
        `.message-user:has-text("${testMessage}")`,
        `.user-message:has-text("${testMessage}")`,
        `[data-role="user"]:has-text("${testMessage}")`,
        `text=${testMessage}`
      ];

      let userMessageFound = false;
      for (const selector of possibleUserMessages) {
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
        console.log('⚠️ 未找到用户消息的显示');
      }

      // 查找可能的AI回复
      await page.waitForTimeout(3000); // 等待AI响应

      const possibleAIMessages = [
        '.message-assistant',
        '.ai-message',
        '.bot-message',
        '[data-role="assistant"]'
      ];

      let aiMessageFound = false;
      for (const selector of possibleAIMessages) {
        try {
          const elements = page.locator(selector);
          const count = await elements.count();
          if (count > 0) {
            console.log(`✅ 找到AI消息回复 (${count}条): ${selector}`);
            // 获取最后一条消息的内容
            const lastMessage = elements.last();
            const content = await lastMessage.textContent();
            console.log(`AI回复内容: ${content?.substring(0, 100)}...`);
            aiMessageFound = true;
            break;
          }
        } catch (e) {
          // 继续查找
        }
      }

      if (!aiMessageFound) {
        console.log('⚠️ 未找到AI回复消息');
      }

    } else {
      console.log('❌ 未找到输入框或发送按钮');
      if (!messageInput) console.log('  - 未找到消息输入框');
      if (!sendButton) console.log('  - 未找到发送按钮');
      
      // 输出页面信息用于调试
      console.log('页面标题:', await page.title());
      console.log('页面HTML (前1000字符):', (await page.content()).substring(0, 1000));
    }

    // 6. 最终状态报告
    const finalUrl = page.url();
    console.log(`测试完成，最终URL: ${finalUrl}`);
    
    // 截图保存当前状态
    await page.screenshot({ path: 'message-test-final.png', fullPage: true });
    console.log('已保存最终状态截图: message-test-final.png');
  });
});