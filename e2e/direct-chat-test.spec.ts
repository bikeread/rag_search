import { test, expect } from '@playwright/test';

test('直接访问聊天页面测试', async ({ page }) => {
  console.log('直接访问聊天页面，检查界面和功能');

  // 访问聊天页面
  await page.goto('/chat');
  console.log('访问聊天页面');

  await page.waitForLoadState('networkidle');
  console.log('页面加载完成');

  console.log('当前URL:', page.url());
  console.log('页面标题:', await page.title());

  // 等待一段时间让页面完全渲染
  await page.waitForTimeout(2000);

  // 检查页面内容
  const pageContent = await page.content();
  console.log('页面是否包含聊天相关内容:');
  console.log('- 包含"聊天":', pageContent.includes('聊天'));
  console.log('- 包含"chat":', pageContent.includes('chat'));
  console.log('- 包含"AI":', pageContent.includes('AI'));
  console.log('- 包含"问答":', pageContent.includes('问答'));
  console.log('- 包含"消息":', pageContent.includes('消息'));

  // 查找所有input元素
  const inputs = page.locator('input');
  const inputCount = await inputs.count();
  console.log(`页面上的input元素数量: ${inputCount}`);

  for (let i = 0; i < Math.min(inputCount, 5); i++) {
    const input = inputs.nth(i);
    const placeholder = await input.getAttribute('placeholder');
    const type = await input.getAttribute('type');
    console.log(`Input ${i + 1}: type="${type}", placeholder="${placeholder}"`);
  }

  // 查找所有button元素
  const buttons = page.locator('button');
  const buttonCount = await buttons.count();
  console.log(`页面上的button元素数量: ${buttonCount}`);

  for (let i = 0; i < Math.min(buttonCount, 5); i++) {
    const button = buttons.nth(i);
    const text = await button.textContent();
    const type = await button.getAttribute('type');
    console.log(`Button ${i + 1}: type="${type}", text="${text?.trim()}"`);
  }

  // 查找可能的聊天容器
  const chatContainers = [
    '.chat-container',
    '.chat-interface', 
    '.messages-container',
    '.message-list',
    '.conversation',
    '[class*="chat"]',
    '[class*="message"]'
  ];

  for (const selector of chatContainers) {
    try {
      const element = page.locator(selector);
      const count = await element.count();
      if (count > 0) {
        console.log(`✅ 找到聊天容器: ${selector} (${count}个)`);
      }
    } catch (e) {
      // 继续查找
    }
  }

  // 截图保存当前状态
  await page.screenshot({ path: 'direct-chat-test.png', fullPage: true });
  console.log('已保存页面截图: direct-chat-test.png');

  // 尝试获取页面上的所有文本内容
  const allText = await page.locator('body').textContent();
  console.log('页面主要文本内容:');
  console.log(allText?.substring(0, 500) + '...');
});