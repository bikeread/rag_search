import { test, expect } from '@playwright/test';

test('简单登录测试', async ({ page }) => {
  // 1. 访问登录页面
  await page.goto('http://localhost:3002/login');
  console.log('访问登录页面');

  // 2. 等待页面加载
  await page.waitForLoadState('networkidle');
  console.log('页面加载完成');

  // 3. 检查是否在登录页面
  console.log('当前URL:', page.url());

  // 4. 查找并填写表单
  const emailInput = page.locator('input[placeholder*="邮箱"]');
  const passwordInput = page.locator('input[placeholder*="密码"]');
  const loginButton = page.locator('button[type="submit"]');

  console.log('检查表单元素可见性');
  await expect(emailInput).toBeVisible({ timeout: 5000 });
  await expect(passwordInput).toBeVisible({ timeout: 5000 });
  await expect(loginButton).toBeVisible({ timeout: 5000 });

  // 5. 填写登录信息
  await emailInput.fill('test@example.com');
  await passwordInput.fill('testpassword123');
  console.log('填写登录信息完成');

  // 6. 监听网络请求
  const responsePromise = page.waitForResponse('**/api/auth/signin');
  
  // 7. 点击登录按钮
  await loginButton.click();
  console.log('点击登录按钮');

  // 8. 等待登录请求响应
  const response = await responsePromise;
  console.log('登录请求状态:', response.status());
  console.log('登录响应:', await response.text());

  // 9. 等待一段时间看页面变化
  await page.waitForTimeout(3000);
  console.log('登录后URL:', page.url());

  // 10. 检查页面内容
  const pageContent = await page.content();
  console.log('页面标题:', await page.title());
});