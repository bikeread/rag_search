import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import { NextApiRequest, NextApiResponse } from 'next';

// Helmet安全头配置
export const helmetConfig = helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      styleSrc: ["'self'", "'unsafe-inline'"],
      scriptSrc: ["'self'", "'unsafe-inline'", "'unsafe-eval'"],
      imgSrc: ["'self'", "data:", "https:"],
      connectSrc: ["'self'"],
      fontSrc: ["'self'"],
      objectSrc: ["'none'"],
      mediaSrc: ["'self'"],
      frameSrc: ["'none'"],
    },
  },
  crossOriginEmbedderPolicy: false,
});

// API限流配置 - 每个IP每分钟最多100个请求
export const apiLimiter = rateLimit({
  windowMs: 60 * 1000, // 1分钟
  max: 100, // 限制每个IP 100个请求
  message: 'Too many requests from this IP, please try again later.',
  standardHeaders: true, // 返回 `RateLimit-*` 头部
  legacyHeaders: false, // 禁用 `X-RateLimit-*` 头部
  handler: (req: any, res: any) => {
    res.status(429).json({
      error: 'Too many requests',
      message: 'API rate limit exceeded. Please try again later.',
      retryAfter: req.rateLimit?.resetTime,
    });
  },
});

// 严格限流 - 用于敏感操作如登录、注册
export const strictLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15分钟
  max: 5, // 限制每个IP 5个请求
  message: 'Too many attempts from this IP, please try again after 15 minutes.',
  standardHeaders: true,
  legacyHeaders: false,
  skipSuccessfulRequests: true, // 成功的请求不计入限制
});

// 文档上传限流 - 更宽松的限制
export const uploadLimiter = rateLimit({
  windowMs: 60 * 1000, // 1分钟
  max: 10, // 限制每个IP 10个上传请求
  message: 'Too many upload requests, please try again later.',
});

// 安全中间件包装器 - 用于Next.js API路由
export function withSecurity(handler: any) {
  return async (req: NextApiRequest, res: NextApiResponse) => {
    // 应用Helmet安全头
    const helmetMiddleware = helmet();
    await new Promise((resolve) => {
      helmetMiddleware(req as any, res as any, resolve);
    });

    // 应用速率限制
    await new Promise((resolve, reject) => {
      apiLimiter(req as any, res as any, (err: any) => {
        if (err) reject(err);
        else resolve(undefined);
      });
    });

    // 调用原始处理器
    return handler(req, res);
  };
}