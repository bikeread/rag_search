import { z } from 'zod'

export const uploadSchema = z.object({
  filename: z.string().min(1).max(255),
  size: z.number().min(1).max(10 * 1024 * 1024), // 10MB
  mimeType: z.string(),
})

export const querySchema = z.object({
  query: z.string().min(1).max(1000),
  topK: z.number().min(1).max(20).optional(),
  useCache: z.boolean().optional(),
})

export const chatSchema = z.object({
  messages: z.array(z.object({
    role: z.enum(['user', 'assistant']),
    content: z.string().min(1),
  })).min(1).max(50),
})

export const registerSchema = z.object({
  email: z.string().email(),
  password: z.string().min(8).max(128),
  name: z.string().min(1).max(100).optional(),
})