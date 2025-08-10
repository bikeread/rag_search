import NextAuth from "next-auth"
import { authOptions } from "@/lib/auth"
import { getCorsHeaders } from "@/lib/cors"
import { NextApiRequest, NextApiResponse } from "next"

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  const origin = req.headers.origin
  
  // 处理OPTIONS预检请求
  if (req.method === 'OPTIONS') {
    const corsHeaders = getCorsHeaders(origin)
    Object.entries(corsHeaders).forEach(([key, value]) => {
      res.setHeader(key, value)
    })
    res.status(200).end()
    return
  }
  
  // 为所有其他请求添加CORS头
  const corsHeaders = getCorsHeaders(origin)
  Object.entries(corsHeaders).forEach(([key, value]) => {
    res.setHeader(key, value)
  })
  
  return NextAuth(req, res, authOptions)
}