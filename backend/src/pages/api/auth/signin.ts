import { NextApiRequest, NextApiResponse } from 'next'
import bcrypt from 'bcryptjs'
import jwt from 'jsonwebtoken'
import { prisma } from '@/lib/prisma'
import { z } from 'zod'
import { getCorsHeaders } from '@/lib/cors'

const signinSchema = z.object({
  email: z.string().email(),
  password: z.string().min(1),
})

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  const origin = req.headers.origin
  
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    const corsHeaders = getCorsHeaders(origin)
    Object.entries(corsHeaders).forEach(([key, value]) => {
      res.setHeader(key, value)
    })
    res.status(200).end()
    return
  }
  
  // Add CORS headers to all other requests
  const corsHeaders = getCorsHeaders(origin)
  Object.entries(corsHeaders).forEach(([key, value]) => {
    res.setHeader(key, value)
  })

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  try {
    const validation = signinSchema.safeParse(req.body)
    
    if (!validation.success) {
      return res.status(400).json({
        error: 'Invalid data',
        details: validation.error.issues
      })
    }

    const { email, password } = validation.data

    // Find user by email
    const user = await prisma.user.findUnique({
      where: { email }
    })

    if (!user || !user.password) {
      return res.status(401).json({ error: 'Invalid credentials' })
    }

    // Verify password
    const isPasswordValid = await bcrypt.compare(password, user.password)

    if (!isPasswordValid) {
      return res.status(401).json({ error: 'Invalid credentials' })
    }

    // Generate JWT token
    const token = jwt.sign(
      { 
        userId: user.id, 
        email: user.email,
        role: user.role 
      },
      process.env.JWT_SECRET || 'fallback-secret-key',
      { expiresIn: '7d' }
    )

    res.status(200).json({
      user: {
        id: user.id,
        email: user.email,
        name: user.name,
        role: user.role,
      },
      token
    })

  } catch (error) {
    console.error('Signin error:', error)
    res.status(500).json({ error: 'Signin failed' })
  }
}