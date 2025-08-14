import type { LoginRequest, RegisterRequest, User } from '@/types'
import { apiClient } from './api'

export const authService = {
  async login(data: LoginRequest): Promise<{ user: User; token?: string }> {
    return apiClient.post('/api/auth/signin', data)
  },

  async register(data: RegisterRequest): Promise<{ user: User }> {
    return apiClient.post('/api/auth/register', data)
  },

  async getCurrentUser(): Promise<User> {
    return apiClient.get('/api/auth/user')
  },

  async logout(): Promise<void> {
    return apiClient.post('/api/auth/signout')
  },
}