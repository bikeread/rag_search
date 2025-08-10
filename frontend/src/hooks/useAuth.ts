import { useAuthStore } from '@/store/authStore'
import { useMutation } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { toast } from 'react-hot-toast'

export const useAuth = () => {
  const navigate = useNavigate()
  const { user, isAuthenticated, login, register, logout } = useAuthStore()

  const loginMutation = useMutation({
    mutationFn: ({ email, password }: { email: string; password: string }) =>
      login(email, password),
    onSuccess: () => {
      toast.success('登录成功！')
      navigate('/dashboard')
    },
    onError: (error: any) => {
      toast.error(error.message || '登录失败')
    },
  })

  const registerMutation = useMutation({
    mutationFn: register,
    onSuccess: () => {
      toast.success('注册成功！请登录')
      navigate('/login')
    },
    onError: (error: any) => {
      toast.error(error.message || '注册失败')
    },
  })

  const logoutHandler = () => {
    logout()
    toast.success('已退出登录')
    navigate('/login')
  }

  return {
    user,
    isAuthenticated,
    login: loginMutation,
    register: registerMutation,
    logout: logoutHandler,
  }
}