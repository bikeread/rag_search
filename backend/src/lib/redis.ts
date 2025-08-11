import Redis from 'ioredis'

const getRedisUrl = () => {
  if (process.env.REDIS_URL) {
    return process.env.REDIS_URL
  }
  return 'redis://localhost:6379'
}

export const redis = new Redis(getRedisUrl())

// 缓存工具函数
export class CacheService {
  private static DEFAULT_TTL = 3600 // 1小时

  static async get<T>(key: string): Promise<T | null> {
    try {
      const value = await redis.get(key)
      return value ? JSON.parse(value) : null
    } catch (error) {
      console.error('Cache get error:', error)
      return null
    }
  }

  static async set(key: string, value: any, ttl: number = CacheService.DEFAULT_TTL): Promise<void> {
    try {
      await redis.setex(key, ttl, JSON.stringify(value))
    } catch (error) {
      console.error('Cache set error:', error)
    }
  }

  static async del(key: string): Promise<void> {
    try {
      await redis.del(key)
    } catch (error) {
      console.error('Cache delete error:', error)
    }
  }

  static async exists(key: string): Promise<boolean> {
    try {
      const result = await redis.exists(key)
      return result === 1
    } catch (error) {
      console.error('Cache exists error:', error)
      return false
    }
  }

  static async keys(pattern: string): Promise<string[]> {
    try {
      return await redis.keys(pattern)
    } catch (error) {
      console.error('Cache keys error:', error)
      return []
    }
  }

  static async delByPattern(pattern: string): Promise<void> {
    try {
      const keys = await CacheService.keys(pattern)
      if (keys.length > 0) {
        await redis.del(...keys)
        console.log(`Cleared ${keys.length} cache keys matching pattern: ${pattern}`)
      }
    } catch (error) {
      console.error('Cache delete by pattern error:', error)
    }
  }
}