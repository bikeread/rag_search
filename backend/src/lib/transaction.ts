import { Prisma } from '@prisma/client'
import { prisma } from './prisma'

export const withTransaction = async <T>(
  operation: (tx: Prisma.TransactionClient) => Promise<T>
): Promise<T> => {
  return prisma.$transaction(async (tx) => {
    try {
      return await operation(tx)
    } catch (error) {
      console.error('Transaction failed:', error)
      throw new Error(`Transaction failed: ${error instanceof Error ? error.message : 'Unknown error'}`)
    }
  }, {
    maxWait: 5000,
    timeout: 30000,
    isolationLevel: 'ReadCommitted',
  })
}