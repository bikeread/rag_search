import { NextApiRequest, NextApiResponse } from 'next';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  res.status(200).json({
    message: 'RAG System Backend API',
    version: '1.0.0',
    services: {
      health: '/api/health',
      upload: '/api/documents/upload',
      query: '/api/query'
    },
    timestamp: new Date().toISOString()
  });
}