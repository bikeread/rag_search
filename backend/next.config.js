/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  env: {
    DATABASE_URL: process.env.DATABASE_URL,
    REDIS_URL: process.env.REDIS_URL,
    RABBITMQ_URL: process.env.RABBITMQ_URL,
    DOCUMENT_PROCESSOR_URL: process.env.DOCUMENT_PROCESSOR_URL,
    VECTOR_SERVICE_URL: process.env.VECTOR_SERVICE_URL,
    RAG_SERVICE_URL: process.env.RAG_SERVICE_URL,
  },
};

module.exports = nextConfig;