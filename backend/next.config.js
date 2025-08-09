/** @type {import('next').NextConfig} */
const nextConfig = {
  env: {
    DOCUMENT_PROCESSOR_URL: process.env.DOCUMENT_PROCESSOR_URL,
    VECTOR_SERVICE_URL: process.env.VECTOR_SERVICE_URL,
    RAG_SERVICE_URL: process.env.RAG_SERVICE_URL,
  },
  reactStrictMode: true,
  swcMinify: true,
}

module.exports = nextConfig