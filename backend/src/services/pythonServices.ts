import axios from 'axios';

const DOCUMENT_PROCESSOR_URL = process.env.DOCUMENT_PROCESSOR_URL || 'http://localhost:8001';
const RAG_SERVICE_URL = process.env.RAG_SERVICE_URL || 'http://localhost:8003';
const VECTOR_SERVICE_URL = process.env.VECTOR_SERVICE_URL || 'http://localhost:8002';

export async function uploadToDocumentProcessor(file: any, documentId: string) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('document_id', documentId);

  const response = await axios.post(
    `${DOCUMENT_PROCESSOR_URL}/process-document`,
    formData,
    {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 30000
    }
  );

  return response.data;
}

export async function queryRAGService(query: string) {
  const response = await axios.post(
    `${RAG_SERVICE_URL}/query`,
    { query },
    { timeout: 15000 }
  );

  return response.data;
}

export async function vectorizeTexts(texts: string[]) {
  const response = await axios.post(
    `${VECTOR_SERVICE_URL}/vectorize`,
    { texts },
    { timeout: 30000 }
  );

  return response.data;
}