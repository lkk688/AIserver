import axios from 'axios';

// We use relative URL which will be proxied by Next.js to backend
const api = axios.create({
  baseURL: '/api/v1',
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface Source {
  id: string;
  name: string;
  path: string;
  config: Record<string, any>;
  created_at: string;
  updated_at: string;
}

export interface Job {
  id: string;
  type: string;
  status: 'pending' | 'running' | 'done' | 'failed';
  progress: number;
  error?: string;
  payload: Record<string, any>;
  created_at: string;
  updated_at: string;
}

export interface Document {
  id: string;
  source_id: string;
  uri: string;
  title?: string;
  mime_type?: string;
  size_bytes?: number;
  status: string;
  created_at: string;
  updated_at: string;
  mtime?: string;
}

export interface Chunk {
  id: string;
  doc_id: string;
  chunk_index: number;
  text: string;
  start_offset: number;
  end_offset: number;
  section_id?: string | null;
}

export interface Section {
  id: string;
  doc_id: string;
  title: string;
  level: number;
  page_start?: number | null;
  page_end?: number | null;
  parent_section_id?: string | null;
}

export interface SearchResult {
  doc_id: string;
  chunk_id: string;
  score: number;
  text: string;
  metadata: Record<string, any>;
}

export const fetchSources = async (): Promise<Source[]> => {
  const { data } = await api.get<Source[]>('/sources');
  return data;
};

export const createSource = async (source: { name: string; path: string; config?: Record<string, any> }): Promise<Source> => {
  const { data } = await api.post<Source>('/sources', source);
  return data;
};

export const scanSource = async (sourceId: string): Promise<Job> => {
  const { data } = await api.post<Job>(`/sources/${sourceId}/scan`);
  return data;
};

export const fetchJobs = async (): Promise<Job[]> => {
  const { data } = await api.get<Job[]>('/jobs');
  return data;
};

export const fetchDocuments = async (sourceId?: string): Promise<Document[]> => {
  const params = sourceId ? { source_id: sourceId } : {};
  const { data } = await api.get<Document[]>('/documents', { params });
  return data;
};

export const fetchDocumentChunks = async (docId: string): Promise<Chunk[]> => {
  const { data } = await api.get<Chunk[]>(`/documents/${docId}/chunks`);
  return data;
};

export const fetchDocumentSections = async (docId: string): Promise<Section[]> => {
  const { data } = await api.get<Section[]>(`/documents/${docId}/sections`);
  return data;
};

export const fetchDocument = async (id: string): Promise<Document> => {
  const { data } = await api.get<Document>(`/documents/${id}`);
  return data;
};

export const deleteDocument = async (id: string): Promise<void> => {
  await api.delete(`/documents/${id}`);
};

export const uploadDocument = async (file: File): Promise<Document> => {
  const formData = new FormData();
  formData.append('file', file);
  const { data } = await api.post<Document>('/documents/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
};

export const search = async (query: string, top_k: number = 10): Promise<SearchResult[]> => {
  const { data } = await api.post<SearchResult[]>('/search', { query, top_k });
  return data;
};
