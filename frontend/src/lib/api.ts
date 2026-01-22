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
  status: string;
  created_at: string;
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

export const search = async (query: string, top_k: number = 10): Promise<SearchResult[]> => {
  const { data } = await api.post<SearchResult[]>('/search', { query, top_k });
  return data;
};
