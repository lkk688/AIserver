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
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
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
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
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
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  metadata: Record<string, any>;
}

export const fetchSources = async (): Promise<Source[]> => {
  const { data } = await api.get<Source[]>('/sources');
  return data;
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
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

// ─── Online Search Toolkit ──────────────────────────────────────────────────

export type OnlineSearchDomain = 'web' | 'news' | 'academic' | 'medical';

export interface OnlineSearchRecord {
  id: string;
  url: string;
  title: string;
  summary: string;
  content?: string | null;
  source: string;
  domain: string;
  category: string;
  language: string;
  published_at?: string | null;
  fetched_at?: string | null;
  is_breaking: boolean;
  record_type: string;
  query?: string | null;
  embedding?: null;
}

export interface OnlineSearchResult {
  query: string;
  domain: string;
  count: number;
  items: OnlineSearchRecord[];
  metadata: Record<string, unknown>;
}

export interface SchedulerJob {
  id: string;
  next_run: string | null;
  trigger: string;
}

export interface SchedulerConfig {
  breaking_interval_minutes: number;
  daily_refresh_hour_1: number;
  daily_refresh_hour_2: number;
  seed_crawl_hour: number;
  seed_urls: string[];
  news_refresh_queries: string[];
  academic_refresh_queries: string[];
  web_refresh_queries: string[];
}

export interface SchedulerStatus {
  running: boolean;
  jobs: SchedulerJob[];
  config: SchedulerConfig;
}

export interface OnlineSearchReq {
  query: string;
  limit?: number;
  language?: string;
  category?: string;
  no_cache?: boolean;
}

export const onlineSearch = async (
  domain: OnlineSearchDomain,
  req: OnlineSearchReq,
): Promise<OnlineSearchResult> => {
  const { data } = await api.post<OnlineSearchResult>(`/online-search/${domain}`, req);
  return data;
};

export const onlineReadUrl = async (
  url: string,
  domain = 'general',
  category = 'general',
  force_refresh = false,
): Promise<OnlineSearchRecord> => {
  const { data } = await api.post<OnlineSearchRecord>('/online-search/url', {
    url, domain, category, force_refresh,
  });
  return data;
};

export const onlineGetCache = async (params: {
  domain?: string;
  category?: string;
  limit?: number;
  recent_hours?: number;
} = {}): Promise<{ count: number; items: OnlineSearchRecord[] }> => {
  const { data } = await api.get('/online-search/cache', { params });
  return data;
};

export const onlineSchedulerStatus = async (): Promise<SchedulerStatus> => {
  const { data } = await api.get<SchedulerStatus>('/online-search/scheduler/status');
  return data;
};

export const onlineSchedulerStart = async (cfg?: Partial<SchedulerConfig>): Promise<{ status: string }> => {
  const { data } = await api.post('/online-search/scheduler/start', cfg ?? {});
  return data;
};

export const onlineSchedulerStop = async (): Promise<{ status: string }> => {
  const { data } = await api.post('/online-search/scheduler/stop');
  return data;
};

export const onlineSchedulerRunNow = async (): Promise<{ status: string }> => {
  const { data } = await api.post('/online-search/scheduler/run-now');
  return data;
};

export const onlineSchedulerUpdateConfig = async (
  cfg: Partial<SchedulerConfig>,
): Promise<{ status: string; config: SchedulerConfig }> => {
  const { data } = await api.put('/online-search/scheduler/config', cfg);
  return data;
};
