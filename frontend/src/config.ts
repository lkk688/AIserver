export interface AgentModelConfig {
  id: string;
  name: string;
  provider: 'openai' | 'anthropic';
  baseUrl: string;
  apiKey: string;
  model: string;
  backend: 'vllm' | 'openai' | 'anthropic' | 'llama.cpp';
}

export interface AgentConfig {
  /** Base URL for the agent REST API */
  apiUrl: string;
  /** Default model id (must match one of the models[] entries) */
  defaultModel: string;
  /** Max context window sent to the backend (0 = auto-detect) */
  maxContext: number;
  /** Max output tokens (0 = auto-detect) */
  maxOutput: number;
  /** Available LLM backends the user can switch between */
  models: AgentModelConfig[];
}

export interface EmbeddingServiceConfig {
  baseUrl: string;
  apiKey: string;
  model: string;
}

export interface RerankerServiceConfig {
  baseUrl: string;
  apiKey: string;
  model: string;
}

export interface MinioConfig {
  endpoint: string;
  accessKey: string;
  secretKey: string;
  bucket: string;
}

export const config = {
  tts: {
    backend: 'auto' as 'native' | 'webgpu' | 'doubao' | 'auto',
    webgpu: {
      zh: {
        model: 'sherpa',
        mode: 'streaming',
      },
      en: {
        model: 'kokoro',
        mode: 'streaming',
      },
    },
    doubao: {
      voice: 'zh_female_vv_uranus_bigtts',
    },
  },

  agent: {
    apiUrl: process.env.NEXT_PUBLIC_AGENT_API_URL || '/api/v1',
    defaultModel: 'local-qwen',
    maxContext: 0,
    maxOutput: 0,
    models: [
      {
        id: 'local-qwen',
        name: 'Qwen (Local)',
        provider: 'openai',
        baseUrl: process.env.NEXT_PUBLIC_VLLM_BASE_URL || 'http://100.110.236.127:8000/v1',
        apiKey: process.env.NEXT_PUBLIC_VLLM_API_KEY || 'EMPTY',
        model: process.env.NEXT_PUBLIC_VLLM_MODEL || 'qwen3-30b-a3b',
        backend: 'vllm',
      },
      {
        id: 'claude-sonnet',
        name: 'Claude Sonnet 4.6',
        provider: 'anthropic',
        baseUrl: '',
        apiKey: process.env.NEXT_PUBLIC_ANTHROPIC_API_KEY || '',
        model: 'claude-sonnet-4-6',
        backend: 'anthropic',
      },
      {
        id: 'gpt-4o',
        name: 'GPT-4o',
        provider: 'openai',
        baseUrl: 'https://api.openai.com/v1',
        apiKey: process.env.NEXT_PUBLIC_OPENAI_API_KEY || '',
        model: 'gpt-4o',
        backend: 'openai',
      },
    ],
  } satisfies AgentConfig,

  embedding: {
    baseUrl: process.env.NEXT_PUBLIC_EMBEDDING_BASE_URL || 'http://100.83.246.7:8002/v1',
    apiKey: process.env.NEXT_PUBLIC_EMBEDDING_API_KEY || 'embeddingp100',
    model: process.env.NEXT_PUBLIC_EMBEDDING_MODEL || 'Qwen/Qwen3-Embedding-0.6B',
  } satisfies EmbeddingServiceConfig,

  reranker: {
    baseUrl: process.env.NEXT_PUBLIC_RERANKER_BASE_URL || 'http://100.83.246.7:8002/v1',
    apiKey: process.env.NEXT_PUBLIC_RERANKER_API_KEY || 'embeddingp100',
    model: process.env.NEXT_PUBLIC_RERANKER_MODEL || 'Qwen/Qwen3-Reranker-0.6B',
  } satisfies RerankerServiceConfig,

  minio: {
    endpoint: process.env.NEXT_PUBLIC_MINIO_ENDPOINT || '100.83.246.7:9000',
    accessKey: process.env.NEXT_PUBLIC_MINIO_ACCESS_KEY || 'my_minio_admin',
    secretKey: process.env.NEXT_PUBLIC_MINIO_SECRET_KEY || 'my_minio_secret',
    bucket: process.env.NEXT_PUBLIC_MINIO_BUCKET || 'aiagent',
  } satisfies MinioConfig,
};
