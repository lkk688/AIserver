from online_search_toolkit import create_online_search_service, SearchConfig

config = SearchConfig()
config.embedding.api_url = "http://localhost:8081/v1"
config.embedding.api_key = "EMPTY"
config.embedding.api_model = "text-embeddings-inference"
config.store.postgres_enabled = False

service = create_online_search_service(config)

r1 = service.search_web("vllm paged attention", limit=5)
print(r1.model_dump())

r2 = service.search_news("OpenAI", limit=5, language="mixed")
print(r2.model_dump())

r3 = service.search_academic("large language models", limit=5)
print(r3.model_dump())

r4 = service.read_url("https://example.com")
print(r4.model_dump())