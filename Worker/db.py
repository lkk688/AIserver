import asyncpg
from typing import Optional
from settings import settings

class Database:
    _pool: Optional[asyncpg.Pool] = None

    @classmethod
    async def connect(cls):
        if cls._pool is None:
            # wait_for=60 is generous, but good for startup race conditions
            cls._pool = await asyncpg.create_pool(
                dsn=settings.database_url,
                min_size=1,
                max_size=10
            )

    @classmethod
    async def disconnect(cls):
        if cls._pool:
            await cls._pool.close()
            cls._pool = None

    @classmethod
    async def execute(cls, query: str, *args):
        if cls._pool is None:
            await cls.connect()
        async with cls._pool.acquire() as connection:
            return await connection.execute(query, *args)

    @classmethod
    async def fetch(cls, query: str, *args):
        if cls._pool is None:
            await cls.connect()
        async with cls._pool.acquire() as connection:
            return await connection.fetch(query, *args)

    @classmethod
    async def fetchrow(cls, query: str, *args):
        if cls._pool is None:
            await cls.connect()
        async with cls._pool.acquire() as connection:
            return await connection.fetchrow(query, *args)

db = Database()
