import redis.asyncio as redis
import socketio
from settings import settings
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class Communication:
    _redis: Optional[redis.Redis] = None
    _sio_mgr: Optional[socketio.AsyncRedisManager] = None

    @classmethod
    async def connect_redis(cls):
        if cls._redis is None:
            cls._redis = redis.from_url(settings.redis_url, decode_responses=False) # Keep binary for flexibility, decode when needed
    
    @classmethod
    async def close_redis(cls):
        if cls._redis:
            await cls._redis.close()
            cls._redis = None

    @classmethod
    async def get_redis(cls) -> redis.Redis:
        if cls._redis is None:
            await cls.connect_redis()
        return cls._redis

    @classmethod
    def get_socketio_manager(cls) -> socketio.AsyncRedisManager:
        if cls._sio_mgr is None:
            # We use the same redis URL for socketio manager
            # Note: AsyncRedisManager handles its own connections
            cls._sio_mgr = socketio.AsyncRedisManager(settings.redis_url, write_only=True)
        return cls._sio_mgr
    
    @classmethod
    async def emit_event(cls, channel: str, event: str, data: dict):
        """
        Emits an event to a specific room/channel via Redis to be picked up by SocketIO servers.
        """
        mgr = cls.get_socketio_manager()
        await mgr.emit(event, data, room=channel)
        logger.info(f"Emitted event {event} to channel {channel}")

comm = Communication()
