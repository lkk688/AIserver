import json
import logging
from typing import Dict, Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/meeting", tags=["Meeting"])

class ConnectionManager:
    def __init__(self):
        # room_id -> set of active WebSockets
        self.active_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, room_id: str):
        await websocket.accept()
        if room_id not in self.active_connections:
            self.active_connections[room_id] = set()
        self.active_connections[room_id].add(websocket)
        logger.info(f"Client joined room {room_id}. Total: {len(self.active_connections[room_id])}")

    def disconnect(self, websocket: WebSocket, room_id: str):
        if room_id in self.active_connections:
            self.active_connections[room_id].remove(websocket)
            logger.info(f"Client left room {room_id}. Total: {len(self.active_connections[room_id])}")
            if not self.active_connections[room_id]:
                del self.active_connections[room_id]

    async def broadcast(self, message: str, room_id: str, sender: WebSocket):
        if room_id in self.active_connections:
            for connection in self.active_connections[room_id]:
                if connection != sender:
                    try:
                        await connection.send_text(message)
                    except Exception as e:
                        logger.error(f"Error sending message to client: {e}")

manager = ConnectionManager()

@router.websocket("/ws/{room_id}")
async def meeting_signaling(websocket: WebSocket, room_id: str):
    """
    WebSocket endpoint for WebRTC signaling.
    Clients connect to a specific room_id and exchange SDP offers, answers, and ICE candidates.
    """
    await manager.connect(websocket, room_id)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.broadcast(data, room_id, websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket, room_id)
        # Notify others that a peer disconnected
        leave_msg = json.dumps({"type": "peer_left"})
        await manager.broadcast(leave_msg, room_id, websocket)
