#!/usr/bin/env python3
import asyncio
import copy
import json
import logging
import uuid
import os
import sys
from dotenv import load_dotenv

# Add backend directory to sys.path to allow imports from app
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import websockets

from doubao_protocol_v3 import (
    EventType,
    MsgType,
    finish_connection,
    finish_session,
    receive_message,
    start_connection,
    start_session,
    task_request,
    wait_for_event,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_resource_id(voice: str) -> str:
    if voice.startswith("S_"):
        return "volc.megatts.default"
    return "volc.service_type.10029"


async def main():
    appid = os.getenv("VOLC_APPID")
    access_token = os.getenv("VOLC_ACCESS_TOKEN")
    cluster_id = os.getenv("VOLC_CLUSTER_ID", "volc.service_type.10029")
    text = "你好，这是一个豆包语音合成的测试。我在做第二次测试。"
    # voice_type = settings.tts_default_voice or "zh_female_shuangkuaishishang_moon_bigtts" # Use a default if not set
    # Try a voice that might work with seed-tts-2.0
    # voice_type = "zh_female_emotion"
    # voice_type = "S_v2_zh_female_emotion"
    voice_type = "zh_female_vv_uranus_bigtts"
    # voice_type = "BV700_streaming"

    encoding = "mp3"
    endpoint = "wss://openspeech.bytedance.com/api/v3/tts/bidirection"

    if not appid or not access_token:
        logger.error("Missing VOLC_APPID or VOLC_ACCESS_TOKEN in settings")
        return

    logger.info(f"AppID: {appid}, ClusterID: {cluster_id}")
    
    # Check if voice is standard and cluster is mismatch
    # BV700_streaming is a standard voice, usually requires volc.service_type.10029
    # if voice_type == "BV700_streaming" and cluster_id != "volc.service_type.10029":
    #     logger.warning(f"Voice {voice_type} might not work with cluster {cluster_id}. Trying volc.service_type.10029...")
    #     # cluster_id = "volc.service_type.10029" 
    #     # Uncomment above line to force fix, but let's see if we can use the helper logic
        
    # Helper logic from sample
    # calculated_resource_id = get_resource_id(voice_type)
    # logger.info(f"Calculated Resource ID for {voice_type}: {calculated_resource_id}")
    
    # Use calculated one if the env one fails? 
    # Let's try to use the one from env first, but if it fails, maybe we should have used the other.
    # The error "resource ID is mismatched" confirms we sent the WRONG resource ID for the voice.
    # So we SHOULD match them.
    
    # if cluster_id != calculated_resource_id:
    #      logger.warning(f"ENV ClusterID {cluster_id} differs from suggested {calculated_resource_id} for voice {voice_type}")
    #      # For this test, let's use the calculated one to verify the protocol works
    #      # cluster_id = calculated_resource_id
    
    # Force seed-tts-2.0 for this test
    cluster_id = "seed-tts-2.0"
    # cluster_id = "volc.service_type.10029"

    # Connect to server
    headers = {
        "X-Api-App-Key": appid,
        "X-Api-Access-Key": access_token,
        "X-Api-Resource-Id": cluster_id,
        "X-Api-Connect-Id": str(uuid.uuid4()),
    }

    logger.info(f"Connecting to {endpoint} with headers: {headers}")
    async with websockets.connect(
        endpoint, additional_headers=headers, max_size=10 * 1024 * 1024
    ) as websocket:
        
        logger.info(f"Connected to WebSocket server")
        # logger.info(f"Connected to WebSocket server, Logid: {websocket.response_headers.get('x-tt-logid')}")

        try:
            # Start connection
            # await start_connection(websocket, connect_id=headers["X-Api-Connect-Id"])
            await start_connection(websocket)
            await wait_for_event(
                websocket, MsgType.FullServerResponse, EventType.ConnectionStarted
            )
            logger.info("Connection Started!")

            # Process sentence
            # every session can have different parameters
            base_request = {
                "user": {
                    "uid": str(uuid.uuid4()),
                },
                "namespace": "BidirectionalTTS",
                "req_params": {
                    "speaker": voice_type,
                    "audio_params": {
                        "format": encoding,
                        "sample_rate": 24000,
                        "enable_timestamp": True,
                    },
                    "additions": json.dumps(
                        {
                            "disable_markdown_filter": False,
                        }
                    ),
                },
            }

            # Start session
            start_session_request = copy.deepcopy(base_request)
            start_session_request["event"] = EventType.StartSession
            session_id = str(uuid.uuid4())
            await start_session(
                websocket, json.dumps(start_session_request).encode(), session_id
            )
            await wait_for_event(
                websocket, MsgType.FullServerResponse, EventType.SessionStarted
            )
            logger.info("Session Started!")

            # Send full text at once (or characters)
            # The sample sends characters one by one. Let's send full text for simplicity?
            # Or stick to sample: send chars.
            # "Send characters one by one"
            
            async def send_chars():
                for char in text:
                    synthesis_request = copy.deepcopy(base_request)
                    synthesis_request["event"] = EventType.TaskRequest
                    synthesis_request["req_params"]["text"] = char
                    await task_request(
                        websocket, json.dumps(synthesis_request).encode(), session_id
                    )
                    # await asyncio.sleep(0.005) 
                
                # Send Finish Session after text is done
                # Note: The sample sends FinishSession after sending all text.
                await finish_session(websocket, session_id)
                logger.info("Sent FinishSession")

            # Start sending characters in background
            send_task = asyncio.create_task(send_chars())

            # Receive audio data
            audio_data = bytearray()
            audio_received = False
            
            while True:
                msg = await receive_message(websocket)

                if msg.type == MsgType.FullServerResponse:
                    if msg.event == EventType.SessionFinished:
                        logger.info("Session Finished Event Received")
                        break
                    if msg.event == EventType.TaskFailed:
                         logger.error(f"Task Failed: {msg}")
                         break
                elif msg.type == MsgType.AudioOnlyServer:
                    if not audio_received and len(msg.payload) > 0:
                        audio_received = True
                    audio_data.extend(msg.payload)
                elif msg.type == MsgType.Error:
                    logger.error(f"Error Message: {msg}")
                    break
                else:
                    # logger.warning(f"Unexpected message type: {msg}")
                    pass

            # Wait for send_chars to complete
            await send_task

            # Save audio file if we received any data
            if audio_data:
                filename = "test_doubao_official.mp3"
                with open(filename, "wb") as f:
                    f.write(audio_data)
                logger.info(f"Audio received: {len(audio_data)}, saved to {filename}")
            else:
                logger.warning("No audio data received")

        finally:
            # Finish connection
            await finish_connection(websocket)
            # msg = await wait_for_event(
            #     websocket, MsgType.FullServerResponse, EventType.ConnectionFinished
            # )
            # logger.info("Connection Finished")


if __name__ == "__main__":
    asyncio.run(main())
