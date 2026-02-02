import asyncio
import websockets


async def stream_tts(
    text: str = "Hello, this is a streaming test for VibeVoice.",
    voice: str | None = None,
    cfg_scale: float = 1.5,
    steps: int | None = None,
    base_url: str = "ws://localhost:50001",
) -> bytes:
    params = []
    params.append(f"text={text.replace(' ', '%20')}")
    params.append(f"cfg={cfg_scale}")
    if steps is not None:
        params.append(f"steps={steps}")
    if voice:
        params.append(f"voice={voice}")
    query = "&".join(params)
    url = f"{base_url}/stream?{query}"

    chunks: list[bytes] = []

    async with websockets.connect(url) as ws:
        async for message in ws:
            if isinstance(message, bytes):
                chunks.append(message)

    return b"".join(chunks)


def test_vibevoice_streaming():
    audio = asyncio.get_event_loop().run_until_complete(
        stream_tts()
    )
    assert isinstance(audio, (bytes, bytearray))
    assert len(audio) > 0

