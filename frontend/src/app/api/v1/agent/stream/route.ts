/**
 * Next.js Route Handler for /api/v1/agent/stream
 *
 * Next.js rewrites() buffer SSE responses, breaking real-time streaming.
 * This Route Handler intercepts the request before the rewrite and proxies it
 * to the FastAPI backend using a passthrough ReadableStream, preserving
 * token-by-token delivery to the browser.
 */
import { NextRequest } from 'next/server';

export const dynamic = 'force-dynamic';

const BACKEND = 'http://127.0.0.1:8080';

export async function POST(req: NextRequest) {
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return new Response('Invalid JSON body', { status: 400 });
  }

  const backendRes = await fetch(`${BACKEND}/api/v1/agent/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify(body),
  });

  if (!backendRes.ok || !backendRes.body) {
    const errPayload = JSON.stringify({ type: 'error', detail: `Backend returned ${backendRes.status}` });
    return new Response(`data: ${errPayload}\n\n`, {
      status: 200,
      headers: { 'Content-Type': 'text/event-stream' },
    });
  }

  // Pass the ReadableStream straight through — no buffering.
  return new Response(backendRes.body, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache, no-transform',
      'X-Accel-Buffering': 'no',
      Connection: 'keep-alive',
    },
  });
}
