"use client";

import { useEffect, useRef, useState } from "react";
import clsx from "clsx";

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6 mb-8">
      <h2 className="text-lg font-semibold text-gray-900 mb-4">{title}</h2>
      {children}
    </section>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-2 mb-4">
      <label className="text-sm font-medium text-gray-700">{label}</label>
      {children}
    </div>
  );
}

export default function AudioPage() {
  const [text, setText] = useState("Hello from the Audio page using VibeVoice streaming.");
  const [voiceOptions, setVoiceOptions] = useState<string[]>([]);
  const [voice, setVoice] = useState<string>("");
  const [cfgScale, setCfgScale] = useState<number>(1.5);
  const [steps, setSteps] = useState<number>(5);
  const [isStreaming, setIsStreaming] = useState(false);
  const [status, setStatus] = useState<string>("");

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const scriptNodeRef = useRef<ScriptProcessorNode | null>(null);
  const pcmBufferRef = useRef<Float32Array>(new Float32Array(0));
  const hasStartedRef = useRef(false);
  const silentFramesRef = useRef(0);

  useEffect(() => {
    const loadVoices = async () => {
      try {
        const r = await fetch("http://localhost:50001/config");
        if (!r.ok) {
          setStatus("Failed to load voices (" + r.status + ")");
          return;
        }
        const data = await r.json();
        const voices: string[] = data.voices || [];
        setVoiceOptions(voices);
        if (data.default_voice) {
          setVoice(data.default_voice);
        } else if (voices.length > 0) {
          setVoice(voices[0]);
        }
      } catch {
        setStatus("Failed to load voices");
      }
    };
    loadVoices();
  }, []);

  const [canStream, setCanStream] = useState(false);

  useEffect(() => {
    const supported =
      typeof window !== "undefined" &&
      typeof WebSocket !== "undefined" &&
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      ((window as any).AudioContext || (window as any).webkitAudioContext);
    if (supported) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setCanStream(true);
    }
  }, []);

  const appendAudio = (chunk: Float32Array) => {
    const existing = pcmBufferRef.current;
    const merged = new Float32Array(existing.length + chunk.length);
    merged.set(existing, 0);
    merged.set(chunk, existing.length);
    pcmBufferRef.current = merged;
  };

  const pullAudio = (frameCount: number) => {
    const available = pcmBufferRef.current.length;
    if (available === 0) {
      return new Float32Array(frameCount);
    }
    if (available <= frameCount) {
      const chunk = pcmBufferRef.current;
      pcmBufferRef.current = new Float32Array(0);
      if (chunk.length < frameCount) {
        const padded = new Float32Array(frameCount);
        padded.set(chunk, 0);
        return padded;
      }
      return chunk;
    }
    const chunk = pcmBufferRef.current.subarray(0, frameCount);
    pcmBufferRef.current = pcmBufferRef.current.subarray(frameCount);
    return chunk;
  };

  const closeSocket = () => {
    const ws = wsRef.current;
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
      ws.close();
    }
    wsRef.current = null;
  };

  const teardownAudio = () => {
    if (scriptNodeRef.current) {
      try {
        scriptNodeRef.current.disconnect();
      } catch {}
      scriptNodeRef.current.onaudioprocess = null;
    }
    if (audioCtxRef.current) {
      try {
        audioCtxRef.current.close();
      } catch {}
    }
    audioCtxRef.current = null;
    scriptNodeRef.current = null;
  };

  const stopStreamInternal = (message?: string) => {
    setIsStreaming(false);
    if (message) {
      setStatus(message);
    }
    closeSocket();
    teardownAudio();
    pcmBufferRef.current = new Float32Array(0);
    hasStartedRef.current = false;
    silentFramesRef.current = 0;
  };

  const createAudioChain = () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const anyWindow = window as any;
    const AudioCtx = anyWindow.AudioContext || anyWindow.webkitAudioContext;
    if (!AudioCtx) {
      stopStreamInternal("AudioContext not available.");
      return;
    }
    teardownAudio();
    pcmBufferRef.current = new Float32Array(0);
    hasStartedRef.current = false;
    silentFramesRef.current = 0;
    const ctx: AudioContext = new AudioCtx({ sampleRate: 24000 });
    const scriptNode = ctx.createScriptProcessor(2048, 0, 1);
    const minBufferSamples = Math.floor(ctx.sampleRate * 0.1);
    scriptNode.onaudioprocess = (event: AudioProcessingEvent) => {
      const output = event.outputBuffer.getChannelData(0);
      const needPrebuffer = !hasStartedRef.current;
      const ws = wsRef.current;
      const socketClosed =
        !ws || ws.readyState === WebSocket.CLOSED || ws.readyState === WebSocket.CLOSING;
      if (needPrebuffer) {
        if (pcmBufferRef.current.length >= minBufferSamples || socketClosed) {
          hasStartedRef.current = true;
        } else {
          output.fill(0);
          return;
        }
      }
      const chunk = pullAudio(output.length);
      output.set(chunk);
      if (
        socketClosed &&
        pcmBufferRef.current.length === 0 &&
        chunk.every((sample) => sample === 0)
      ) {
        silentFramesRef.current += 1;
        if (silentFramesRef.current >= 4) {
          stopStreamInternal("Stream ended");
        }
      } else {
        silentFramesRef.current = 0;
      }
    };
    scriptNode.connect(ctx.destination);
    audioCtxRef.current = ctx;
    scriptNodeRef.current = scriptNode;
  };

  const startStream = async () => {
    if (!text.trim()) return;
    setIsStreaming(true);
    setStatus("Connecting…");

    if (!canStream) {
      setStatus("Streaming not supported by this browser.");
      setIsStreaming(false);
      return;
    }

    createAudioChain();

    const url = new URL("http://localhost:50001/stream");
    url.searchParams.set("text", text);
    url.searchParams.set("cfg", String(cfgScale));
    url.searchParams.set("steps", String(steps));
    if (voice) url.searchParams.set("voice", voice);

    const ws = new WebSocket(url.toString().replace("http", "ws"));
    wsRef.current = ws;
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      setStatus("Streaming audio…");
    };
    ws.onmessage = (event) => {
      if (typeof event.data === "string") {
        return;
      }
      const raw = event.data as ArrayBuffer;
      const view = new DataView(raw);
      const floatChunk = new Float32Array(view.byteLength / 2);
      for (let i = 0; i < floatChunk.length; i += 1) {
        floatChunk[i] = view.getInt16(i * 2, true) / 32768;
      }
      appendAudio(floatChunk);
    };
    ws.onclose = () => {
      wsRef.current = null;
    };
    ws.onerror = () => {
      stopStreamInternal("Stream error");
    };
  };

  useEffect(() => {
    return () => {
      closeSocket();
      teardownAudio();
    };
  }, []);

  const synthOnce = async () => {
    try {
      setStatus("Generating single WAV…");
      const resp = await fetch("http://localhost:50001/v1/audio/speech", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "vibevoice",
          input: text,
          voice,
        }),
      });
      if (!resp.ok) {
        setStatus("Error " + resp.status);
        return;
      }
      const buf = await resp.arrayBuffer();
      const blob = new Blob([buf], { type: "audio/wav" });
      const url = URL.createObjectURL(blob);
      const audioEl = audioRef.current!;
      audioEl.src = url;
      audioEl.play().catch(() => {});
      setStatus("Ready");
    } catch {
      setStatus("Failed to fetch audio");
    }
  };

  return (
    <div className="space-y-8">
      <div className="bg-gradient-to-r from-indigo-600 to-purple-600 rounded-2xl p-6 text-white shadow-lg">
        <h1 className="text-2xl font-bold mb-2">Audio Lab</h1>
        <p className="text-indigo-100">Test streaming and non-streaming TTS with professional controls.</p>
      </div>

      <Section title="Synthesis Controls">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Field label="Text">
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={4}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 outline-none"
            />
          </Field>
          <div>
            <Field label="Voice">
              <select
                value={voice}
                onChange={(e) => setVoice(e.target.value)}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 outline-none"
              >
                {voiceOptions.length === 0 ? (
                  <option>Loading…</option>
                ) : (
                  voiceOptions.map((v) => <option key={v} value={v}>{v}</option>)
                )}
              </select>
            </Field>
            <div className="grid grid-cols-2 gap-4">
              <Field label="CFG Scale">
                <input
                  type="range"
                  min={1.3}
                  max={3}
                  step={0.05}
                  value={cfgScale}
                  onChange={(e) => setCfgScale(parseFloat(e.target.value))}
                />
                <div className="text-xs text-gray-500 mt-1">Current: {cfgScale.toFixed(2)}</div>
              </Field>
              <Field label="Inference Steps">
                <input
                  type="number"
                  min={1}
                  max={30}
                  value={steps}
                  onChange={(e) => setSteps(parseInt(e.target.value || "5"))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg"
                />
              </Field>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-4 mt-2">
          <button
            onClick={startStream}
            disabled={isStreaming}
            className={clsx(
              "px-5 py-2.5 rounded-lg font-semibold text-white transition-colors",
              isStreaming ? "bg-gray-400" : "bg-indigo-600 hover:bg-indigo-700"
            )}
          >
            {isStreaming ? "Streaming…" : "Start Streaming"}
          </button>
          <button
            onClick={synthOnce}
            className="px-5 py-2.5 rounded-lg font-semibold text-indigo-600 border border-indigo-200 hover:bg-indigo-50 transition-colors"
          >
            Generate Single WAV
          </button>
          <span className="text-sm text-gray-500">{status}</span>
        </div>
      </Section>

      <Section title="Playback">
        <audio ref={audioRef} controls className="w-full" />
        <p className="text-xs text-gray-500 mt-2">
          Streaming uses WebSocket PCM16 chunks played via AudioContext; single requests return WAV.
        </p>
      </Section>
    </div>
  );
}
