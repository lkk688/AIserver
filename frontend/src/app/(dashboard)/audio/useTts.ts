
import { useState, useEffect, useRef, useCallback } from 'react';
import { isNativeTTSAvailable, isWebGPUAvailable } from './asr-detection';
import { config } from '@/config';

export type TTSType = 'native' | 'webgpu' | 'doubao' | 'none';

export function useTts(language: 'zh' | 'en' = 'en') {
    const [ttsType, setTtsType] = useState<TTSType>('none');
    const [availableTTS, setAvailableTTS] = useState<TTSType[]>([]);
    const [isEnabled, setIsEnabled] = useState(false);
    const [isSpeaking, setIsSpeaking] = useState(false);
    const [isPlaying, setIsPlayingState] = useState(false); // Tracks if audio is actually emitting sound
    const [isPaused, setIsPaused] = useState(false);
    const isPausedRef = useRef(false); // Ref to track pause state synchronously for async callbacks
    const [isReady, setIsReadyState] = useState(false);
    const isReadyRef = useRef(false);

    // Volume Control
    const [volume, setVolume] = useState(1.0);
    const gainNodeRef = useRef<GainNode | null>(null);

    // Audio Cache for Cloud TTS (Text -> BlobURL)
    const audioCacheRef = useRef<Map<string, string>>(new Map());
    const currentAudioRef = useRef<HTMLAudioElement | null>(null);

    const setIsReady = (ready: boolean) => {
        setIsReadyState(ready);
        isReadyRef.current = ready;
    };
    const [statusMessage, setStatusMessage] = useState('');

    // Expose current text to allow UI to know what is playing
    const [currentText, setCurrentText] = useState<string | null>(null);
    const [currentId, setCurrentId] = useState<string | null>(null);

    const workerRef = useRef<Worker | null>(null);
    const audioContextRef = useRef<AudioContext | null>(null);
    const queueRef = useRef<{ text: string, id?: string }[]>([]);
    const isProcessingRef = useRef(false);
    const currentTextRef = useRef<string | null>(null); // Track what is currently being spoken

    // Sync ref with state
    useEffect(() => {
        setCurrentText(currentTextRef.current);
    }, [isSpeaking, isPaused]); // Update when status changes, but we might need more granular updates

    // We need to update currentText state whenever currentTextRef changes.
    // Since ref changes don't trigger re-render, we usually update state where we update ref.
    const updateCurrentText = (text: string | null, id?: string) => {
        currentTextRef.current = text;
        setCurrentText(text);
        if (id !== undefined) setCurrentId(id || null);
    };

    const sessionIdRef = useRef(0);
    const pendingRequestsRef = useRef(0);
    // Use a ref to hold the latest processQueue function to avoid stale closures in worker callback
    const processQueueRef = useRef<() => void>(() => { });

    // Reordering Refs for async TTS (Doubao)
    const nextRequestSequenceRef = useRef(0);
    const nextPlaybackSequenceRef = useRef(0);
    const audioBufferCacheRef = useRef<Map<number, AudioBuffer>>(new Map());

    // Helper to clean markdown and newlines
    const cleanText = (text: string) => {
        // Replace newlines with space to avoid accidental splitting
        return text.replace(/[*#`_\[\]]/g, '').replace(/\n/g, ' ').trim();
    };

    // Detection
    useEffect(() => {
        const check = async () => {
            const native = isNativeTTSAvailable();
            const webgpu = await isWebGPUAvailable();

            const available: TTSType[] = [];
            if (native) available.push('native');
            if (webgpu) available.push('webgpu');
            available.push('doubao'); // Always available

            setAvailableTTS(available);

            // Select based on config preference
            const pref = config.tts.backend; // 'native' | 'webgpu' | 'doubao' | 'auto'

            if (pref === 'native' && native) {
                setTtsType('native');
                setIsReady(true);
            } else if (pref === 'webgpu' && webgpu) {
                setTtsType('webgpu');
            } else if (pref === 'doubao') {
                setTtsType('doubao');
                setIsReady(true);
            } else {
                // Default fallback
                if (native) {
                    setTtsType('native');
                    setIsReady(true);
                } else if (webgpu) {
                    setTtsType('webgpu');
                } else {
                    setTtsType('doubao');
                    setIsReady(true);
                }
            }
        };
        check();
    }, []);

    const audioQueueRef = useRef<{ buffer: AudioBuffer, source?: AudioBufferSourceNode, id?: string }[]>([]);
    const isPlayingAudioRef = useRef(false);

    // WebGPU Worker Init
    useEffect(() => {
        if (ttsType === 'webgpu' && isEnabled) {
            setIsReady(false);
            setStatusMessage('Loading TTS model...');

            let worker: Worker;
            try {
                // Determine which model to use based on language
                const modelConfig = language === 'zh' ? config.tts.webgpu.zh : config.tts.webgpu.en;
                const useSherpa = modelConfig.model === 'sherpa';

                if (useSherpa) {
                    // Sherpa-ONNX Worker (Classic)
                    worker = new Worker('/asr/sherpa-tts-worker.js?v=20');
                } else {
                    // Transformers.js / Kokoro.js Worker (Module)
                    // Add version to force reload worker and avoid caching issues
                    worker = new Worker('/asr/tts-worker.js?v=36', { type: 'module' });
                }

                workerRef.current = worker;

                worker.onerror = (err) => {
                    console.error("[TTS] WebGPU Worker Error:", err);
                    setStatusMessage('Model Error');
                    setIsReady(false);
                    // Fallback to Doubao if WebGPU crashes
                    setTtsType('doubao');
                };

                worker.onmessage = (e) => {
                    const { type, audio, sampling_rate, error, data, sessionId } = e.data;

                    if (type === 'ready' || type === 'wasm_ready') {
                        if (type === 'ready') {
                            console.log("[TTS] WebGPU Model Ready");
                            setIsReady(true);
                            setStatusMessage('Model Ready');
                            // Trigger queue processing when ready via ref
                            processQueueRef.current();
                        }
                    } else if (type === 'result') {
                        // Check session ID to ignore stale results from cancelled requests
                        if (sessionId !== undefined && sessionId !== sessionIdRef.current) {
                            console.log(`[TTS] Ignoring stale result (Session: ${sessionId} != ${sessionIdRef.current})`);
                            return;
                        }

                        console.log(`[TTS] Received Result (Session: ${sessionId}). Pending before dec: ${pendingRequestsRef.current}`);
                        pendingRequestsRef.current = Math.max(0, pendingRequestsRef.current - 1);
                        // Native WebGPU doesn't easily support ID tracking per chunk yet unless we pass it through worker.
                        // For now, assuming queueAudio can take undefined ID or we need to track it.
                        // But since WebGPU is one-at-a-time usually, we can maybe peek queue?
                        // Actually, WebGPU worker returns result for a request.
                        // We need to know which request it was.
                        // But for simplicity, let's skip ID for WebGPU for now or implement it properly later if needed.
                        queueAudio(audio, sampling_rate);

                        isProcessingRef.current = false; // Mark processing done so next item can be picked up

                        // Trigger next item in queue via ref
                        processQueueRef.current();
                    } else if (type === 'complete') {
                        // Trigger next item in queue via ref
                        processQueueRef.current();
                    } else if (type === 'error') {
                        console.error("[TTS] WebGPU Error:", error);
                        if (sessionId !== undefined && sessionId !== sessionIdRef.current) {
                            // Ignore errors from stale sessions
                            return;
                        }
                        pendingRequestsRef.current = Math.max(0, pendingRequestsRef.current - 1);
                        // Trigger next item in queue via ref
                        processQueueRef.current();
                    } else if (type === 'progress') {
                        if (data?.status === 'progress') {
                            setStatusMessage(`Loading TTS: ${Math.round(data.progress)}%`);
                        }
                    }
                };

                // Send load message
                if (useSherpa) {
                    worker.postMessage({ type: 'load' });
                } else {
                    const targetModelId = modelConfig.model;
                    worker.postMessage({
                        type: 'load',
                        model_id: targetModelId
                    });
                }

                return () => {
                    worker.terminate();
                    workerRef.current = null;
                };

            } catch (err) {
                console.error("[TTS] Failed to initialize WebGPU worker:", err);
                setTtsType('doubao'); // Fallback
            }
        }
    }, [ttsType, isEnabled, language]);

    const queueAudio = async (audioData: Float32Array, sampleRate: number, id?: string) => {
        if (!audioContextRef.current) {
            audioContextRef.current = new AudioContext();
        }
        const ctx = audioContextRef.current;

        // Ensure Gain Node exists
        if (!gainNodeRef.current) {
            gainNodeRef.current = ctx.createGain();
            gainNodeRef.current.gain.value = volume;
            gainNodeRef.current.connect(ctx.destination);
        }

        const buffer = ctx.createBuffer(1, audioData.length, sampleRate);
        buffer.getChannelData(0).set(audioData);

        audioQueueRef.current.push({ buffer, id });
        playNextAudio();
    };

    const playNextAudio = () => {
        if (isPlayingAudioRef.current || audioQueueRef.current.length === 0 || !audioContextRef.current) return;

        // Ensure context is running (it might be suspended from a pause)
        if (audioContextRef.current.state === 'suspended' && !isPausedRef.current) {
            audioContextRef.current.resume();
        }

        // If explicitly paused, don't start new segment
        if (isPausedRef.current) return;

        isPlayingAudioRef.current = true;
        // Re-affirm speaking state just in case
        setIsSpeaking(true);
        setIsPlayingState(true);

        const item = audioQueueRef.current[0]; // Peek

        // Update ID if present
        if (item.id) {
            console.log(`[TTS] playNextAudio updating currentId to ${item.id}`);
            setCurrentId(item.id);
        }

        const source = audioContextRef.current.createBufferSource();
        source.buffer = item.buffer;

        // Connect through Gain Node for volume control
        if (!gainNodeRef.current) {
            gainNodeRef.current = audioContextRef.current.createGain();
            gainNodeRef.current.gain.value = volume;
            gainNodeRef.current.connect(audioContextRef.current.destination);
        }
        source.connect(gainNodeRef.current);

        // Store source to be able to stop it
        item.source = source;

        source.onended = () => {
            audioQueueRef.current.shift(); // Remove played
            isPlayingAudioRef.current = false;

            // Check if queue is empty or play next
            if (audioQueueRef.current.length > 0) {
                playNextAudio();
            } else {
                setIsPlayingState(false); // No audio playing right now

                // If queue is empty AND text queue is empty
                // Also check pendingRequests for streaming TTS to avoid premature stop
                if (queueRef.current.length === 0 && !isProcessingRef.current && pendingRequestsRef.current === 0) {
                    setIsSpeaking(false);
                    setIsPaused(false);
                    isPausedRef.current = false;
                    updateCurrentText(null);
                    setCurrentId(null);
                }
            }
        };

        source.start(0);
    };

    // Limit concurrent requests to prevent flooding the worker
    // 1 = Serial (Robust), 2+ = Pipelined (Faster)
    const MAX_CONCURRENT_REQUESTS = 5;

    const processQueue = useCallback(() => {
        if (queueRef.current.length === 0) return;

        console.log(`[TTS] processQueue. ttsType=${ttsType}`);

        if (ttsType === 'native') {
            if (isProcessingRef.current) return;

            const item = queueRef.current.shift();
            if (!item) return;
            const { text, id } = item;

            currentTextRef.current = text;
            setCurrentText(text); // Sync state
            if (id) setCurrentId(id);

            setIsSpeaking(true);
            // Don't auto-resume if already paused (streaming/queue processing)
            // setIsPaused(false); 
            // isPausedRef.current = false;
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = language === 'zh' ? 'zh-CN' : 'en-US';
            utterance.volume = volume; // Set volume

            // Attempt to find a "better" voice
            // Wait for voices to be loaded if they aren't yet
            const setVoice = () => {
                const voices = window.speechSynthesis.getVoices();
                if (voices.length > 0) {
                    const preferredVoice = voices.find(v =>
                        v.lang.startsWith(language === 'zh' ? 'zh' : 'en') &&
                        (v.name.includes('Google') || v.name.includes('Microsoft') || v.name.includes('Natural'))
                    );
                    if (preferredVoice) {
                        utterance.voice = preferredVoice;
                    }
                }
            };

            if (window.speechSynthesis.getVoices().length === 0) {
                window.speechSynthesis.onvoiceschanged = setVoice;
            } else {
                setVoice();
            }

            utterance.onend = () => {
                if (queueRef.current.length === 0) {
                    setIsSpeaking(false);
                    setIsPaused(false);
                    updateCurrentText(null);
                }
                isProcessingRef.current = false;
                processQueue();
            };
            utterance.onerror = () => {
                isProcessingRef.current = false;
                processQueue();
            };
            isProcessingRef.current = true;
            window.speechSynthesis.speak(utterance);
        } else if (ttsType === 'webgpu') {
            if (isProcessingRef.current || !workerRef.current || !isReady) return;

            const item = queueRef.current.shift();
            if (!item) return;
            const { text, id } = item;

            console.log(`[TTS] WebGPU processing item. text="${text.substring(0, 10)}..."`);

            updateCurrentText(text, id);
            setIsSpeaking(true);
            setStatusMessage('Generating audio...');
            pendingRequestsRef.current++;
            isProcessingRef.current = true;

            // Send to worker
            workerRef.current.postMessage({
                type: 'speak', // Changed from 'generate' to 'speak' to match worker
                text,
                language: language === 'zh' ? 'zh' : 'en',
                voice: language === 'zh' ? 'zf_xiaobei' : 'af_heart', // Default voices
                sessionId: sessionIdRef.current
            });

        } else if (ttsType === 'doubao') {
            // Streaming fetch
            while (queueRef.current.length > 0 && pendingRequestsRef.current < MAX_CONCURRENT_REQUESTS) {
                const item = queueRef.current.shift();
                if (!item) continue;

                const text = typeof item === 'string' ? item : item.text;
                const id = typeof item === 'object' ? item.id : undefined;
                console.log(`[TTS] Doubao processing item. text="${text.substring(0, 10)}...", id=${id}`);

                // Assign Sequence ID
                const requestId = nextRequestSequenceRef.current++;

                // Check Cache
                if (audioCacheRef.current.has(text)) {
                    const blobUrl = audioCacheRef.current.get(text)!;
                    console.log("[TTS] Playing from cache:", text.substring(0, 20));

                    // Decode cached blob and play
                    (async () => {
                        try {
                            const response = await fetch(blobUrl);
                            const arrayBuffer = await response.arrayBuffer();
                            if (!audioContextRef.current) audioContextRef.current = new AudioContext();

                            // Resume if suspended ONLY if not explicitly paused
                            if (audioContextRef.current.state === 'suspended' && !isPausedRef.current) await audioContextRef.current.resume();

                            const audioBuffer = await audioContextRef.current.decodeAudioData(arrayBuffer);

                            // Store in reordering buffer
                            audioBufferCacheRef.current.set(requestId, audioBuffer);

                            // Push to playback queue in correct order
                            while (audioBufferCacheRef.current.has(nextPlaybackSequenceRef.current)) {
                                const nextBuffer = audioBufferCacheRef.current.get(nextPlaybackSequenceRef.current)!;
                                audioBufferCacheRef.current.delete(nextPlaybackSequenceRef.current);

                                // Attach ID if it matches the current request being processed (best effort for cache)
                                const currentId = (nextPlaybackSequenceRef.current === requestId) ? id : undefined;
                                audioQueueRef.current.push({ buffer: nextBuffer, id: currentId });

                                nextPlaybackSequenceRef.current++;
                            }
                            playNextAudio();
                        } catch (e) {
                            console.error("Cache playback error", e);
                        }
                    })();
                    continue;
                }

                // If cache miss, proceed to fetch
                // ...

                updateCurrentText(text, id);
                console.log(`[TTS] Set speaking state. id=${id}`);
                setIsSpeaking(true);
                // Don't auto-resume if already paused (streaming/queue processing)
                // setIsPaused(false);
                // isPausedRef.current = false;
                setStatusMessage('Streaming audio...');
                pendingRequestsRef.current++;

                // Use a closure to capture current text context
                const currentSessionId = sessionIdRef.current;
                (async () => {
                    try {
                        // Double check cache inside async in case another request filled it (race condition)
                        if (audioCacheRef.current.has(text)) {
                            // ... (Reuse cache logic - simplified for now: just skip network and let the other request handle it? 
                            // No, duplicate requests might happen. 
                            // Ideally we should have a "pending" cache to avoid duplicate fetches for same text.)
                            // For now, let's just proceed.
                        }

                        const voice = config.tts.doubao.voice;
                        // Use Next.js API Route Proxy (which forwards to /v1/tts/doubao/stream)
                        // If in production and NGINX is configured to handle /v1/ directly, we can use that to bypass Next.js API route overhead/issues.
                        // Try /v1/tts/doubao/stream first (Nginx), fallback to /api/tts/doubao (Next.js)

                        // Actually, to fix the 404 on production where /api/ is intercepted by Nginx (sending to FastAPI which doesn't have /api/tts/doubao),
                        // we MUST use the path that Nginx correctly routes to the backend's TTS endpoint.
                        // The backend has /v1/tts/doubao/stream.
                        // We configured Nginx to proxy /v1/ -> backend.
                        // So we should use /v1/tts/doubao/stream directly.

                        const payload = {
                            text,
                            voice: voice || 'zh_female_vv_uranus_bigtts',
                            speed: 1.0,
                        };
                        const endpoints = ['/v1/tts/doubao/stream', '/api/tts/doubao'];
                        let response: Response | null = null;
                        let lastError = '';
                        for (const endpoint of endpoints) {
                            try {
                                const candidate = await fetch(endpoint, {
                                    method: 'POST',
                                    headers: { 'Content-Type': 'application/json' },
                                    body: JSON.stringify(payload),
                                });
                                if (candidate.ok && candidate.body) {
                                    response = candidate;
                                    break;
                                }
                                const raw = await candidate.text();
                                let parsed = raw;
                                try {
                                    const json = JSON.parse(raw);
                                    parsed = String(json?.detail || json?.error || raw);
                                } catch { }
                                lastError = `${endpoint}: ${candidate.status} ${parsed || candidate.statusText}`;
                                if (endpoint === '/v1/tts/doubao/stream' && candidate.status >= 500) {
                                    break;
                                }
                            } catch (err: any) {
                                lastError = `${endpoint}: ${String(err?.message || err)}`;
                            }
                        }

                        if (currentSessionId !== sessionIdRef.current) return;

                        if (!response || !response.body) {
                            throw new Error(`TTS API Error: ${lastError || 'No audio response from backend'}`);
                        }

                        const reader = response.body.getReader();
                        const chunks: Uint8Array[] = [];

                        while (true) {
                            const { done, value } = await reader.read();
                            if (done) break;
                            chunks.push(value);
                        }

                        if (currentSessionId !== sessionIdRef.current) return;

                        // Create Blob for Cache
                        const blob = new Blob(chunks as BlobPart[], { type: 'audio/mpeg' });
                        const blobUrl = URL.createObjectURL(blob);
                        audioCacheRef.current.set(text, blobUrl);
                        console.log("[TTS] Cached audio for:", text.substring(0, 20));

                        const totalLength = chunks.reduce((acc, c) => acc + c.length, 0);
                        const fullAudio = new Uint8Array(totalLength);
                        let offset = 0;
                        for (const chunk of chunks) {
                            fullAudio.set(chunk, offset);
                            offset += chunk.length;
                        }

                        // Decode
                        if (!audioContextRef.current) audioContextRef.current = new AudioContext();
                        if (audioContextRef.current.state === 'suspended' && !isPausedRef.current) await audioContextRef.current.resume();

                        const audioBuffer = await audioContextRef.current.decodeAudioData(fullAudio.buffer);

                        // Store in reordering buffer
                        audioBufferCacheRef.current.set(requestId, audioBuffer);

                        // Push to playback queue in correct order
                        while (audioBufferCacheRef.current.has(nextPlaybackSequenceRef.current)) {
                            const nextBuffer = audioBufferCacheRef.current.get(nextPlaybackSequenceRef.current)!;
                            audioBufferCacheRef.current.delete(nextPlaybackSequenceRef.current);
                            audioQueueRef.current.push({ buffer: nextBuffer });
                            nextPlaybackSequenceRef.current++;
                        }
                        playNextAudio();

                    } catch (e) {
                        console.error('[DoubaoTTS] Error:', e);
                        if (isNativeTTSAvailable() && currentSessionId === sessionIdRef.current) {
                            await new Promise<void>((resolve) => {
                                try {
                                    const utterance = new SpeechSynthesisUtterance(cleanText(text));
                                    utterance.lang = language === 'zh' ? 'zh-CN' : 'en-US';
                                    utterance.volume = volume;
                                    utterance.onstart = () => {
                                        setStatusMessage('Using browser TTS fallback');
                                        setIsSpeaking(true);
                                    };
                                    utterance.onend = () => resolve();
                                    utterance.onerror = () => resolve();
                                    window.speechSynthesis.speak(utterance);
                                } catch {
                                    resolve();
                                }
                            });
                        }
                    } finally {
                        if (currentSessionId === sessionIdRef.current) {
                            pendingRequestsRef.current = Math.max(0, pendingRequestsRef.current - 1);
                            processQueue(); // Process next
                        }
                    }
                })();
            }
        }
    }, [ttsType, language, volume]); // Removed isReady dependency

    const cancel = useCallback(() => {
        queueRef.current = [];
        currentTextRef.current = null;
        if (ttsType === 'native') {
            window.speechSynthesis.cancel();
            setIsSpeaking(false);
            setIsPlayingState(false);
            setIsPaused(false);
            isPausedRef.current = false;
            isProcessingRef.current = false;
            setCurrentId(null);
        } else {
            // Invalidate session
            sessionIdRef.current++;
            pendingRequestsRef.current = 0;

            if (audioContextRef.current) {
                // Stop any currently playing source
                // Iterate through ALL queued items and stop them
                audioQueueRef.current.forEach(item => {
                    try { item.source?.stop(); } catch (e) { }
                });

                // Suspend the context instead of closing it to preserve the "unlocked" state on mobile
                // But actually, we might want to keep it running for the next utterance?
                // If we suspend, we need to resume later.
                // Let's just stop the sources and clear the queue. 
                // Keeping the context 'running' is fine and better for responsiveness.
            }

            // Clear audio queue immediately
            audioQueueRef.current = [];
            isPlayingAudioRef.current = false;

            // Important: Reset state synchronously
            setIsSpeaking(false);
            setIsPlayingState(false);
            setIsPaused(false);
            isPausedRef.current = false;
            isProcessingRef.current = false;
            setCurrentId(null);

            // Reset sequence counters
            nextRequestSequenceRef.current = 0;
            nextPlaybackSequenceRef.current = 0;
            audioBufferCacheRef.current.clear();
        }
    }, [ttsType]);

    const speak = useCallback((text: string, force = false, id?: string) => {
        console.log(`[TTS] speak called. text="${text.substring(0, 10)}...", force=${force}, id=${id}, isEnabled=${isEnabled}`);

        // If not forced (manual click) and disabled, don't speak automatically
        if ((!isEnabled && !force) || !text.trim()) {
            console.log("[TTS] speak ignored (disabled or empty)");
            return;
        }

        // If forcing a new speech (manual click), cancel previous
        if (force) {
            console.log("[TTS] speak forcing, cancelling previous");
            cancel();
            // Use requestAnimationFrame or a very short timeout to allow state to settle
            // But if we want "immediate" response, we should clear and queue immediately.
            // The issue is if cancel() is async or if React state updates batching causes delay.
            // cancel() updates refs synchronously, but state updates are async.
            // However, we can just push to queue and call processQueue.

            // Explicitly reset isProcessingRef to allow processQueue to pick up immediately
            isProcessingRef.current = false;

            const cleaned = cleanText(text);
            if (!cleaned) return;

            // Intelligent splitting to improve perceived latency
            // Match sentence endings but keep delimiters.
            // Chinese: 。！？：，；
            // English: .!?:,; (avoid decimal points like 1.5, requiring space or end)

            // Regex explanation:
            // ([^.!?。！？：:，,;；]+[.!?。！？：:，,;；]+)  -> Match chunk ending with punctuation
            // |
            // ([^.!?。！？：:，,;；]+$)           -> Match remaining chunk at end

            // NOTE: We do NOT include \n in the regex class because we already removed them.
            // This ensures we only split on actual semantic punctuation.

            let mode = 'streaming';
            if (ttsType === 'webgpu') {
                const modelConfig = language === 'zh' ? config.tts.webgpu.zh : config.tts.webgpu.en;
                mode = modelConfig.mode || 'streaming';
            }

            let sentences: string[] = [];

            if (mode === 'batch') {
                // For batch mode, send the whole text as one chunk
                sentences = [cleaned];
            } else {
                // For streaming mode (WebGPU or Doubao), split into sentences/phrases
                // Note: We use the same split logic for Doubao now to reduce start latency
                // by fetching smaller initial chunks first.

                // Strategy: First chunk should be small (fast start), subsequent chunks can be larger (efficient network).
                // 1. Split by major punctuation first (Sentence level).
                const majorSentences = cleaned.match(/[^.!?。！？]+[.!?。！？]+|[^.!?。！？]+$/g) || [cleaned];

                const finalChunks: string[] = [];

                // 2. Process first sentence: Split aggressively (commas) to get audio ASAP
                if (majorSentences.length > 0) {
                    const firstSentence = majorSentences[0];
                    // Only split if it's actually long enough to matter
                    if (firstSentence.length > 20) {
                        // Original regex: /[^,，;；:：]+[,，;；:：]+|[^,，;；:：]+$/g
                        // We split by commas but we want to avoid creating tiny chunks (e.g. "Hi,").
                        // We accumulate chunks until they reach a threshold (e.g. 15 chars) to ensure better network efficiency and continuity.
                        const rawSubChunks = firstSentence.match(/[^,，;；:：]+[,，;；:：]+|[^,，;；:：]+$/g) || [firstSentence];

                        const mergedSubChunks: string[] = [];
                        let buffer = '';
                        const MIN_CHUNK_LENGTH = 15;

                        for (const chunk of rawSubChunks) {
                            buffer += chunk;
                            if (buffer.length >= MIN_CHUNK_LENGTH) {
                                mergedSubChunks.push(buffer);
                                buffer = '';
                            }
                        }
                        if (buffer) {
                            if (mergedSubChunks.length > 0) {
                                // Append remainder to last chunk to avoid a final tiny chunk
                                mergedSubChunks[mergedSubChunks.length - 1] += buffer;
                            } else {
                                mergedSubChunks.push(buffer);
                            }
                        }

                        finalChunks.push(...mergedSubChunks);
                    } else {
                        finalChunks.push(firstSentence);
                    }

                    // 3. Process remaining sentences: Keep them whole (or combine if very short)
                    // This minimizes network overhead for the bulk of the content.
                    // We can even combine multiple short sentences into one request if needed, 
                    // but for now, sentence-level granularity is a good balance.
                    for (let i = 1; i < majorSentences.length; i++) {
                        finalChunks.push(majorSentences[i]);
                    }
                }

                sentences = finalChunks;
            }

            sentences.forEach(s => {
                if (s.trim()) queueRef.current.push({ text: s.trim(), id: id || text }); // Assign ID (fallback to text)
            });

            console.log(`[TTS] Queued ${sentences.length} chunks. Triggering processQueue.`);
            // Trigger immediately
            processQueue();
            return;
        }

        const cleaned = cleanText(text);
        if (!cleaned) return;

        // If we were paused and adding new text (not forced), we might want to just queue it.
        // But if we are adding new text, we usually want to ensure we are 'speaking' state if not already.

        queueRef.current.push({ text: cleaned, id: id || text });
        processQueue();
    }, [isEnabled, processQueue, cancel]);

    // Update ref when processQueue changes
    useEffect(() => {
        processQueueRef.current = processQueue;
    }, [processQueue]);

    const pause = useCallback(() => {
        if (ttsType === 'native') {
            window.speechSynthesis.pause();
            setIsPaused(true);
            isPausedRef.current = true;
            setIsSpeaking(true); // Still speaking, just paused
        } else if (ttsType === 'webgpu' || ttsType === 'doubao') {
            if (audioContextRef.current && audioContextRef.current.state === 'running') {
                audioContextRef.current.suspend();
            }
            // Always update state, even if audioContext is not ready yet
            setIsPaused(true);
            isPausedRef.current = true;
            setIsSpeaking(true);
        }
    }, [ttsType]);

    const resume = useCallback(() => {
        if (ttsType === 'native') {
            // Chrome bug workaround: Sometimes resume() doesn't fire if pause was too long.
            // But standard API is simple.
            window.speechSynthesis.resume();
            setIsPaused(false);
            isPausedRef.current = false;
            setIsSpeaking(true);
        } else if (ttsType === 'webgpu' || ttsType === 'doubao') {
            if (audioContextRef.current && audioContextRef.current.state === 'suspended') {
                audioContextRef.current.resume();
            }

            // Always update state
            setIsPaused(false);
            isPausedRef.current = false;
            setIsSpeaking(true);

            // If we were waiting to play something (audio queue has items but wasn't playing)
            if (audioContextRef.current && !isPlayingAudioRef.current && audioQueueRef.current.length > 0) {
                playNextAudio();
            }
        }
    }, [ttsType]);

    const clearCache = useCallback(() => {
        console.log("[TTS] Clearing audio cache");
        // Revoke all blob URLs to free memory
        audioCacheRef.current.forEach((url) => {
            URL.revokeObjectURL(url);
        });
        audioCacheRef.current.clear();
    }, []);

    const initAudio = useCallback(() => {
        if (ttsType === 'webgpu' || ttsType === 'doubao') {
            if (!audioContextRef.current) {
                // Ensure sample rate is compatible with most devices (44100 or 48000)
                // iOS sometimes requires explicit unlock events
                const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
                if (AudioContextClass) {
                    audioContextRef.current = new AudioContextClass();
                }
            }
            if (audioContextRef.current) {
                // Resume if suspended
                if (audioContextRef.current.state === 'suspended') {
                    audioContextRef.current.resume().then(() => {
                        console.log("[TTS] AudioContext resumed successfully");
                    }).catch(err => console.error("[TTS] Failed to resume audio context:", err));
                }

                // Play silent buffer to unlock iOS audio session for volume control
                try {
                    const buffer = audioContextRef.current.createBuffer(1, 1, 22050);
                    const source = audioContextRef.current.createBufferSource();
                    source.buffer = buffer;
                    source.connect(audioContextRef.current.destination);
                    source.start(0);
                    console.log("[TTS] Silent buffer played to unlock audio session");
                } catch (e) {
                    console.error("[TTS] Failed to play silent buffer:", e);
                }
            }
        }
    }, [ttsType]);

    return {
        isEnabled,
        setIsEnabled,
        ttsType,
        setTtsType,
        availableTTS,
        speak,
        pause,
        resume,
        cancel,
        clearCache, // Exported
        initAudio, // Exported for mobile autoplay policy
        isSpeaking,
        isPlaying, // Exported
        isPaused,
        isReady,
        statusMessage,
        currentText, // Exported
        currentId, // Exported
        volume,
        setVolume
    };
}
