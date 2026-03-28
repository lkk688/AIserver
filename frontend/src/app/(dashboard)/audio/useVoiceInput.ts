import { useState, useEffect, useRef, useCallback } from 'react';
import { isNativeASRAvailable, isWebGPUAvailable } from './asr-detection';
import { config } from '@/config';

export type ASRType = 'native' | 'webgpu' | 'none';

export function useVoiceInput(language: 'zh' | 'en' = 'en') {
    const [asrType, setAsrType] = useState<ASRType>('none');
    const [availableASR, setAvailableASR] = useState<ASRType[]>([]);

    const [isRecording, setIsRecording] = useState(false);
    const [isProcessing, setIsProcessing] = useState(false);
    const [isReady, setIsReady] = useState(false);
    const [transcript, setTranscript] = useState('');
    const [finalTranscript, setFinalTranscript] = useState('');
    const [error, setError] = useState<string | null>(null);
    const [statusMessage, setStatusMessage] = useState('');
    const [volume, setVolume] = useState(0);

    const recognitionRef = useRef<any>(null);
    const workerRef = useRef<Worker | null>(null);
    const audioContextRef = useRef<AudioContext | null>(null);
    const mediaStreamRef = useRef<MediaStream | null>(null);
    const audioBufferRef = useRef<Float32Array[]>([]);
    const startTimeRef = useRef<number>(0);
    const ignoreResultsRef = useRef<boolean>(false);
    // When true, recognition restarts automatically after silence ends (for hold-to-talk)
    const autoRestartRef = useRef<boolean>(false);

    // Debug: Log language changes
    useEffect(() => {
        console.log(`[ASR] Language changed to: ${language}`);
    }, [language]);

    // Detection
    useEffect(() => {
        const check = async () => {
            const native = isNativeASRAvailable();
            const webgpu = await isWebGPUAvailable();

            console.log(`[ASR] Detection - Native: ${native}, WebGPU: ${webgpu}`);

            const available: ASRType[] = [];
            if (native) available.push('native');
            if (webgpu) available.push('webgpu');

            setAvailableASR(available);

            if (native) {
                setAsrType('native');
                setIsReady(true);
            } else if (webgpu) {
                setAsrType('webgpu');
            } else {
                setAsrType('none');
                setError('Voice input not supported');
            }
        };
        check();
    }, []);

    // WebGPU Worker
    useEffect(() => {
        if (asrType === 'webgpu') {
            setIsReady(false);
            setStatusMessage('Loading Whisper model...');
            console.log("[ASR] Initializing WebGPU Worker...");

            const worker = new Worker('/asr/whisper-worker.js', { type: 'module' });
            workerRef.current = worker;

            worker.onmessage = (e) => {
                const { type, result, error, data } = e.data;
                if (type === 'ready') {
                    console.log("[ASR] WebGPU Model Ready");
                    setIsReady(true);
                    setStatusMessage('Model loaded');
                } else if (type === 'result') {
                    console.log("[ASR] WebGPU Result:", result);
                    setFinalTranscript(prev => (prev + ' ' + result.text).trim());
                    setIsProcessing(false);
                    setStatusMessage('');
                } else if (type === 'error') {
                    console.error("[ASR] WebGPU Error:", error);
                    setError(error);
                    setIsProcessing(false);
                    setStatusMessage('Error processing audio');
                } else if (type === 'progress') {
                    if (data?.status === 'progress') {
                        setStatusMessage(`Loading: ${Math.round(data.progress)}%`);
                    }
                }
            };

            worker.postMessage({ type: 'load' });

            return () => {
                worker.terminate();
                workerRef.current = null;
            };
        } else {
            if (workerRef.current) {
                workerRef.current.terminate();
                workerRef.current = null;
            }
            if (asrType === 'native') setIsReady(true);
        }
    }, [asrType]);

    const startRecording = useCallback(async () => {
        setError(null);
        ignoreResultsRef.current = false;
        startTimeRef.current = Date.now();
        console.log(`[ASR] Start Recording - Type: ${asrType}, Lang: ${language}`);

        if (asrType === 'native') {
            const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
            if (!SpeechRecognition) return;

            const recognition = new SpeechRecognition();
            recognition.continuous = true;
            recognition.interimResults = true;
            recognition.lang = language === 'zh' ? 'zh-CN' : 'en-US';
            console.log(`[ASR] Native Recognition Lang set to: ${recognition.lang}`);

            recognition.onresult = (event: any) => {
                if (ignoreResultsRef.current) return;
                let interim = '';
                let final = '';
                for (let i = event.resultIndex; i < event.results.length; ++i) {
                    if (event.results[i].isFinal) {
                        final += event.results[i][0].transcript;
                    } else {
                        interim += event.results[i][0].transcript;
                    }
                }
                if (final) {
                    console.log("[ASR] Native Final:", final);
                    setFinalTranscript(prev => (prev + ' ' + final).trim());
                }
                setTranscript(interim);
            };

            recognition.onerror = (event: any) => {
                if (ignoreResultsRef.current) return;
                console.error('[ASR] Native Error:', event.error);
                if (event.error === 'not-allowed') {
                    setError('Microphone permission denied');
                } else {
                    setError(event.error);
                }
                setIsRecording(false);
            };

            recognition.onend = () => {
                console.log("[ASR] Native Recognition Ended");
                setIsRecording(false);
                recognitionRef.current = null;
                // Auto-restart for hold-to-talk: browser stops after silence, we restart seamlessly
                if (autoRestartRef.current) {
                    setTimeout(() => {
                        if (autoRestartRef.current) {
                            const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
                            if (!SR) return;
                            const r2 = new SR();
                            r2.continuous = true;
                            r2.interimResults = true;
                            r2.lang = language === 'zh' ? 'zh-CN' : 'en-US';
                            r2.onresult = recognition.onresult;
                            r2.onerror = recognition.onerror;
                            r2.onend = recognition.onend;
                            recognitionRef.current = r2;
                            r2.start();
                            setIsRecording(true);
                        }
                    }, 80);
                }
            };

            recognitionRef.current = recognition;
            recognition.start();
            setIsRecording(true);

        } else if (asrType === 'webgpu') {
            if (!isReady) return;
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaStreamRef.current = stream;
                audioContextRef.current = new AudioContext({ sampleRate: 16000 });

                await audioContextRef.current.audioWorklet.addModule('/asr/audio-processor.js');
                const source = audioContextRef.current.createMediaStreamSource(stream);
                const worklet = new AudioWorkletNode(audioContextRef.current, 'audio-processor');

                audioBufferRef.current = [];

                worklet.port.onmessage = (e) => {
                    audioBufferRef.current.push(e.data);
                    const sum = e.data.reduce((a: number, b: number) => a + Math.abs(b), 0);
                    setVolume(Math.min(100, (sum / e.data.length) * 500));
                };

                source.connect(worklet);
                worklet.connect(audioContextRef.current.destination);

                setIsRecording(true);
                setStatusMessage('Recording...');
            } catch (err: any) {
                console.error("[ASR] WebGPU Recording Error:", err);
                setError(err.message);
            }
        }
    }, [asrType, isReady, language]);

    const stopRecording = useCallback(() => {
        const duration = Date.now() - startTimeRef.current;
        console.log(`[ASR] Stop Recording. Duration: ${duration}ms`);

        if (duration < 500) {
            console.log("[ASR] Recording too short, ignoring...");
            ignoreResultsRef.current = true;
            resetTranscript();
            setStatusMessage('');
            setError(null);

            // Clean up resources even if ignoring
            if (asrType === 'native' && recognitionRef.current) {
                recognitionRef.current.abort(); // Use abort for immediate stop
            } else if (asrType === 'webgpu') {
                setIsRecording(false);
                if (mediaStreamRef.current) {
                    mediaStreamRef.current.getTracks().forEach(t => t.stop());
                }
                if (audioContextRef.current) {
                    audioContextRef.current.close();
                }
                audioBufferRef.current = [];
            }
            return;
        }

        if (asrType === 'native') {
            if (recognitionRef.current) {
                recognitionRef.current.stop();
            }
        } else if (asrType === 'webgpu') {
            setIsRecording(false);
            setStatusMessage('Processing...');
            if (mediaStreamRef.current) {
                mediaStreamRef.current.getTracks().forEach(t => t.stop());
            }
            if (audioContextRef.current) {
                audioContextRef.current.close();
            }

            if (workerRef.current && audioBufferRef.current.length > 0) {
                setIsProcessing(true);
                const totalLength = audioBufferRef.current.reduce((acc, val) => acc + val.length, 0);
                const result = new Float32Array(totalLength);
                let offset = 0;
                for (const chunk of audioBufferRef.current) {
                    result.set(chunk, offset);
                    offset += chunk.length;
                }

                console.log(`[ASR] Sending ${totalLength} samples to WebGPU worker. Lang: ${language}`);
                workerRef.current.postMessage({
                    type: 'run',
                    audio: result,
                    language: language // Pass 'en' or 'zh' directly, don't map to 'chinese'/'english'
                });
                audioBufferRef.current = [];
            } else {
                setStatusMessage('');
            }
        }
    }, [asrType, language]);

    const resetTranscript = useCallback(() => {
        setTranscript('');
        setFinalTranscript('');
    }, []);

    const requestPermission = useCallback(async () => {
        console.log(`[ASR] Requesting Permission - Type: ${asrType}`);
        if (asrType === 'native') {
            const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
            if (SpeechRecognition) {
                try {
                    const recognition = new SpeechRecognition();
                    // Just start and stop to trigger permission prompt
                    recognition.onstart = () => {
                        console.log("[ASR] Permission granted (warmup)");
                        recognition.stop();
                    };
                    recognition.start();
                } catch (e) {
                    console.error("[ASR] Failed to request permission", e);
                }
            }
        } else if (asrType === 'webgpu') {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                stream.getTracks().forEach(t => t.stop());
                console.log("[ASR] Permission granted (warmup)");
            } catch (e) {
                console.error("[ASR] Permission denied", e);
                setError('Microphone permission denied');
            }
        }
    }, [asrType]);

    return {
        isRecording,
        isProcessing,
        isReady,
        transcript,
        finalTranscript,
        error,
        statusMessage,
        volume,
        asrType,
        setAsrType,
        availableASR,
        startRecording,
        stopRecording,
        resetTranscript,
        requestPermission, // Exported
        setAutoRestart: (val: boolean) => { autoRestartRef.current = val; },
        mode: asrType as any,
        setMode: (mode: any) => setAsrType(mode)
    };
}
