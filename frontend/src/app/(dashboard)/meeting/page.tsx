"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { Mic, MicOff, Video as VideoIcon, VideoOff, PhoneOff, PhoneCall, Circle, StopCircle, DownloadCloud } from "lucide-react";
import clsx from "clsx";

const ICE_SERVERS = {
  iceServers: [
    { urls: "stun:stun.l.google.com:19302" },
  ],
};

export default function MeetingPage() {
  const [roomId, setRoomId] = useState("room-1234");
  const [isInCall, setIsInCall] = useState(false);
  const [isAudioMuted, setIsAudioMuted] = useState(false);
  const [isVideoMuted, setIsVideoMuted] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [status, setStatus] = useState("");
  const [asrResult, setAsrResult] = useState("");

  const localVideoRef = useRef<HTMLVideoElement>(null);
  const remoteVideoRef = useRef<HTMLVideoElement>(null);

  const localStreamRef = useRef<MediaStream | null>(null);
  const peerConnectionRef = useRef<RTCPeerConnection | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // Recording refs
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<BlobPart[]>([]);

  const handleMessage = useCallback(async (event: MessageEvent) => {
    try {
      const data = JSON.parse(event.data);
      const pc = peerConnectionRef.current;
      if (!pc) return;

      if (data.type === "offer") {
        await pc.setRemoteDescription(new RTCSessionDescription(data));
        const answer = await pc.createAnswer();
        await pc.setLocalDescription(answer);
        wsRef.current?.send(JSON.stringify(pc.localDescription));
      } else if (data.type === "answer") {
        await pc.setRemoteDescription(new RTCSessionDescription(data));
      } else if (data.type === "candidate") {
        await pc.addIceCandidate(new RTCIceCandidate(data.candidate));
      } else if (data.type === "peer_left") {
        setStatus("Remote peer left the call.");
        if (remoteVideoRef.current) remoteVideoRef.current.srcObject = null;
      }
    } catch (e) {
      console.error("Error handling WS message:", e);
    }
  }, []);

  const joinCall = async () => {
    if (!roomId) return;
    try {
      setStatus("Requesting media access...");
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      localStreamRef.current = stream;
      if (localVideoRef.current) {
        localVideoRef.current.srcObject = stream;
      }

      setStatus("Connecting to signaling server...");
      const wsUrl = process.env.NEXT_PUBLIC_API_URL 
        ? process.env.NEXT_PUBLIC_API_URL.replace("http", "ws") 
        : "ws://localhost:8000/api/v1";
      
      const ws = new WebSocket(`${wsUrl}/meeting/ws/${roomId}`);
      wsRef.current = ws;

      ws.onopen = async () => {
        setStatus("Connected to room. Setting up peer connection...");
        const pc = new RTCPeerConnection(ICE_SERVERS);
        peerConnectionRef.current = pc;

        // Add local tracks to PC
        stream.getTracks().forEach((track) => pc.addTrack(track, stream));

        // Handle remote tracks
        pc.ontrack = (event) => {
          if (remoteVideoRef.current) {
            remoteVideoRef.current.srcObject = event.streams[0];
          }
        };

        // Handle ICE candidates
        pc.onicecandidate = (event) => {
          if (event.candidate && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: "candidate", candidate: event.candidate }));
          }
        };

        // Create offer (if we are the first, we send it. Usually both might try, so basic implementation)
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        ws.send(JSON.stringify(pc.localDescription));
        
        setIsInCall(true);
        setStatus("In call.");
      };

      ws.onmessage = handleMessage;

      ws.onclose = () => {
        if (isInCall) leaveCall();
      };

    } catch (e) {
      console.error(e);
      setStatus("Failed to join call. Check media permissions.");
    }
  };

  const leaveCall = () => {
    if (isRecording) {
      stopRecording();
    }
    
    localStreamRef.current?.getTracks().forEach(track => track.stop());
    localStreamRef.current = null;
    
    peerConnectionRef.current?.close();
    peerConnectionRef.current = null;

    wsRef.current?.close();
    wsRef.current = null;

    if (localVideoRef.current) localVideoRef.current.srcObject = null;
    if (remoteVideoRef.current) remoteVideoRef.current.srcObject = null;

    setIsInCall(false);
    setStatus("Disconnected.");
  };

  const toggleAudio = () => {
    if (localStreamRef.current) {
      const audioTrack = localStreamRef.current.getAudioTracks()[0];
      if (audioTrack) {
        audioTrack.enabled = !audioTrack.enabled;
        setIsAudioMuted(!audioTrack.enabled);
      }
    }
  };

  const toggleVideo = () => {
    if (localStreamRef.current) {
      const videoTrack = localStreamRef.current.getVideoTracks()[0];
      if (videoTrack) {
        videoTrack.enabled = !videoTrack.enabled;
        setIsVideoMuted(!videoTrack.enabled);
      }
    }
  };

  const startRecording = () => {
    if (!localStreamRef.current) return;
    
    recordedChunksRef.current = [];
    
    try {
      // Use webm for browser recording. Whisper handles webm correctly in short mode.
      const mimeType = MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : 'audio/mp4';
      const mediaRecorder = new MediaRecorder(localStreamRef.current, { mimeType });
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          recordedChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const blob = new Blob(recordedChunksRef.current, { type: mimeType });
        await uploadRecordingToASR(blob);
      };

      mediaRecorder.start();
      mediaRecorderRef.current = mediaRecorder;
      setIsRecording(true);
      setStatus("Recording started...");
    } catch (e) {
      console.error("Recording error:", e);
      setStatus("Failed to start recording.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setStatus("Recording stopped, processing ASR...");
    }
  };

  const uploadRecordingToASR = async (blob: Blob) => {
    try {
      const formData = new FormData();
      // Send as webm, short mode usually passes directly to whisper which handles webm automatically
      formData.append("file", blob, "meeting_recording.webm");
      formData.append("mode", "short");
      formData.append("model", "large-v3");

      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";
      const resp = await fetch(`${apiUrl}/audio/asr`, {
        method: "POST",
        body: formData,
      });

      if (!resp.ok) {
        setStatus("ASR Error: " + resp.statusText);
        return;
      }

      const data = await resp.json();
      setAsrResult(data.text || "");
      setStatus("ASR complete.");
    } catch (e) {
      console.error(e);
      setStatus("Failed to transcribe recording.");
    }
  };

  useEffect(() => {
    return () => {
      leaveCall();
    };
  }, []);

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-blue-600 to-indigo-600 rounded-2xl p-6 text-white shadow-lg">
        <h1 className="text-2xl font-bold mb-2">Video Meeting Room</h1>
        <p className="text-indigo-100">Local P2P WebRTC Call with Integrated AI Transcription</p>
      </div>

      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
        <div className="flex flex-col md:flex-row gap-4 justify-between mb-6">
          <div className="flex items-center gap-4">
            <input
              type="text"
              value={roomId}
              onChange={(e) => setRoomId(e.target.value)}
              disabled={isInCall}
              placeholder="Enter Room ID"
              className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 outline-none"
            />
            {!isInCall ? (
              <button
                onClick={joinCall}
                className="flex items-center gap-2 px-5 py-2.5 rounded-lg font-semibold text-white bg-green-600 hover:bg-green-700 transition-colors"
              >
                <PhoneCall className="w-5 h-5" />
                Join Call
              </button>
            ) : (
              <button
                onClick={leaveCall}
                className="flex items-center gap-2 px-5 py-2.5 rounded-lg font-semibold text-white bg-red-600 hover:bg-red-700 transition-colors"
              >
                <PhoneOff className="w-5 h-5" />
                Leave Call
              </button>
            )}
          </div>
          
          <div className="flex items-center gap-2 text-sm text-gray-600 font-medium">
            <span className={clsx("w-3 h-3 rounded-full", isInCall ? "bg-green-500" : "bg-gray-300")} />
            {status || (isInCall ? "Connected" : "Disconnected")}
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="relative bg-gray-900 rounded-xl overflow-hidden aspect-video shadow-inner">
            <video
              ref={localVideoRef}
              autoPlay
              playsInline
              muted
              className={clsx("w-full h-full object-cover", isVideoMuted && "opacity-0")}
            />
            <div className="absolute bottom-4 left-4 bg-black/50 px-3 py-1 rounded-md text-white text-sm backdrop-blur-sm">
              You {isAudioMuted && "(Muted)"}
            </div>
            {isVideoMuted && (
              <div className="absolute inset-0 flex items-center justify-center text-white flex-col">
                <VideoOff className="w-12 h-12 mb-2 opacity-50" />
                <span className="text-gray-400">Camera Off</span>
              </div>
            )}
          </div>
          
          <div className="relative bg-gray-900 rounded-xl overflow-hidden aspect-video shadow-inner">
            <video
              ref={remoteVideoRef}
              autoPlay
              playsInline
              className="w-full h-full object-cover"
            />
            <div className="absolute bottom-4 left-4 bg-black/50 px-3 py-1 rounded-md text-white text-sm backdrop-blur-sm">
              Remote Peer
            </div>
            {!isInCall && (
              <div className="absolute inset-0 flex items-center justify-center text-white flex-col">
                <span className="text-gray-400">Waiting for peer...</span>
              </div>
            )}
          </div>
        </div>

        {isInCall && (
          <div className="flex flex-wrap justify-center gap-4 mt-6 p-4 bg-gray-50 rounded-xl border border-gray-200">
            <button
              onClick={toggleAudio}
              className={clsx(
                "p-4 rounded-full transition-colors",
                isAudioMuted ? "bg-red-100 text-red-600 hover:bg-red-200" : "bg-white border shadow-sm text-gray-700 hover:bg-gray-50"
              )}
              title={isAudioMuted ? "Unmute Microphone" : "Mute Microphone"}
            >
              {isAudioMuted ? <MicOff className="w-6 h-6" /> : <Mic className="w-6 h-6" />}
            </button>
            <button
              onClick={toggleVideo}
              className={clsx(
                "p-4 rounded-full transition-colors",
                isVideoMuted ? "bg-red-100 text-red-600 hover:bg-red-200" : "bg-white border shadow-sm text-gray-700 hover:bg-gray-50"
              )}
              title={isVideoMuted ? "Turn on Camera" : "Turn off Camera"}
            >
              {isVideoMuted ? <VideoOff className="w-6 h-6" /> : <VideoIcon className="w-6 h-6" />}
            </button>
            
            <div className="w-px bg-gray-300 mx-2" />

            {!isRecording ? (
              <button
                onClick={startRecording}
                className="flex items-center gap-2 px-6 py-3 rounded-full font-semibold text-red-600 bg-red-50 hover:bg-red-100 border border-red-200 transition-colors"
              >
                <Circle className="w-5 h-5 fill-current" />
                Record Meeting
              </button>
            ) : (
              <button
                onClick={stopRecording}
                className="flex items-center gap-2 px-6 py-3 rounded-full font-semibold text-white bg-red-600 hover:bg-red-700 shadow-md animate-pulse transition-colors"
              >
                <StopCircle className="w-5 h-5" />
                Stop & Transcribe
              </button>
            )}
          </div>
        )}

        {asrResult && (
          <div className="mt-8 bg-indigo-50 border border-indigo-100 rounded-xl p-6">
            <div className="flex items-center gap-2 mb-4 text-indigo-800">
              <DownloadCloud className="w-5 h-5" />
              <h3 className="font-semibold text-lg">Meeting Transcription</h3>
            </div>
            <p className="text-gray-800 whitespace-pre-wrap leading-relaxed">
              {asrResult}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}