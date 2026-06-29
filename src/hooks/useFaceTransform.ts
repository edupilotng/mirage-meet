import { useRef, useState, useCallback, useEffect } from 'react';
import type { TransformationSettings, BackgroundOption } from '../types';

interface UseFaceTransformReturn {
  processedStream: MediaStream | null;
  transformationSettings: TransformationSettings;
  setTransformationSettings: React.Dispatch<React.SetStateAction<TransformationSettings>>;
  referenceImage: HTMLImageElement | null;
  setReferenceImage: (img: HTMLImageElement | null) => void;
  backgroundOptions: BackgroundOption[];
  isProcessing: boolean;
  statusMessage: string;
  modelLoadProgress: number;
  initializeTransform: (stream: MediaStream) => Promise<void>;
  updateBackground: (backgroundId: string) => void;
  cleanup: () => void;
}

export const backgroundOptions: BackgroundOption[] = [
  { id: 'none', name: 'None', thumbnail: '', value: '' },
  {
    id: 'office',
    name: 'Modern Office',
    thumbnail: 'https://images.pexels.com/photos/667838/pexels-photo-667838.jpeg?auto=compress&cs=tinysrgb&w=200',
    value: 'https://images.pexels.com/photos/667838/pexels-photo-667838.jpeg?auto=compress&cs=tinysrgb&w=1280',
  },
  {
    id: 'luxury',
    name: 'Luxury Office',
    thumbnail: 'https://images.pexels.com/photos/1743555/pexels-photo-1743555.jpeg?auto=compress&cs=tinysrgb&w=200',
    value: 'https://images.pexels.com/photos/1743555/pexels-photo-1743555.jpeg?auto=compress&cs=tinysrgb&w=1280',
  },
  {
    id: 'studio',
    name: 'Studio',
    thumbnail: 'https://images.pexels.com/photos/1571463/pexels-photo-1571463.jpeg?auto=compress&cs=tinysrgb&w=200',
    value: 'https://images.pexels.com/photos/1571463/pexels-photo-1571463.jpeg?auto=compress&cs=tinysrgb&w=1280',
  },
  {
    id: 'conference',
    name: 'Conference Room',
    thumbnail: 'https://images.pexels.com/photos/416320/pexels-photo-416320.jpeg?auto=compress&cs=tinysrgb&w=200',
    value: 'https://images.pexels.com/photos/416320/pexels-photo-416320.jpeg?auto=compress&cs=tinysrgb&w=1280',
  },
  {
    id: 'apartment',
    name: 'Modern Apartment',
    thumbnail: 'https://images.pexels.com/photos/1571460/pexels-photo-1571460.jpeg?auto=compress&cs=tinysrgb&w=200',
    value: 'https://images.pexels.com/photos/1571460/pexels-photo-1571460.jpeg?auto=compress&cs=tinysrgb&w=1280',
  },
];

const SERVER = 'http://localhost:3001';
// How often to send frames to the AI server (ms).
// CPU inference on inswapper takes ~200-500ms per frame, so 3fps balances quality vs lag.
const TRANSFORM_INTERVAL_MS = 333;

export function useFaceTransform(): UseFaceTransformReturn {
  const [processedStream, setProcessedStream] = useState<MediaStream | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [statusMessage, setStatusMessage] = useState('Camera Ready');
  const [modelLoadProgress, setModelLoadProgress] = useState(0);
  const [transformationSettings, setTransformationSettings] = useState<TransformationSettings>({
    enabled: false,
    referenceImage: null,
    background: '',
  });
  const [referenceImage, setReferenceImage] = useState<HTMLImageElement | null>(null);

  const outputCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const hostVideoRef = useRef<HTMLVideoElement | null>(null);
  const animFrameRef = useRef<number | null>(null);
  const bgImgRef = useRef<HTMLImageElement | null>(null);
  const swappedFrameRef = useRef<HTMLImageElement | null>(null);

  const selfieSegRef = useRef<any>(null);
  const segResultRef = useRef<any>(null);
  const segBusyRef = useRef(false);
  const segBusyAtRef = useRef(0);

  const frameRef = useRef(0);
  const settingsRef = useRef(transformationSettings);
  const refImageRef = useRef<HTMLImageElement | null>(null);
  const currentBgRef = useRef('');
  const statusCacheRef = useRef('');
  const initDoneRef = useRef(false);

  // AI server state
  const aiReadyRef = useRef(false);
  const faceRegisteredRef = useRef(false);
  const transformBusyRef = useRef(false);
  const lastTransformRef = useRef(0);
  const statusPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const setStatus = useCallback((msg: string) => {
    if (statusCacheRef.current !== msg) {
      statusCacheRef.current = msg;
      setStatusMessage(msg);
    }
  }, []);

  useEffect(() => { settingsRef.current = transformationSettings; }, [transformationSettings]);
  useEffect(() => { refImageRef.current = referenceImage; }, [referenceImage]);

  const loadScript = useCallback((id: string, src: string): Promise<void> =>
    new Promise((res, rej) => {
      if (document.getElementById(id)) { res(); return; }
      const s = Object.assign(document.createElement('script'), { id, src, crossOrigin: 'anonymous' });
      s.onload = () => res();
      s.onerror = () => rej(new Error(`Script failed: ${src}`));
      document.head.appendChild(s);
    }), []);

  // Poll AI server status until ready
  const pollAIStatus = useCallback(() => {
    if (statusPollRef.current) clearInterval(statusPollRef.current);

    statusPollRef.current = setInterval(async () => {
      try {
        const res = await fetch(`${SERVER}/ai/status`);
        if (!res.ok) return;
        const data = await res.json();

        if (data.state === 'ready') {
          aiReadyRef.current = true;
          clearInterval(statusPollRef.current!);
          statusPollRef.current = null;
          setModelLoadProgress(100);
          setStatus('AI Engine Ready');
          console.log('[AI] Server-side AI engine is ready');

          // If user already uploaded a reference image, register it now
          if (refImageRef.current && !faceRegisteredRef.current) {
            registerFaceWithServer(refImageRef.current);
          }
        } else if (data.state === 'downloading') {
          const progValues = Object.values(data.progress || {}) as number[];
          const avg = progValues.length > 0 ? progValues.reduce((a, b) => a + b, 0) / progValues.length : 0;
          setModelLoadProgress(Math.round(avg * 0.8)); // downloading = 0-80%
          setStatus(`Downloading AI models... ${data.progress ? Object.entries(data.progress).map(([k, v]) => `${k}:${v}%`).join(' ') : ''}`);
        } else if (data.state === 'loading') {
          setModelLoadProgress(85);
          setStatus('Loading AI models into memory...');
        } else if (data.state === 'error') {
          clearInterval(statusPollRef.current!);
          statusPollRef.current = null;
          setModelLoadProgress(0);
          setStatus(`AI Error: ${data.error}`);
          console.error('[AI] Server error:', data.error);
        } else {
          // idle — server just started
          setStatus('Waiting for AI server...');
        }
      } catch {
        // Server not yet up, keep polling
      }
    }, 2000);
  }, [setStatus]);

  // Register reference face image with the AI server
  const registerFaceWithServer = useCallback(async (img: HTMLImageElement) => {
    if (faceRegisteredRef.current) return;
    if (!aiReadyRef.current) {
      setStatus('AI models still loading, queued...');
      return;
    }

    setStatus('Detecting reference face...');
    faceRegisteredRef.current = true;

    try {
      // Convert HTMLImageElement to blob
      const canvas = document.createElement('canvas');
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext('2d')!;
      ctx.drawImage(img, 0, 0);

      const blob: Blob = await new Promise(res => canvas.toBlob(b => res(b!), 'image/jpeg', 0.95));
      const form = new FormData();
      form.append('image', blob, 'reference.jpg');

      const res = await fetch(`${SERVER}/ai/register-face`, { method: 'POST', body: form });
      const data = await res.json();

      if (!res.ok) {
        faceRegisteredRef.current = false;
        setStatus(`Face detection failed: ${data.error}`);
        return;
      }

      console.log('[AI] Reference face registered:', data);
      setStatus('Reference face registered');
    } catch (err: any) {
      faceRegisteredRef.current = false;
      setStatus(`Registration error: ${err.message}`);
    }
  }, [setStatus]);

  // When referenceImage changes, register it
  useEffect(() => {
    if (referenceImage) {
      faceRegisteredRef.current = false; // reset so it re-registers
      if (aiReadyRef.current) {
        registerFaceWithServer(referenceImage);
      }
    } else {
      faceRegisteredRef.current = false;
      fetch(`${SERVER}/ai/clear-face`, { method: 'POST' }).catch(() => {});
    }
  }, [referenceImage, registerFaceWithServer]);

  // Send a frame to the AI server for transformation
  const sendFrameForTransform = useCallback(async (videoEl: HTMLVideoElement) => {
    if (transformBusyRef.current) return;
    if (!aiReadyRef.current || !faceRegisteredRef.current) return;

    const now = Date.now();
    if (now - lastTransformRef.current < TRANSFORM_INTERVAL_MS) return;
    lastTransformRef.current = now;
    transformBusyRef.current = true;

    try {
      // Capture current video frame at reduced resolution for speed
      // 480p is sufficient for face swap while keeping processing fast
      const W = Math.min(480, videoEl.videoWidth);
      const H = Math.round(W * videoEl.videoHeight / videoEl.videoWidth);
      const cap = document.createElement('canvas');
      cap.width = W; cap.height = H;
      cap.getContext('2d')!.drawImage(videoEl, 0, 0, W, H);

      const blob: Blob = await new Promise(res => cap.toBlob(b => res(b!), 'image/jpeg', 0.85));
      console.log('[Transform] Sending frame to AI server, size:', blob.size, 'bytes, dims:', W, 'x', H);
      const form = new FormData();
      form.append('frame', blob, 'frame.jpg');

      const res = await fetch(`${SERVER}/ai/transform-frame`, { method: 'POST', body: form });

      if (res.status === 204) {
        // No face detected in this frame — keep last good transformed frame, do NOT clear it
        console.log('[Transform] No face in frame (204), keeping last good frame');
        transformBusyRef.current = false;
        return;
      }

      if (!res.ok) {
        console.warn('[Transform] Server error:', res.status);
        transformBusyRef.current = false;
        return;
      }

      const jpegBlob = await res.blob();
      console.log('[Transform] Received transformed frame, size:', jpegBlob.size, 'bytes');
      const url = URL.createObjectURL(jpegBlob);

      const img = new Image();
      img.onload = () => {
        console.log('[Transform] Transformed frame loaded, dims:', img.naturalWidth, 'x', img.naturalHeight);
        // Revoke old URL only after new img is confirmed loaded
        const prev = swappedFrameRef.current;
        swappedFrameRef.current = img;
        if (prev?.src?.startsWith('blob:')) {
          URL.revokeObjectURL(prev.src);
        }
        transformBusyRef.current = false;
      };
      img.onerror = () => {
        console.warn('[Transform] Failed to load transformed frame img');
        URL.revokeObjectURL(url);
        transformBusyRef.current = false;
      };
      img.src = url;
    } catch (err: any) {
      console.warn('[Transform] sendFrameForTransform error:', err.message);
      transformBusyRef.current = false;
    }
  }, []);

  // Selfie segmentation for background replacement
  const initSelfie = useCallback(async () => {
    try {
      await loadScript('mp-selfie', 'https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/selfie_segmentation.js');
      const SS = (window as any).SelfieSegmentation;
      if (!SS) return;
      const seg = new SS({ locateFile: (f: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/${f}` });
      seg.setOptions({ modelSelection: 1, selfieMode: false });
      seg.onResults((r: any) => { segResultRef.current = r; segBusyRef.current = false; });
      selfieSegRef.current = seg;
      console.log('[AI] Selfie segmentation initialized');
    } catch (err) {
      console.warn('[AI] Selfie segmentation failed:', err);
    }
  }, [loadScript]);

  const startRenderLoop = useCallback(() => {
    const tick = () => {
      const vid = hostVideoRef.current;
      const out = outputCanvasRef.current;
      const s = settingsRef.current;

      frameRef.current++;
      const frame = frameRef.current;

      if (vid && out && vid.readyState >= 2) {
        const W = out.width, H = out.height;

        // Selfie segmentation every 2 frames; reset stuck-busy after 1.5s
        if (segBusyRef.current && Date.now() - segBusyAtRef.current > 1500) {
          segBusyRef.current = false;
        }
        if (!segBusyRef.current && selfieSegRef.current && frame % 2 === 0) {
          segBusyRef.current = true;
          segBusyAtRef.current = Date.now();
          selfieSegRef.current.send({ image: vid }).catch(() => { segBusyRef.current = false; });
        }

        // Send to AI server for transformation
        if (s.enabled && faceRegisteredRef.current) {
          sendFrameForTransform(vid);
        }

        renderFrame(W, H, s);
      }

      animFrameRef.current = requestAnimationFrame(tick);
    };
    animFrameRef.current = requestAnimationFrame(tick);
  }, [sendFrameForTransform]);

  const renderFrame = useCallback((W: number, H: number, s: TransformationSettings) => {
    const out = outputCanvasRef.current;
    const vid = hostVideoRef.current;
    const seg = segResultRef.current;
    const bgImg = bgImgRef.current;
    const bgVal = currentBgRef.current;
    const swapped = swappedFrameRef.current;

    if (!out || !vid || vid.readyState < 2) return;
    const ctx = out.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, W, H);

    // 1. Background layer
    if (bgVal && bgImg?.complete && bgImg.naturalWidth > 0) {
      ctx.drawImage(bgImg, 0, 0, W, H);
    } else {
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, W, H);
    }

    // 2. Person layer (transformed or original)
    const hasValidSwap = s.enabled && swapped != null && swapped.complete && swapped.naturalWidth > 0;
    const sourceFrame = hasValidSwap ? swapped! : vid;
    if (frameRef.current % 60 === 0 && hasValidSwap) console.log('[DEBUG] Using transformed frame as output');

    if (seg?.segmentationMask && bgVal) {
      const personOff = new OffscreenCanvas(W, H);
      const pCtx = personOff.getContext('2d') as OffscreenCanvasRenderingContext2D;
      pCtx.drawImage(sourceFrame, 0, 0, W, H);
      pCtx.globalCompositeOperation = 'destination-in';
      pCtx.drawImage(seg.segmentationMask, 0, 0, W, H);
      pCtx.globalCompositeOperation = 'source-over';
      ctx.drawImage(personOff, 0, 0);
    } else {
      ctx.drawImage(sourceFrame, 0, 0, W, H);
    }

    // Status — use same valid-swap check as sourceFrame selection
    if (hasValidSwap) {
      setStatus('AI Transformation Active');
    } else if (s.enabled && faceRegisteredRef.current) {
      setStatus('Processing frames...');
    } else if (s.enabled && !faceRegisteredRef.current) {
      setStatus('Waiting for reference face...');
    } else if (bgVal && bgImg?.complete) {
      setStatus('Background Active');
    } else {
      setStatus('Camera Ready');
    }
  }, [setStatus]);

  const initializeTransform = useCallback(async (stream: MediaStream) => {
    if (initDoneRef.current) return;
    initDoneRef.current = true;
    setIsProcessing(true);
    setStatus('Starting camera...');

    const vid = document.createElement('video');
    vid.srcObject = stream;
    vid.playsInline = true;
    vid.muted = true;

    try {
      await vid.play();
    } catch {
      setStatus('Camera Error');
      setIsProcessing(false);
      initDoneRef.current = false;
      return;
    }

    hostVideoRef.current = vid;

    const out = document.createElement('canvas');
    out.width = 1280;
    out.height = 720;
    outputCanvasRef.current = out;

    const outStream = out.captureStream(30);
    stream.getAudioTracks().forEach(t => outStream.addTrack(t));
    setProcessedStream(outStream);

    // Start selfie segmentation (non-blocking)
    initSelfie();

    // Start polling AI server status
    pollAIStatus();

    startRenderLoop();
    setIsProcessing(false);
  }, [initSelfie, pollAIStatus, startRenderLoop]);

  const updateBackground = useCallback((backgroundId: string) => {
    const opt = backgroundOptions.find(o => o.id === backgroundId);
    const bgVal = opt?.value ?? '';
    currentBgRef.current = bgVal;
    setTransformationSettings(prev => ({ ...prev, background: bgVal }));

    if (!bgVal) { bgImgRef.current = null; setStatus('Camera Ready'); return; }

    setStatus('Loading background...');
    const img = document.createElement('img');
    img.crossOrigin = 'anonymous';
    img.onload = () => { bgImgRef.current = img; setStatus('Background Active'); };
    img.onerror = () => { bgImgRef.current = null; setStatus('Background load failed'); };
    img.src = bgVal;
  }, [setStatus]);

  const cleanup = useCallback(() => {
    if (animFrameRef.current) { cancelAnimationFrame(animFrameRef.current); animFrameRef.current = null; }
    if (statusPollRef.current) { clearInterval(statusPollRef.current); statusPollRef.current = null; }
    try { selfieSegRef.current?.close(); } catch { /* noop */ }
    selfieSegRef.current = null;
    if (hostVideoRef.current) { hostVideoRef.current.pause(); hostVideoRef.current.srcObject = null; hostVideoRef.current = null; }
    if (swappedFrameRef.current?.src?.startsWith('blob:')) URL.revokeObjectURL(swappedFrameRef.current.src);
    swappedFrameRef.current = null;
    bgImgRef.current = null;
    outputCanvasRef.current = null;
    segResultRef.current = null;
    currentBgRef.current = '';
    frameRef.current = 0;
    segBusyRef.current = false;
    transformBusyRef.current = false;
    aiReadyRef.current = false;
    faceRegisteredRef.current = false;
    initDoneRef.current = false;
    setProcessedStream(null);
    setIsProcessing(false);
    setStatusMessage('Camera Ready');
    setModelLoadProgress(0);
  }, []);

  useEffect(() => () => { cleanup(); }, [cleanup]);

  return {
    processedStream,
    transformationSettings,
    setTransformationSettings,
    referenceImage,
    setReferenceImage,
    backgroundOptions,
    isProcessing,
    statusMessage,
    modelLoadProgress,
    initializeTransform,
    updateBackground,
    cleanup,
  };
}
