import { useRef, useState, useCallback, useEffect } from 'react';
import type { TransformationSettings, BackgroundOption } from '../types';

interface UseFaceTransformReturn {
  processedStream: MediaStream | null;
  transformationSettings: TransformationSettings;
  setTransformationSettings: React.Dispatch<React.SetStateAction<TransformationSettings>>;
  referenceVideo: HTMLVideoElement | null;
  setReferenceVideo: (video: HTMLVideoElement | null) => void;
  backgroundOptions: BackgroundOption[];
  isProcessing: boolean;
  statusMessage: string;
  initializeTransform: (stream: MediaStream) => Promise<void>;
  updateBackground: (backgroundId: string) => void;
  cleanup: () => void;
}

export const backgroundOptions: BackgroundOption[] = [
  { id: 'none', name: 'None', thumbnail: '', value: '' },
  { id: 'office', name: 'Modern Office', thumbnail: 'https://images.pexels.com/photos/1170412/pexels-photo-1170412.jpeg?auto=compress&cs=tinysrgb&w=200', value: 'https://images.pexels.com/photos/1170412/pexels-photo-1170412.jpeg?auto=compress&cs=tinysrgb&w=1280' },
  { id: 'luxury', name: 'Luxury Office', thumbnail: 'https://images.pexels.com/photos/3801740/pexels-photo-3801740.jpeg?auto=compress&cs=tinysrgb&w=200', value: 'https://images.pexels.com/photos/3801740/pexels-photo-3801740.jpeg?auto=compress&cs=tinysrgb&w=1280' },
  { id: 'studio', name: 'Studio', thumbnail: 'https://images.pexels.com/photos/1595385/pexels-photo-1595385.jpeg?auto=compress&cs=tinysrgb&w=200', value: 'https://images.pexels.com/photos/1595385/pexels-photo-1595385.jpeg?auto=compress&cs=tinysrgb&w=1280' },
  { id: 'conference', name: 'Conference Room', thumbnail: 'https://images.pexels.com/photos/159711/books-coffee-students-room-159711.jpeg?auto=compress&cs=tinysrgb&w=200', value: 'https://images.pexels.com/photos/159711/books-coffee-students-room-159711.jpeg?auto=compress&cs=tinysrgb&w=1280' },
  { id: 'apartment', name: 'Modern Apartment', thumbnail: 'https://images.pexels.com/photos/1571460/pexels-photo-1571460.jpeg?auto=compress&cs=tinysrgb&w=200', value: 'https://images.pexels.com/photos/1571460/pexels-photo-1571460.jpeg?auto=compress&cs=tinysrgb&w=1280' },
];

declare global {
  interface Window {
    FaceMesh: any;
    SelfieSegmentation: any;
  }
}

export function useFaceTransform(): UseFaceTransformReturn {
  const [processedStream, setProcessedStream] = useState<MediaStream | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [statusMessage, setStatusMessage] = useState('Camera Ready');
  const [transformationSettings, setTransformationSettings] = useState<TransformationSettings>({
    enabled: false,
    referenceVideo: null,
    background: '',
  });
  const [referenceVideo, setReferenceVideo] = useState<HTMLVideoElement | null>(null);

  const outputCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const inputCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const refFrameCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const bgCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const backgroundImgRef = useRef<HTMLImageElement | null>(null);
  const inputStreamRef = useRef<MediaStream | null>(null);
  const selfieSegmentationRef = useRef<any>(null);
  const faceMeshRef = useRef<any>(null);
  const isInitializedRef = useRef(false);
  const isProcessingRef = useRef(false);

  const currentBackgroundRef = useRef<string>('');
  const settingsRef = useRef(transformationSettings);
  const referenceFrameIdxRef = useRef<number>(0);
  const refVideoReadyRef = useRef<boolean>(false);

  useEffect(() => {
    settingsRef.current = transformationSettings;
  }, [transformationSettings]);

  const loadScript = useCallback((id: string, src: string): Promise<void> => {
    return new Promise((resolve, reject) => {
      const existing = document.getElementById(id);
      if (existing) {
        resolve();
        return;
      }
      const script = document.createElement('script');
      script.id = id;
      script.src = src;
      script.crossOrigin = 'anonymous';
      script.onload = () => resolve();
      script.onerror = () => reject(new Error(`Failed to load ${src}`));
      document.head.appendChild(script);
    });
  }, []);

  const loadMediaPipeScripts = useCallback(async () => {
    setStatusMessage('Loading AI models...');

    await loadScript('mediapipe-selfie', 'https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/selfie_segmentation.js');
    await loadScript('mediapipe-face-mesh', 'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619/face_mesh.js');

    isInitializedRef.current = true;
  }, [loadScript]);

  const initializeFaceMesh = useCallback(async () => {
    if (faceMeshRef.current) return;

    if (!window.FaceMesh) {
      throw new Error('FaceMesh not loaded');
    }

    const faceMesh = new window.FaceMesh({
      locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619/${file}`,
    });

    faceMesh.setOptions({
      maxNumFaces: 1,
      refineLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    faceMeshRef.current = faceMesh;
  }, []);

  const initializeSelfieSegmentation = useCallback(async () => {
    if (selfieSegmentationRef.current) return selfieSegmentationRef.current;

    if (!window.SelfieSegmentation) {
      throw new Error('SelfieSegmentation not loaded');
    }

    const selfieSegmentation = new window.SelfieSegmentation({
      locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/${file}`,
    });

    selfieSegmentation.setOptions({
      modelSelection: 1,
      selfieMode: false,
    });

    selfieSegmentationRef.current = selfieSegmentation;
    return selfieSegmentation;
  }, []);

  const initializeTransform = useCallback(async (stream: MediaStream) => {
    if (isProcessingRef.current) {
      setStatusMessage('Already initializing...');
      return;
    }
    isProcessingRef.current = true;
    inputStreamRef.current = stream;
    setIsProcessing(true);
    setStatusMessage('Starting camera...');

    const video = document.createElement('video');
    video.srcObject = stream;
    video.playsInline = true;
    video.muted = true;

    try {
      await video.play();
      videoRef.current = video;
      setStatusMessage('Camera Ready');
    } catch (err) {
      console.error('Video play error:', err);
      setStatusMessage('Camera Error');
      isProcessingRef.current = false;
      return;
    }

    const outputCanvas = document.createElement('canvas');
    outputCanvas.width = 1280;
    outputCanvas.height = 720;
    outputCanvasRef.current = outputCanvas;

    const inputCanvas = document.createElement('canvas');
    inputCanvas.width = 1280;
    inputCanvas.height = 720;
    inputCanvasRef.current = inputCanvas;

    const bgCanvas = document.createElement('canvas');
    bgCanvas.width = 1280;
    bgCanvas.height = 720;
    bgCanvasRef.current = bgCanvas;

    const refFrameCanvas = document.createElement('canvas');
    refFrameCanvas.width = 640;
    refFrameCanvas.height = 360;
    refFrameCanvasRef.current = refFrameCanvas;

    const outputStream = outputCanvas.captureStream(30);
    stream.getAudioTracks().forEach(track => outputStream.addTrack(track));
    setProcessedStream(outputStream);

    try {
      await loadMediaPipeScripts();
      await initializeSelfieSegmentation();
      setStatusMessage('AI models loaded');
    } catch (error) {
      console.error('Error loading MediaPipe:', error);
      setStatusMessage('AI model load failed');
      isProcessingRef.current = false;
      return;
    }

    startProcessingLoop();
    isProcessingRef.current = false;
  }, [loadMediaPipeScripts, initializeSelfieSegmentation]);

  const startProcessingLoop = useCallback(() => {
    const selfieSegmentation = selfieSegmentationRef.current;
    const video = videoRef.current;
    const outputCanvas = outputCanvasRef.current;
    const inputCanvas = inputCanvasRef.current;
    const refFrameCanvas = refFrameCanvasRef.current;

    if (!selfieSegmentation || !video || !outputCanvas || !inputCanvas) {
      animationFrameRef.current = requestAnimationFrame(startProcessingLoop);
      return;
    }

    const outputCtx = outputCanvas.getContext('2d', { willReadFrequently: true });
    const inputCtx = inputCanvas.getContext('2d', { willReadFrequently: true });

    if (!outputCtx || !inputCtx) {
      animationFrameRef.current = requestAnimationFrame(startProcessingLoop);
      return;
    }

    let lastProcessTime = 0;
    const targetFPS = 30;
    const frameInterval = 1000 / targetFPS;

    const processFrame = async (timestamp: number) => {
      if (!videoRef.current || !outputCanvasRef.current || !inputCanvasRef.current) {
        animationFrameRef.current = requestAnimationFrame(processFrame);
        return;
      }

      const elapsed = timestamp - lastProcessTime;

      if (elapsed >= frameInterval && videoRef.current.readyState >= 2) {
        lastProcessTime = timestamp - (elapsed % frameInterval);

        const inCtx = inputCanvasRef.current.getContext('2d');
        if (inCtx) {
          inCtx.drawImage(videoRef.current, 0, 0, inputCanvasRef.current.width, inputCanvasRef.current.height);
        }

        const settings = settingsRef.current;
        const refVideo = referenceVideo;
        const bgValue = currentBackgroundRef.current;

        try {
          await selfieSegmentationRef.current.send({ image: videoRef.current });
        } catch (e) {
          const outCtx = outputCanvasRef.current.getContext('2d');
          if (outCtx && videoRef.current) {
            outCtx.drawImage(videoRef.current, 0, 0, outputCanvasRef.current.width, outputCanvasRef.current.height);
          }
        }
      }

      animationFrameRef.current = requestAnimationFrame(processFrame);
    };

    selfieSegmentation.onResults((results: any) => {
      const outputCanvas = outputCanvasRef.current;
      const inputCanvas = inputCanvasRef.current;
      const bgCanvas = bgCanvasRef.current;
      const video = videoRef.current;

      if (!outputCanvas || !inputCanvas || !video) return;

      const outCtx = outputCanvas.getContext('2d', { willReadFrequently: true });
      const bgCtx = bgCanvas?.getContext('2d');

      if (!outCtx) return;

      outCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);

      const settings = settingsRef.current;
      const refVideo = referenceVideo;
      const bgValue = currentBackgroundRef.current;

      const shouldTransform = settings.enabled && refVideo && refVideo.readyState >= 2;

      if (shouldTransform && refFrameCanvasRef.current) {
        const refCtx = refFrameCanvasRef.current.getContext('2d', { willReadFrequently: true });
        if (refCtx && refVideo) {
          refCtx.drawImage(refVideo, 0, 0, refFrameCanvasRef.current.width, refFrameCanvasRef.current.height);
        }
      }

      if (bgValue && backgroundImgRef.current && backgroundImgRef.current.complete) {
        outCtx.drawImage(results.image, 0, 0, outputCanvas.width, outputCanvas.height);

        if (results.segmentationMask) {
          outCtx.globalCompositeOperation = 'destination-out';
          outCtx.drawImage(results.segmentationMask, 0, 0, outputCanvas.width, outputCanvas.height);
          outCtx.globalCompositeOperation = 'destination-over';
          outCtx.drawImage(backgroundImgRef.current, 0, 0, outputCanvas.width, outputCanvas.height);
          outCtx.globalCompositeOperation = 'source-over';
        }

        if (shouldTransform && refFrameCanvasRef.current) {
          applyFaceTransformation(outCtx, outputCanvas, refFrameCanvasRef.current, results);
        }
      } else {
        outCtx.drawImage(results.image, 0, 0, outputCanvas.width, outputCanvas.height);
      }

      if (currentBackgroundRef.current === '' && !settings.enabled) {
        setStatusMessage('Camera Ready');
      } else if (currentBackgroundRef.current && backgroundImgRef.current?.complete) {
        setStatusMessage('Background Active');
      }

      if (shouldTransform) {
        setStatusMessage('Transformation Active');
      }
    });

    animationFrameRef.current = requestAnimationFrame(processFrame);
  }, [referenceVideo]);

  const applyFaceTransformation = (
    outputCtx: CanvasRenderingContext2D,
    outputCanvas: HTMLCanvasElement,
    refFrameCanvas: HTMLCanvasElement,
    results: any
  ) => {
    const refCtx = refFrameCanvas.getContext('2d', { willReadFrequently: true } as any);
    if (!refCtx) return;

    const width = outputCanvas.width;
    const height = outputCanvas.height;

    const outputData = outputCtx.getImageData(0, 0, width, height);
    const outputPixels = outputData.data;

    const refData = refCtx.getImageData(0, 0, refFrameCanvas.width, refFrameCanvas.height);
    const refPixels = refData.data;

    const scaleX = refFrameCanvas.width / width;
    const scaleY = refFrameCanvas.height / height;

    const sampleStep = 4;
    const faceRegionCenterX = width * 0.5;
    const faceRegionCenterY = height * 0.4;
    const faceRegionWidth = width * 0.35;
    const faceRegionHeight = height * 0.4;

    for (let y = 0; y < height; y += sampleStep) {
      for (let x = 0; x < width; x += sampleStep) {
        const dx = x - faceRegionCenterX;
        const dy = y - faceRegionCenterY;

        const normalizedDistX = dx / (faceRegionWidth / 2);
        const normalizedDistY = dy / (faceRegionHeight / 2);
        const distance = Math.sqrt(normalizedDistX * normalizedDistX + normalizedDistY * normalizedDistY);

        if (distance < 1.3) {
          const blendFactor = Math.pow(Math.max(0, 1 - distance / 1.3), 1.2);

          const refX = Math.floor(x * scaleX);
          const refY = Math.floor(y * scaleY);

          if (refX >= 0 && refX < refFrameCanvas.width && refY >= 0 && refY < refFrameCanvas.height) {
            const srcIdx = (y * width + x) * 4;
            const refIdx = (refY * refFrameCanvas.width + refX) * 4;

            const srcR = outputPixels[srcIdx] || 0;
            const srcG = outputPixels[srcIdx + 1] || 0;
            const srcB = outputPixels[srcIdx + 2] || 0;

            const refR = refPixels[refIdx] || 0;
            const refG = refPixels[refIdx + 1] || 0;
            const refB = refPixels[refIdx + 2] || 0;

            const brightnessSrc = (srcR + srcG + srcB) / 3;
            const brightnessRef = (refR + refG + refB) / 3;
            const brightnessRatio = brightnessSrc > 0 ? brightnessRef / brightnessSrc : 1;

            let adjustedR = refR;
            let adjustedG = refG;
            let adjustedB = refB;

            if (brightnessRatio > 0.3 && brightnessRatio < 3) {
              adjustedR = Math.min(255, Math.max(0, refR * (0.85 + 0.15 * brightnessRatio)));
              adjustedG = Math.min(255, Math.max(0, refG * (0.85 + 0.15 * brightnessRatio)));
              adjustedB = Math.min(255, Math.max(0, refB * (0.85 + 0.15 * brightnessRatio)));
            }

            outputPixels[srcIdx] = Math.round(srcR * (1 - blendFactor) + adjustedR * blendFactor);
            outputPixels[srcIdx + 1] = Math.round(srcG * (1 - blendFactor) + adjustedG * blendFactor);
            outputPixels[srcIdx + 2] = Math.round(srcB * (1 - blendFactor) + adjustedB * blendFactor);
          }
        }
      }
    }

    outputCtx.putImageData(outputData, 0, 0);
  };

  const updateBackground = useCallback((backgroundId: string) => {
    const bgOption = backgroundOptions.find(opt => opt.id === backgroundId);
    const bgValue = bgOption?.value || '';

    currentBackgroundRef.current = bgValue;

    setTransformationSettings(prev => ({
      ...prev,
      background: bgValue,
    }));

    if (bgValue) {
      setStatusMessage('Loading background...');
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.src = bgValue;

      img.onload = () => {
        backgroundImgRef.current = img;
        setStatusMessage('Background Active');
      };

      img.onerror = () => {
        backgroundImgRef.current = null;
        setStatusMessage('Background Load Failed');
      };
    } else {
      backgroundImgRef.current = null;
      setStatusMessage('Camera Ready');
    }
  }, []);

  const cleanup = useCallback(() => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }

    if (selfieSegmentationRef.current) {
      try {
        selfieSegmentationRef.current.close();
      } catch (e) {}
      selfieSegmentationRef.current = null;
    }

    if (faceMeshRef.current) {
      try {
        faceMeshRef.current.close();
      } catch (e) {}
      faceMeshRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
      videoRef.current = null;
    }

    if (referenceVideo) {
      referenceVideo.pause();
      referenceVideo.src = '';
    }

    inputStreamRef.current = null;
    backgroundImgRef.current = null;
    outputCanvasRef.current = null;
    inputCanvasRef.current = null;
    bgCanvasRef.current = null;
    refFrameCanvasRef.current = null;

    currentBackgroundRef.current = '';
    setProcessedStream(null);
    setIsProcessing(false);
    setStatusMessage('Camera Ready');
    isProcessingRef.current = false;
  }, [referenceVideo]);

  useEffect(() => {
    return () => {
      cleanup();
    };
  }, [cleanup]);

  return {
    processedStream,
    transformationSettings,
    setTransformationSettings,
    referenceVideo,
    setReferenceVideo,
    backgroundOptions,
    isProcessing,
    statusMessage,
    initializeTransform,
    updateBackground,
    cleanup,
  };
}
