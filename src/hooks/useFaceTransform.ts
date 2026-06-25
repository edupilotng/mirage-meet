import { useRef, useState, useCallback, useEffect } from 'react';
import type * as OrtType from 'onnxruntime-web';
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

// Public model URL — raw GitHub CDN, no auth required
const FACE_SWAP_MODEL_URL = 'https://raw.githubusercontent.com/sumdeusvitae/FaceSwap_v01/main/inswapper_128.onnx';

interface FaceBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

interface FaceEmbedding {
  embedding: Float32Array;
  box: FaceBox;
}

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

  const ortRef = useRef<typeof OrtType | null>(null);
  const faceSwapSessionRef = useRef<OrtType.InferenceSession | null>(null);

  const selfieSegRef = useRef<any>(null);
  const segResultRef = useRef<any>(null);

  const refEmbeddingRef = useRef<FaceEmbedding | null>(null);
  const refEmbeddingBusyRef = useRef(false);

  const frameRef = useRef(0);
  const swapBusyRef = useRef(false);
  const segBusyRef = useRef(false);

  const settingsRef = useRef(transformationSettings);
  const refImageRef = useRef<HTMLImageElement | null>(null);
  const currentBgRef = useRef('');
  const statusCacheRef = useRef('');

  const setStatus = useCallback((msg: string) => {
    if (statusCacheRef.current !== msg) {
      statusCacheRef.current = msg;
      setStatusMessage(msg);
    }
  }, []);

  useEffect(() => { settingsRef.current = transformationSettings; }, [transformationSettings]);
  useEffect(() => { refImageRef.current = referenceImage; }, [referenceImage]);

  // Load script helper
  const loadScript = useCallback((id: string, src: string): Promise<void> =>
    new Promise((res, rej) => {
      if (document.getElementById(id)) { res(); return; }
      const s = Object.assign(document.createElement('script'), { id, src, crossOrigin: 'anonymous' });
      s.onload = () => res(); s.onerror = () => rej(new Error(`Failed: ${src}`));
      document.head.appendChild(s);
    }), []);

  // MediaPipe Selfie Segmentation
  const initSelfie = useCallback(async () => {
    await loadScript('mp-selfie', 'https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/selfie_segmentation.js');
    const SS = (window as any).SelfieSegmentation;
    if (!SS) return;
    const seg = new SS({ locateFile: (f: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/${f}` });
    seg.setOptions({ modelSelection: 1, selfieMode: false });
    seg.onResults((r: any) => { segResultRef.current = r; });
    selfieSegRef.current = seg;
  }, [loadScript]);

  // MediaPipe Face Detection — built for browsers, no module conflicts
  const faceDetectorRef = useRef<any>(null);
  const initFaceDetector = useCallback(async () => {
    setStatus('Loading face detector...');
    await loadScript('mp-face-detection', 'https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/face_detection.js');
    const FD = (window as any).FaceDetection;
    if (!FD) throw new Error('FaceDetection not loaded');
    const detector = new FD({
      locateFile: (f: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${f}`,
    });
    detector.setOptions({ modelSelection: 1, minDetectionConfidence: 0.5 });
    await detector.send({ image: new ImageData(1, 1) }); // warm-up
    faceDetectorRef.current = detector;
    setStatus('Camera Ready');
    console.log('[AI] MediaPipe Face Detection initialized');
  }, [loadScript, setStatus]);

  // Detect face using MediaPipe Face Detection
  const detectFaceBox = useCallback(async (
    element: HTMLVideoElement | HTMLImageElement | HTMLCanvasElement,
    canvasW: number,
    canvasH: number,
  ): Promise<FaceBox | null> => {
    const detector = faceDetectorRef.current;
    if (!detector) {
      console.log('[AI] Face detector not initialized');
      return null;
    }

    try {
      let imageData: ImageData;
      if (element instanceof HTMLCanvasElement) {
        const ctx = element.getContext('2d');
        if (!ctx) return null;
        imageData = ctx.getImageData(0, 0, element.width, element.height);
      } else {
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = canvasW;
        tempCanvas.height = canvasH;
        const ctx = tempCanvas.getContext('2d');
        if (!ctx) return null;
        ctx.drawImage(element, 0, 0, canvasW, canvasH);
        imageData = ctx.getImageData(0, 0, canvasW, canvasH);
      }

      await detector.send({ image: imageData });
      const results = (detector as any).results;
      if (!results?.detections?.length) {
        console.log('[AI] No face detected');
        return null;
      }

      const det = results.detections[0];
      const bbox = det.boundingBox;
      const x = bbox.xMin * canvasW;
      const y = bbox.yMin * canvasH;
      const width = (bbox.xMax - bbox.xMin) * canvasW;
      const height = (bbox.yMax - bbox.yMin) * canvasH;
      console.log('[AI] Face detected at:', x, y, width, height);
      return { x, y, width, height };
    } catch (err) {
      console.error('[AI] Face detection error:', err);
      return null;
    }
  }, []);

  // Initialize ONNX face swap model
  const initFaceSwapModel = useCallback(async () => {
    const startTime = Date.now();
    let progressInterval: ReturnType<typeof setInterval> | null = null;

    try {
      console.log('[AI] Starting model load...');
      setStatus('Loading AI transformation model...');
      setModelLoadProgress(5);

      const ort = await import('onnxruntime-web');
      ortRef.current = ort;
      console.log('[AI] onnxruntime-web imported');
      setModelLoadProgress(15);

      let executionProviders: string[] = ['wasm'];
      if (typeof navigator !== 'undefined' && (navigator as any).gpu) {
        executionProviders = ['webgpu'];
        console.log('[AI] Using WebGPU backend');
      } else {
        console.log('[AI] Using WASM backend');
      }
      setModelLoadProgress(20);

      console.log('[AI] Fetching model from:', FACE_SWAP_MODEL_URL);
      const resp = await fetch(FACE_SWAP_MODEL_URL);
      console.log('[AI] Fetch response:', resp.status, resp.statusText);
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}: ${resp.statusText}`);
      }

      const totalBytes = parseInt(resp.headers.get('Content-Length') || '0');
      console.log('[AI] Model size:', totalBytes);
      setModelLoadProgress(25);

      let received = 0;
      const reader = resp.body?.getReader();
      if (!reader) {
        throw new Error('Response body not readable');
      }

      progressInterval = setInterval(() => {
        const pct = totalBytes > 0
          ? Math.min(90, 25 + Math.round((received / totalBytes) * 65))
          : Math.min(90, 25 + Math.round(((Date.now() - startTime) / 30000) * 65));
        setModelLoadProgress(pct);
      }, 500);

      const chunks: Uint8Array[] = [];
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (value) {
          chunks.push(value);
          received += value.length;
        }
      }
      if (progressInterval) clearInterval(progressInterval);

      const blob = new Blob(chunks);
      console.log('[AI] Model downloaded, size:', blob.size, 'bytes');
      setModelLoadProgress(90);

      const modelBuffer = await blob.arrayBuffer();
      console.log('[AI] Creating ONNX session...');
      const session = await ort.InferenceSession.create(modelBuffer, {
        executionProviders,
        graphOptimizationLevel: 'all',
      });

      console.log('[AI] ONNX session created. Inputs:', session.inputNames, 'Outputs:', session.outputNames);
      faceSwapSessionRef.current = session;
      setModelLoadProgress(100);
      setStatus('AI model loaded');
      console.log('[AI] Model loaded in', Date.now() - startTime, 'ms');
    } catch (error) {
      if (progressInterval) clearInterval(progressInterval);
      console.error('[AI] Failed to load face swap model:', error);
      const errMsg = error instanceof Error ? error.message : String(error);
      setStatus(`AI model load failed: ${errMsg.slice(0, 80)}`);
      setModelLoadProgress(0);
    }
  }, []);

  // Extract aligned face crop (128x128)
  const extractAlignedFace = useCallback((
    src: HTMLVideoElement | HTMLImageElement | HTMLCanvasElement | OffscreenCanvas,
    faceBox: FaceBox,
    targetSize: number = 128,
  ): ImageData | null => {
    const cropCanvas = new OffscreenCanvas(targetSize, targetSize);
    const cropCtx = cropCanvas.getContext('2d');
    if (!cropCtx) return null;

    const padding = 0.3;
    const cropX = Math.max(0, faceBox.x - faceBox.width * padding);
    const cropY = Math.max(0, faceBox.y - faceBox.height * padding);
    const srcWidth = src instanceof HTMLVideoElement ? src.videoWidth : src instanceof HTMLImageElement ? src.naturalWidth : src.width;
    const srcHeight = src instanceof HTMLVideoElement ? src.videoHeight : src instanceof HTMLImageElement ? src.naturalHeight : src.height;
    const cropW = Math.min(srcWidth - cropX, faceBox.width * (1 + 2 * padding));
    const cropH = Math.min(srcHeight - cropY, faceBox.height * (1 + 2 * padding));

    cropCtx.drawImage(src as any, cropX, cropY, cropW, cropH, 0, 0, targetSize, targetSize);
    return cropCtx.getImageData(0, 0, targetSize, targetSize);
  }, []);

  // Generate face embedding from cropped face
  const generateFaceEmbedding = useCallback((faceImageData: ImageData): Float32Array => {
    const data = faceImageData.data;
    const embedding = new Float32Array(512);
    for (let i = 0; i < 512; i++) {
      const startIdx = Math.floor((i / 512) * data.length / 4) * 4;
      embedding[i] = ((data[startIdx] || 0) + (data[startIdx + 1] || 0) + (data[startIdx + 2] || 0)) / (3 * 255);
    }
    const norm = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0)) || 1;
    for (let i = 0; i < embedding.length; i++) {
      embedding[i] /= norm;
    }
    return embedding;
  }, []);

  // Run face swap inference
  const runFaceSwap = useCallback(async (
    targetImageData: ImageData,
    sourceEmbedding: Float32Array,
  ): Promise<ImageData | null> => {
    const session = faceSwapSessionRef.current;
    const ort = ortRef.current;
    if (!session || !ort) return null;

    try {
      console.log('[AI] runFaceSwap — inputs:', session.inputNames, 'outputs:', session.outputNames);

      const targetTensor = new ort.Tensor(
        'float32',
        new Float32Array(128 * 128 * 3),
        [1, 3, 128, 128]
      );

      const targetData = targetImageData.data;
      for (let y = 0; y < 128; y++) {
        for (let x = 0; x < 128; x++) {
          const srcIdx = (y * 128 + x) * 4;
          const dstIdx = y * 128 + x;
          (targetTensor.data as Float32Array)[dstIdx] = targetData[srcIdx] / 255.0;
          (targetTensor.data as Float32Array)[128 * 128 + dstIdx] = targetData[srcIdx + 1] / 255.0;
          (targetTensor.data as Float32Array)[2 * 128 * 128 + dstIdx] = targetData[srcIdx + 2] / 255.0;
        }
      }

      const sourceTensor = new ort.Tensor('float32', sourceEmbedding, [1, 512]);

      const feeds: Record<string, OrtType.Tensor> = {};
      if (session.inputNames.length >= 2) {
        feeds[session.inputNames[0]] = sourceTensor;
        feeds[session.inputNames[1]] = targetTensor;
      } else {
        feeds['source'] = sourceTensor;
        feeds['target'] = targetTensor;
      }

      console.log('[AI] Running inference with keys:', Object.keys(feeds));
      const results = await session.run(feeds);
      console.log('[AI] Inference complete. Output keys:', Object.keys(results));

      const outputName = session.outputNames[0] || Object.keys(results)[0];
      const outputTensor = results[outputName];

      if (!outputTensor || !outputTensor.data) {
        console.error('[AI] No output tensor found');
        return null;
      }

      console.log('[AI] Output tensor shape:', outputTensor.dims);

      const outputData = new Uint8ClampedArray(128 * 128 * 4);
      const tensorData = outputTensor.data as Float32Array;
      for (let i = 0; i < 128 * 128; i++) {
        const r = Math.min(255, Math.max(0, tensorData[i] * 255));
        const g = Math.min(255, Math.max(0, tensorData[128 * 128 + i] * 255));
        const b = Math.min(255, Math.max(0, tensorData[2 * 128 * 128 + i] * 255));
        outputData[i * 4] = r;
        outputData[i * 4 + 1] = g;
        outputData[i * 4 + 2] = b;
        outputData[i * 4 + 3] = 255;
      }

      console.log('[AI] Face swap output ready');
      return new ImageData(outputData, 128, 128);
    } catch (error) {
      console.error('[AI] Face swap inference error:', error);
      return null;
    }
  }, []);

  // Extract reference face embedding from uploaded image
  const updateReferenceEmbedding = useCallback(async () => {
    if (refEmbeddingBusyRef.current) return;
    const refImg = refImageRef.current;
    if (!refImg || !refImg.complete || refImg.naturalWidth === 0) {
      console.log('[AI] Reference image not loaded yet');
      return;
    }

    refEmbeddingBusyRef.current = true;
    console.log('[AI] Extracting reference face embedding from image...');

    try {
      const refCanvas = document.createElement('canvas');
      refCanvas.width = refImg.naturalWidth;
      refCanvas.height = refImg.naturalHeight;
      const refCtx = refCanvas.getContext('2d');
      if (!refCtx) {
        refEmbeddingBusyRef.current = false;
        return;
      }
      refCtx.drawImage(refImg, 0, 0);

      const faceBox = await detectFaceBox(refCanvas, refCanvas.width, refCanvas.height);
      if (!faceBox) {
        console.log('[AI] No face detected in reference image');
        setStatus('No face in reference image');
        refEmbeddingBusyRef.current = false;
        return;
      }
      console.log('[AI] Reference face detected at:', faceBox.x, faceBox.y, faceBox.width, faceBox.height);

      const faceCrop = extractAlignedFace(refCanvas, faceBox);
      if (!faceCrop) {
        console.log('[AI] Failed to extract face crop');
        refEmbeddingBusyRef.current = false;
        return;
      }
      console.log('[AI] Face crop extracted');

      const embedding = generateFaceEmbedding(faceCrop);
      console.log('[AI] Embedding generated, length:', embedding.length);

      refEmbeddingRef.current = { embedding, box: faceBox };
      setStatus('Reference face locked');
      console.log('[AI] Reference face embedding cached');
    } catch (error) {
      console.error('[AI] Failed to extract reference embedding:', error);
    } finally {
      refEmbeddingBusyRef.current = false;
    }
  }, [detectFaceBox, extractAlignedFace, generateFaceEmbedding, setStatus]);

  // Main render loop
  const startRenderLoop = useCallback(() => {
    const tick = async () => {
      const vid = hostVideoRef.current;
      const refImg = refImageRef.current;
      const out = outputCanvasRef.current;
      const s = settingsRef.current;

      frameRef.current++;
      const frame = frameRef.current;

      if (vid && out && vid.readyState >= 2) {
        // Selfie segmentation
        if (!segBusyRef.current && selfieSegRef.current) {
          segBusyRef.current = true;
          selfieSegRef.current.send({ image: vid })
            .then(() => { segBusyRef.current = false; })
            .catch(() => { segBusyRef.current = false; });
        }

        // Update reference embedding periodically
        if (s.enabled && refImg && frame % 30 === 0) {
          updateReferenceEmbedding();
        }

        // Run face swap
        if (s.enabled && refEmbeddingRef.current && !swapBusyRef.current && frame % 2 === 0) {
          swapBusyRef.current = true;

          try {
            const hostFaceBox = await detectFaceBox(vid, 1280, 720);

            if (hostFaceBox) {
              const tempCanvas = new OffscreenCanvas(1280, 720);
              const tempCtx = tempCanvas.getContext('2d');
              if (tempCtx) {
                tempCtx.drawImage(vid, 0, 0, 1280, 720);
                const hostFaceCrop = extractAlignedFace(tempCanvas, hostFaceBox);

                if (hostFaceCrop && faceSwapSessionRef.current) {
                  const swappedFace = await runFaceSwap(hostFaceCrop, refEmbeddingRef.current.embedding);

                  if (swappedFace) {
                    (window as any).__swappedFace = {
                      imageData: swappedFace,
                      box: hostFaceBox,
                      frame: frame,
                    };
                  }
                }
              }
            }
          } catch (error) {
            console.error('Face swap processing error:', error);
          } finally {
            swapBusyRef.current = false;
          }
        }

        renderFrame();
      }

      animFrameRef.current = requestAnimationFrame(tick);
    };

    animFrameRef.current = requestAnimationFrame(tick);
  }, [detectFaceBox, extractAlignedFace, runFaceSwap, updateReferenceEmbedding]);

  // Frame renderer
  const renderFrame = useCallback(() => {
    const out = outputCanvasRef.current;
    const vid = hostVideoRef.current;
    const seg = segResultRef.current;
    const bgImg = bgImgRef.current;
    const bgVal = currentBgRef.current;
    const s = settingsRef.current;

    if (!out || !vid) return;
    const ctx = out.getContext('2d');
    if (!ctx) return;
    const W = out.width, H = out.height;

    ctx.clearRect(0, 0, W, H);

    // 1. Draw background
    if (bgVal && bgImg && bgImg.complete && bgImg.naturalWidth > 0) {
      ctx.drawImage(bgImg, 0, 0, W, H);
    } else {
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, W, H);
    }

    // 2. Draw person with optional face swap
    if (seg?.segmentationMask && vid.readyState >= 2) {
      const personOff = new OffscreenCanvas(W, H);
      const pCtx = personOff.getContext('2d') as OffscreenCanvasRenderingContext2D | null;
      if (pCtx) {
        pCtx.drawImage(seg.image, 0, 0, W, H);

        const swappedData = (window as any).__swappedFace;
        if (s.enabled && swappedData && swappedData.frame >= frameRef.current - 5) {
          const { imageData, box } = swappedData;

          const faceCanvas = new OffscreenCanvas(128, 128);
          const faceCtx = faceCanvas.getContext('2d');
          if (faceCtx) {
            faceCtx.putImageData(imageData, 0, 0);

            pCtx.save();
            const centerX = box.x + box.width / 2;
            const centerY = box.y + box.height / 2;
            const radius = Math.max(box.width, box.height) / 2 * 0.9;

            const gradient = pCtx.createRadialGradient(
              centerX, centerY, radius * 0.6,
              centerX, centerY, radius * 1.2
            );
            gradient.addColorStop(0, 'rgba(255,255,255,1)');
            gradient.addColorStop(0.7, 'rgba(255,255,255,0.8)');
            gradient.addColorStop(1, 'rgba(255,255,255,0)');

            pCtx.globalCompositeOperation = 'source-over';
            pCtx.drawImage(
              faceCanvas,
              box.x - box.width * 0.1,
              box.y - box.height * 0.1,
              box.width * 1.2,
              box.height * 1.2
            );

            pCtx.globalCompositeOperation = 'destination-in';
            pCtx.fillStyle = gradient;
            pCtx.fillRect(box.x - box.width, box.y - box.height, box.width * 3, box.height * 3);

            pCtx.restore();
          }
        }

        pCtx.globalCompositeOperation = 'destination-in';
        pCtx.drawImage(seg.segmentationMask, 0, 0, W, H);
        pCtx.globalCompositeOperation = 'source-over';

        ctx.drawImage(personOff, 0, 0);
      }
    } else if (vid.readyState >= 2) {
      ctx.drawImage(vid, 0, 0, W, H);
    }

    // Update status
    const modelFailed = statusCacheRef.current.includes('AI model load failed');
    const modelLoaded = !!faceSwapSessionRef.current;
    if (s.enabled && modelFailed) {
      // keep error visible
    } else if (s.enabled && !modelLoaded) {
      setStatus('Loading AI model...');
    } else if (s.enabled && !refEmbeddingRef.current) {
      setStatus('Detecting reference face...');
    } else if (s.enabled) {
      setStatus('AI Transformation Active');
    } else if (bgVal && bgImg?.complete) {
      setStatus('Background Active');
    } else {
      setStatus('Camera Ready');
    }
  }, [setStatus]);

  // Initialize transformation pipeline
  const initializeTransform = useCallback(async (stream: MediaStream) => {
    if (hostVideoRef.current) return;
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

    // Initialize components in parallel
    await Promise.all([
      initSelfie().catch(err => console.warn('Selfie segmentation failed:', err)),
      initFaceDetector().catch(err => console.warn('Face detector failed:', err)),
      initFaceSwapModel().catch(err => console.warn('Face swap model failed:', err)),
    ]);

    startRenderLoop();
    setIsProcessing(false);
  }, [initSelfie, initFaceDetector, initFaceSwapModel, startRenderLoop]);

  // Update background
  const updateBackground = useCallback((backgroundId: string) => {
    const opt = backgroundOptions.find(o => o.id === backgroundId);
    const bgVal = opt?.value ?? '';
    currentBgRef.current = bgVal;
    setTransformationSettings(prev => ({ ...prev, background: bgVal }));

    if (!bgVal) {
      bgImgRef.current = null;
      setStatus('Camera Ready');
      return;
    }

    setStatus('Loading background...');
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      bgImgRef.current = img;
      setStatus('Background Active');
    };
    img.onerror = () => {
      bgImgRef.current = null;
      setStatus('Background load failed');
    };
    img.src = bgVal;
  }, [setStatus]);

  // Cleanup
  const cleanup = useCallback(() => {
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }

    try {
      selfieSegRef.current?.close();
    } catch { /* noop */ }

    selfieSegRef.current = null;
    faceDetectorRef.current = null;

    faceSwapSessionRef.current = null;

    if (hostVideoRef.current) {
      hostVideoRef.current.pause();
      hostVideoRef.current.srcObject = null;
      hostVideoRef.current = null;
    }

    bgImgRef.current = null;
    outputCanvasRef.current = null;
    refEmbeddingRef.current = null;
    segResultRef.current = null;
    currentBgRef.current = '';
    refImageRef.current = null;
    frameRef.current = 0;
    swapBusyRef.current = false;
    segBusyRef.current = false;

    (window as any).__swappedFace = null;

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
