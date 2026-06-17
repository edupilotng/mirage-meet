import { useRef, useState, useCallback, useEffect } from 'react';

interface FaceLandmarks {
  keypoints: Array<{ x: number; y: number; z: number }>;
  box: { xMin: number; yMin: number; width: number; height: number };
}

interface UseFaceMeshTransformReturn {
  isReady: boolean;
  status: string;
  initialize: () => Promise<void>;
  processFrame: (
    srcCanvas: HTMLCanvasElement,
    refCanvas: HTMLCanvasElement,
    outputCanvas: HTMLCanvasElement,
    refVideo: HTMLVideoElement
  ) => Promise<void>;
  cleanup: () => void;
}

const MEDIAPIPE_FACE_MESH_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619/face_mesh.js';
const MEDIAPIPE_CAMERA_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils@0.3.1675466862/camera_utils.js';
const MEDIAPIPE_DRAWING_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils@0.3.1675466124/drawing_utils.js';

export function useFaceMeshTransform(): UseFaceMeshTransformReturn {
  const [isReady, setIsReady] = useState(false);
  const [status, setStatus] = useState('Initializing...');

  const faceMeshRef = useRef<any>(null);
  const sourceLandmarksRef = useRef<FaceLandmarks | null>(null);
  const referenceLandmarksRef = useRef<FaceLandmarks | null>(null);
  const isInitializedRef = useRef(false);

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

  const initialize = useCallback(async () => {
    if (isInitializedRef.current) return;

    try {
      setStatus('Loading AI models...');

      await loadScript('mediapipe-face-mesh', MEDIAPIPE_FACE_MESH_URL);
      await loadScript('mediapipe-camera', MEDIAPIPE_CAMERA_URL);
      await loadScript('mediapipe-drawing', MEDIAPIPE_DRAWING_URL);

      if (!(window as any).FaceMesh) {
        throw new Error('FaceMesh not loaded');
      }

      setStatus('Initializing face mesh...');

      const faceMesh = new (window as any).FaceMesh({
        locateFile: (file: string) => {
          return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619/${file}`;
        }
      });

      faceMesh.setOptions({
        maxNumFaces: 1,
        refineLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
      });

      faceMesh.onResults((results: any) => {
        if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
          const landmarks = results.multiFaceLandmarks[0];
          const box = results.multiFaceLandmarks[0].boundingBox || calculateBoundingBox(landmarks, results.image.width, results.image.height);

          sourceLandmarksRef.current = {
            keypoints: landmarks.map((lm: any) => ({
              x: lm.x,
              y: lm.y,
              z: lm.z
            })),
            box: {
              xMin: box.xMin || box.originX || 0,
              yMin: box.yMin || box.originY || 0,
              width: box.width || 200,
              height: box.height || 200
            }
          };
        }
      });

      faceMeshRef.current = faceMesh;
      isInitializedRef.current = true;
      setIsReady(true);
      setStatus('Face mesh ready');
    } catch (error) {
      console.error('Failed to initialize face mesh:', error);
      setStatus('Failed to load AI models');
    }
  }, [loadScript]);

  const calculateBoundingBox = (landmarks: any[], width: number, height: number) => {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;

    for (const lm of landmarks) {
      minX = Math.min(minX, lm.x * width);
      maxX = Math.max(maxX, lm.x * width);
      minY = Math.min(minY, lm.y * height);
      maxY = Math.max(maxY, lm.y * height);
    }

    return {
      xMin: minX,
      yMin: minY,
      width: maxX - minX,
      height: maxY - minY
    };
  };

  const detectFaceLandmarks = useCallback((video: HTMLVideoElement): Promise<FaceLandmarks | null> => {
    return new Promise((resolve) => {
      if (!faceMeshRef.current) {
        resolve(null);
        return;
      }

      faceMeshRef.current.onResults((results: any) => {
        if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
          const landmarks = results.multiFaceLandmarks[0];
          const box = calculateBoundingBox(landmarks, results.image.width, results.image.height);
          resolve({
            keypoints: landmarks.map((lm: any) => ({
              x: lm.x,
              y: lm.y,
              z: lm.z
            })),
            box: {
              xMin: box.xMin,
              yMin: box.yMin,
              width: box.width,
              height: box.height
            }
          });
        } else {
          resolve(null);
        }
      });

      faceMeshRef.current.send({ image: video });
    });
  }, []);

  const getReferenceFrameLandmarks = useCallback(async (refVideo: HTMLVideoElement): Promise<FaceLandmarks | null> => {
    if (!faceMeshRef.current) return null;

    return new Promise((resolve) => {
      faceMeshRef.current.onResults((results: any) => {
        if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
          const landmarks = results.multiFaceLandmarks[0];
          const box = calculateBoundingBox(landmarks, results.image.width, results.image.height);
          referenceLandmarksRef.current = {
            keypoints: landmarks.map((lm: any) => ({
              x: lm.x,
              y: lm.y,
              z: lm.z
            })),
            box: {
              xMin: box.xMin,
              yMin: box.yMin,
              width: box.width,
              height: box.height
            }
          };
          resolve(referenceLandmarksRef.current);
        } else {
          resolve(null);
        }
      });

      faceMeshRef.current.send({ image: refVideo });
    });
  }, []);

  const processFrame = useCallback(async (
    srcCanvas: HTMLCanvasElement,
    refCanvas: HTMLCanvasElement,
    outputCanvas: HTMLCanvasElement,
    refVideo: HTMLVideoElement
  ) => {
    const ctx = outputCanvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    const width = outputCanvas.width;
    const height = outputCanvas.height;

    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(srcCanvas, 0, 0, width, height);

    if (!faceMeshRef.current || refVideo.readyState < 2) {
      return;
    }

    const refLandmarks = await getReferenceFrameLandmarks(refVideo);

    if (!refLandmarks || refLandmarks.keypoints.length === 0) {
      return;
    }

    renderFaceTransformation(ctx, srcCanvas, refCanvas, refLandmarks, width, height);
  }, [getReferenceFrameLandmarks]);

  const renderFaceTransformation = (
    ctx: CanvasRenderingContext2D,
    srcCanvas: HTMLCanvasElement,
    refCanvas: HTMLCanvasElement,
    refLandmarks: FaceLandmarks,
    width: number,
    height: number
  ) => {
    const srcCtx = srcCanvas.getContext('2d', { willReadFrequently: true });
    const refCtx = refCanvas.getContext('2d', { willReadFrequently: true } as any);

    if (!srcCtx || !refCtx) return;

    const srcData = srcCtx.getImageData(0, 0, srcCanvas.width, srcCanvas.height);
    const refData = refCtx.getImageData(0, 0, refCanvas.width, refCanvas.height);

    const srcBox = srcCanvas.getBoundingClientRect();
    const refBox = refCanvas.getBoundingClientRect();

    const faceIndices = getFaceRegionIndices();
    const faceOvalIndices = getFaceOvalIndices();
    const lipsIndices = getLipsIndices();
    const leftEyeIndices = getLeftEyeIndices();
    const rightEyeIndices = getRightEyeIndices();

    const refFaceBox = refLandmarks.box;

    const scaleFaceX = (srcCanvas.width / refCanvas.width);
    const scaleFaceY = (srcCanvas.height / refCanvas.height);

    const faceCenterX = refFaceBox.xMin + refFaceBox.width / 2;
    const faceCenterY = refFaceBox.yMin + refFaceBox.height / 2;

    const outputData = ctx.getImageData(0, 0, width, height);
    const outputPixels = outputData.data;

    const refPixels = refData.data;
    const srcPixels = srcData.data;

    const scaleX = width / srcCanvas.width;
    const scaleY = height / srcCanvas.height;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const srcX = x / scaleX;
        const srcY = y / scaleY;

        const relX = (srcX - faceCenterX) / (refFaceBox.width / 2);
        const relY = (srcY - faceCenterY) / (refFaceBox.height / 2);

        const distance = Math.sqrt(relX * relX + relY * relY);

        if (distance < 1.2) {
          const blendFactor = Math.max(0, 1 - distance / 1.2);
          const edgeSoftness = Math.pow(blendFactor, 0.5);

          const refX = Math.floor(srcX * scaleFaceX);
          const refY = Math.floor(srcY * scaleFaceY);

          if (refX >= 0 && refX < refCanvas.width && refY >= 0 && refY < refCanvas.height) {
            const srcIdx = (Math.floor(srcY) * srcCanvas.width + Math.floor(srcX)) * 4;
            const refIdx = (refY * refCanvas.width + refX) * 4;
            const outIdx = (y * width + x) * 4;

            const srcR = srcPixels[srcIdx] || 0;
            const srcG = srcPixels[srcIdx + 1] || 0;
            const srcB = srcPixels[srcIdx + 2] || 0;
            const srcA = srcPixels[srcIdx + 3] || 255;

            const refR = refPixels[refIdx] || 0;
            const refG = refPixels[refIdx + 1] || 0;
            const refB = refPixels[refIdx + 2] || 0;
            const refA = refPixels[refIdx + 3] || 255;

            const matchFactor = calculateColorMatch(srcR, srcG, srcB, refR, refG, refB);
            const colorCorrection = 0.3 + matchFactor * 0.2;

            const adjustedRefR = srcR + (refR - srcR) * colorCorrection;
            const adjustedRefG = srcG + (refG - srcG) * colorCorrection;
            const adjustedRefB = srcB + (refB - srcB) * colorCorrection;

            outputPixels[outIdx] = Math.round(srcR * (1 - edgeSoftness) + adjustedRefR * edgeSoftness);
            outputPixels[outIdx + 1] = Math.round(srcG * (1 - edgeSoftness) + adjustedRefG * edgeSoftness);
            outputPixels[outIdx + 2] = Math.round(srcB * (1 - edgeSoftness) + adjustedRefB * edgeSoftness);
            outputPixels[outIdx + 3] = Math.round(srcA * (1 - edgeSoftness) + refA * edgeSoftness);
          }
        }
      }
    }

    ctx.putImageData(outputData, 0, 0);
  };

  const calculateColorMatch = (r1: number, g1: number, b1: number, r2: number, g2: number, b2: number): number => {
    const avg1 = (r1 + g1 + b1) / 3;
    const avg2 = (r2 + g2 + b2) / 3;
    const diff = Math.abs(avg1 - avg2) / 255;
    return 1 - diff;
  };

  const getFaceRegionIndices = (): number[] => {
    return [
      10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
      397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
      172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
    ];
  };

  const getFaceOvalIndices = (): number[] => {
    return [
      10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
      397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
      172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
    ];
  };

  const getLipsIndices = (): number[] => {
    return [
      61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409,
      78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308
    ];
  };

  const getLeftEyeIndices = (): number[] => {
    return [
      33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158,
      159, 160, 161, 246
    ];
  };

  const getRightEyeIndices = (): number[] => {
    return [
      362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387,
      386, 385, 384, 398
    ];
  };

  const cleanup = useCallback(() => {
    if (faceMeshRef.current) {
      try {
        faceMeshRef.current.close();
      } catch (e) {}
      faceMeshRef.current = null;
    }
    sourceLandmarksRef.current = null;
    referenceLandmarksRef.current = null;
    isInitializedRef.current = false;
    setIsReady(false);
    setStatus('Cleaned up');
  }, []);

  useEffect(() => {
    return () => {
      cleanup();
    };
  }, [cleanup]);

  return {
    isReady,
    status,
    initialize,
    processFrame,
    cleanup
  };
}
