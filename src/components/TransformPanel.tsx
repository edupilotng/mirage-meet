import { Upload, X, ChevronLeft, ChevronRight, CheckCircle, AlertCircle, Loader, Sparkles } from 'lucide-react';
import { useState, useRef, useEffect } from 'react';
import type { TransformationSettings } from '../types';
import { backgroundOptions } from '../hooks/useFaceTransform';

interface TransformPanelProps {
  transformationSettings: TransformationSettings;
  setTransformationSettings: React.Dispatch<React.SetStateAction<TransformationSettings>>;
  referenceVideo: HTMLVideoElement | null;
  setReferenceVideo: (video: HTMLVideoElement | null) => void;
  onBackgroundChange: (backgroundId: string) => void;
  isCollapsed: boolean;
  onToggleCollapse: () => void;
  statusMessage: string;
}

export default function TransformPanel({
  transformationSettings,
  setTransformationSettings,
  referenceVideo: _referenceVideo,
  setReferenceVideo,
  onBackgroundChange,
  isCollapsed,
  onToggleCollapse,
  statusMessage,
}: TransformPanelProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [errorMessage, setErrorMessage] = useState<string>('');
  const videoInputRef = useRef<HTMLInputElement>(null);
  const previewVideoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const handleVideoUpload = (file: File) => {
    setUploadStatus('loading');
    setErrorMessage('');

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    const url = URL.createObjectURL(file);
    setPreviewUrl(url);

    const video = document.createElement('video');
    video.src = url;
    video.loop = true;
    video.muted = true;
    video.playsInline = true;
    video.crossOrigin = 'anonymous';

    video.onloadedmetadata = () => {
      video.play().catch(() => {});
    };

    video.onloadeddata = () => {
      setUploadStatus('success');
      setReferenceVideo(video);

      setTransformationSettings(prev => ({
        ...prev,
        referenceVideo: url,
      }));
    };

    video.onerror = () => {
      setUploadStatus('error');
      setErrorMessage('Failed to load video. Try a different format.');
    };
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('video/')) {
      handleVideoUpload(file);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleVideoUpload(file);
    }
  };

  const clearVideo = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(null);
    setUploadStatus('idle');
    setErrorMessage('');
    setReferenceVideo(null);
    setTransformationSettings(prev => ({
      ...prev,
      referenceVideo: null,
      enabled: false,
    }));
  };

  const toggleTransformation = () => {
    setTransformationSettings(prev => ({
      ...prev,
      enabled: !prev.enabled,
    }));
  };

  const getStatusColor = () => {
    if (statusMessage.includes('Active') || statusMessage.includes('Ready')) return 'text-green-400';
    if (statusMessage.includes('Loading') || uploadStatus === 'loading') return 'text-yellow-400';
    if (statusMessage.includes('Failed') || statusMessage.includes('Error')) return 'text-red-400';
    return 'text-dark-300';
  };

  const getStatusIcon = () => {
    if (statusMessage.includes('Loading') || uploadStatus === 'loading') {
      return <Loader size={14} className="text-yellow-400 animate-spin" />;
    }
    if (statusMessage.includes('Active')) {
      return <Sparkles size={14} className="text-primary-400" />;
    }
    if (statusMessage.includes('Ready') || uploadStatus === 'success') {
      return <CheckCircle size={14} className="text-green-400" />;
    }
    if (statusMessage.includes('Failed') || statusMessage.includes('Error')) {
      return <AlertCircle size={14} className="text-red-400" />;
    }
    return null;
  };

  return (
    <div
      className={`fixed top-16 right-0 bottom-16 w-80 bg-dark-900 border-l border-dark-700
        transition-transform duration-300 z-20
        ${isCollapsed ? 'translate-x-full' : 'translate-x-0'}`}
    >
      <button
        onClick={onToggleCollapse}
        className="absolute -left-10 top-1/2 -translate-y-1/2 w-10 h-20 bg-dark-800
          border border-dark-700 rounded-l-lg flex items-center justify-center
          hover:bg-dark-700 transition-colors"
      >
        {isCollapsed ? (
          <ChevronLeft size={20} className="text-white" />
        ) : (
          <ChevronRight size={20} className="text-white" />
        )}
      </button>

      <div className="h-full overflow-y-auto p-4">
        <h2 className="text-lg font-semibold text-white mb-2">AI Transformation</h2>
        <p className="text-xs text-dark-400 mb-4">
          Upload a reference video to transform your appearance. The AI will map the face from the video onto your webcam feed.
        </p>

        <div className="mb-4 flex items-center gap-2 px-3 py-2 bg-dark-800 rounded-lg border border-dark-700">
          {getStatusIcon()}
          <span className={`text-xs ${getStatusColor()}`}>{statusMessage}</span>
        </div>

        <div className="space-y-6">
          <div>
            <label className="text-sm font-medium text-dark-300 block mb-2">
              Reference Video (Face Source)
            </label>
            <div
              className={`relative border-2 border-dashed rounded-lg p-4 transition-colors cursor-pointer ${
                isDragging ? 'border-primary-400 bg-primary-400/10' : 'border-dark-600 hover:border-dark-500'
              } ${previewUrl ? 'border-primary-500' : ''}`}
              onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              onClick={() => videoInputRef.current?.click()}
            >
              {previewUrl ? (
                <div className="relative">
                  <video
                    ref={previewVideoRef}
                    src={previewUrl}
                    className="w-full h-32 object-cover rounded-lg"
                    autoPlay
                    loop
                    muted
                    playsInline
                  />
                  {uploadStatus === 'success' && (
                    <div className="absolute top-2 left-2 px-2 py-1 bg-green-500/90 rounded text-xs text-white flex items-center gap-1">
                      <CheckCircle size={12} />
                      Video Loaded
                    </div>
                  )}
                  {uploadStatus === 'error' && (
                    <div className="absolute top-2 left-2 px-2 py-1 bg-red-500/90 rounded text-xs text-white flex items-center gap-1">
                      <AlertCircle size={12} />
                      {errorMessage || 'Upload Failed'}
                    </div>
                  )}
                  <button
                    onClick={(e) => { e.stopPropagation(); clearVideo(); }}
                    className="absolute top-2 right-2 w-6 h-6 rounded-full bg-red-500 flex items-center justify-center hover:bg-red-600 transition-colors"
                  >
                    <X size={14} className="text-white" />
                  </button>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center h-32 text-center">
                  {uploadStatus === 'loading' ? (
                    <div className="flex flex-col items-center">
                      <Loader size={24} className="text-primary-400 animate-spin mb-2" />
                      <p className="text-sm text-dark-400">Processing video...</p>
                    </div>
                  ) : (
                    <>
                      <Upload size={24} className="text-dark-400 mb-2" />
                      <p className="text-sm text-dark-400">
                        Drag & drop video or click to upload
                      </p>
                      <p className="text-xs text-dark-500 mt-1">
                        MP4, WebM, MOV supported
                      </p>
                    </>
                  )}
                </div>
              )}
              <input
                ref={videoInputRef}
                type="file"
                accept="video/*"
                className="hidden"
                onChange={handleFileSelect}
              />
            </div>
            <p className="text-xs text-dark-500 mt-2">
              Upload a video containing a face. The AI will extract and apply facial features.
            </p>
          </div>

          <div>
            <label className="text-sm font-medium text-dark-300 block mb-3">
              Virtual Background
            </label>
            <div className="grid grid-cols-2 gap-2">
              {backgroundOptions.map((bg) => (
                <button
                  key={bg.id}
                  className={`relative rounded-lg overflow-hidden aspect-video border-2 transition-all ${
                    transformationSettings.background === bg.value
                      ? 'border-primary-400 ring-2 ring-primary-400/30'
                      : 'border-dark-600 hover:border-dark-500'
                  }`}
                  onClick={() => onBackgroundChange(bg.id)}
                >
                  {bg.id === 'none' ? (
                    <div className="w-full h-full bg-dark-700 flex items-center justify-center">
                      <span className="text-xs text-dark-400">None</span>
                    </div>
                  ) : (
                    <img
                      src={bg.thumbnail}
                      alt={bg.name}
                      className="w-full h-full object-cover"
                      loading="lazy"
                      onError={(e) => {
                        const target = e.target as HTMLImageElement;
                        target.style.display = 'none';
                        if (target.parentElement) {
                          target.parentElement.innerHTML = `<div class="w-full h-full bg-dark-700 flex items-center justify-center"><span class="text-xs text-dark-400">${bg.name}</span></div>`;
                        }
                      }}
                    />
                  )}
                  <div className="absolute bottom-0 left-0 right-0 bg-dark-900/80 px-2 py-1">
                    <span className="text-xs text-white truncate">{bg.name}</span>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {transformationSettings.referenceVideo && (
            <div className="pt-4 border-t border-dark-700">
              <button
                onClick={toggleTransformation}
                className={`w-full py-3 px-4 rounded-lg font-medium transition-all flex items-center justify-center gap-2 ${
                  transformationSettings.enabled
                    ? 'bg-primary-500 text-white hover:bg-primary-600 shadow-lg shadow-primary-500/20'
                    : 'bg-dark-700 text-dark-300 hover:bg-dark-600 hover:text-white'
                }`}
              >
                {transformationSettings.enabled ? (
                  <>
                    <Sparkles size={16} />
                    Transformation Active
                  </>
                ) : (
                  <>
                    <Sparkles size={16} className="opacity-50" />
                    Enable Face Transformation
                  </>
                )}
              </button>
              <p className="text-xs text-dark-400 mt-2 text-center">
                {transformationSettings.enabled
                  ? 'AI is transforming your face using the reference video. Move your head for best results.'
                  : 'Click to activate AI-powered face transformation'}
              </p>

              {transformationSettings.enabled && (
                <div className="mt-3 p-2 bg-primary-500/10 border border-primary-500/20 rounded-lg">
                  <p className="text-xs text-primary-300">
                    Tip: For best results, ensure good lighting and face the camera directly. The AI tracks your face position and applies the reference features.
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
