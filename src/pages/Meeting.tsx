import { useEffect, useState, useRef, useCallback } from 'react';
import { useParams, useSearchParams, useNavigate } from 'react-router-dom';
import Logo from '../components/Logo';
import VideoFrame from '../components/VideoFrame';
import TransformPanel from '../components/TransformPanel';
import { useWebRTC } from '../hooks/useWebRTC';
import { useFaceTransform } from '../hooks/useFaceTransform';
import { socket } from '../lib/socket';
import {
  Video, VideoOff, Mic, MicOff, PhoneOff, Copy, Check, Users, PanelRightOpen, AlertCircle
} from 'lucide-react';

export default function Meeting() {
  const { roomId } = useParams<{ roomId: string }>();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();

  // Determine if user is host (creating room) or joining via link
  const isCreatingRoom = searchParams.get('host') === 'true';
  const isJoiningRoom = searchParams.get('join') === 'true';

  const [isAdmin, setIsAdmin] = useState(false);
  const [displayName, setDisplayName] = useState(() => searchParams.get('name') || '');
  const [hasJoined, setHasJoined] = useState(false);
  const [meetingDuration, setMeetingDuration] = useState(0);
  const [showInvitePrompt, setShowInvitePrompt] = useState(false);
  const [copied, setCopied] = useState(false);
  const [panelCollapsed, setPanelCollapsed] = useState(false);
  const [needsNameInput, setNeedsNameInput] = useState(isJoiningRoom && !searchParams.get('name'));
  const [error, setError] = useState<string | null>(null);

  const processedStreamRef = useRef<MediaStream | null>(null);
  const initializedRef = useRef(false);

  const {
    processedStream,
    transformationSettings,
    setTransformationSettings,
    referenceImage,
    setReferenceImage,
    initializeTransform,
    updateBackground,
    statusMessage,
    modelLoadProgress,
    cleanup: cleanupTransform,
  } = useFaceTransform();

  useEffect(() => {
    processedStreamRef.current = processedStream;
  }, [processedStream]);

  const {
    localStream,
    participants,
    startLocalStream,
    stopLocalStream,
    toggleCamera,
    toggleMicrophone,
    cameraEnabled,
    microphoneEnabled,
    initPeerConnections,
    cleanup,
  } = useWebRTC(roomId || null, processedStreamRef.current);

  const nameInputRef = useRef<HTMLInputElement>(null);
  const actualLocalStream = processedStream || localStream;

  // Connect socket on mount
  useEffect(() => {
    socket.connect();

    socket.on('connect_error', (err) => {
      console.error('Socket connection error:', err);
      setError('Could not connect to server');
    });

    return () => {
      socket.disconnect();
    };
  }, []);

  // Handle room creation for host
  useEffect(() => {
    if (isCreatingRoom && roomId && !initializedRef.current) {
      initializedRef.current = true;
      createRoomAndJoin();
    }
  }, [isCreatingRoom, roomId]);

  // Duration timer
  useEffect(() => {
    const interval = setInterval(() => {
      if (hasJoined) {
        setMeetingDuration(prev => prev + 1);
      }
    }, 1000);
    return () => clearInterval(interval);
  }, [hasJoined]);

  // Initialize transform when local stream is ready
  useEffect(() => {
    if (localStream && !processedStream) {
      initializeTransform(localStream);
    }
  }, [localStream, processedStream, initializeTransform]);

  const createRoomAndJoin = async () => {
    try {
      setError(null);

      // Start camera first
      const stream = await startLocalStream();

      // Create room via socket
      socket.emit('create-room', { roomId, displayName: 'Host' });

      socket.once('room-created', (data) => {
        setIsAdmin(true);
        setDisplayName(data.displayName || 'Host');
        setHasJoined(true);
        setShowInvitePrompt(true);
        setTimeout(() => setShowInvitePrompt(false), 5000);

        initPeerConnections(roomId!);

        if (stream) {
          initializeTransform(stream);
        }
      });

      socket.once('error', (err) => {
        setError(err.message);
      });
    } catch (err) {
      console.error('Failed to create room:', err);
      setError('Failed to access camera or create room');
    }
  };

  const joinRoomWithDisplayName = async (name: string) => {
    if (!roomId) {
      setError('Invalid room ID');
      return;
    }

    try {
      setError(null);
      setDisplayName(name);
      setNeedsNameInput(false);

      // Start camera
      const stream = await startLocalStream();

      // Join room via socket
      socket.emit('join-room', { roomId, displayName: name });

      socket.once('room-joined', (data) => {
        setIsAdmin(data.isAdmin);
        setDisplayName(data.displayName || name);
        setHasJoined(true);

        initPeerConnections(roomId);

        if (stream) {
          initializeTransform(stream);
        }
      });

      socket.once('error', (err) => {
        setError(err.message || 'Failed to join room');
      });
    } catch (err) {
      console.error('Failed to join room:', err);
      setError('Failed to access camera or join room');
    }
  };

  const leaveMeeting = useCallback(() => {
    cleanup();
    cleanupTransform();
    stopLocalStream();
    socket.disconnect();
    navigate('/');
  }, [cleanup, cleanupTransform, stopLocalStream, navigate]);

  const copyInviteLink = useCallback(() => {
    navigator.clipboard.writeText(`${window.location.origin}/meeting/${roomId}?join=true`);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [roomId]);

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const allParticipants = Array.from(participants.values());
  const totalParticipants = allParticipants.length + 1;

  // Name input screen (for guests joining via link)
  if (needsNameInput) {
    return (
      <div className="min-h-screen gradient-dark flex flex-col items-center justify-center px-4">
        <Logo size="lg" />
        <div className="max-w-md w-full mt-8 bg-dark-800/50 backdrop-blur-sm border border-dark-700 rounded-2xl p-8">
          <h2 className="text-xl font-semibold text-white mb-6 text-center">
            Join Meeting
          </h2>
          {error && (
            <div className="mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg flex items-center gap-2 text-red-400 text-sm">
              <AlertCircle size={16} />
              {error}
            </div>
          )}
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-dark-300 text-left mb-2">
                Enter your name
              </label>
              <input
                ref={nameInputRef}
                type="text"
                placeholder="Your name"
                className="w-full px-4 py-3 bg-dark-900 border border-dark-600 rounded-lg text-white placeholder-dark-500 focus:outline-none focus:border-primary-500 focus:ring-1 focus:ring-primary-500 transition-all"
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    const name = nameInputRef.current?.value?.trim();
                    if (name) joinRoomWithDisplayName(name);
                  }
                }}
              />
            </div>
            <div className="flex gap-3">
              <button
                onClick={() => navigate('/')}
                className="flex-1 px-6 py-3 bg-dark-700 hover:bg-dark-600 text-white rounded-lg font-medium transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  const name = nameInputRef.current?.value?.trim();
                  if (name) joinRoomWithDisplayName(name);
                }}
                className="flex-1 px-6 py-3 bg-primary-500 hover:bg-primary-600 text-white rounded-lg font-medium transition-colors"
              >
                Join
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Connecting screen
  if (!hasJoined) {
    return (
      <div className="min-h-screen gradient-dark flex flex-col items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <div className="w-12 h-12 border-4 border-primary-500 border-t-transparent rounded-full animate-spin" />
          <p className="text-dark-300">
            {error || 'Connecting to meeting...'}
          </p>
          {error && (
            <button
              onClick={() => navigate('/')}
              className="px-6 py-2 bg-primary-500 hover:bg-primary-600 text-white rounded-lg"
            >
              Back to Home
            </button>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen gradient-dark flex flex-col">
      {/* Header */}
      <header className="flex-shrink-0 h-16 bg-dark-900/90 backdrop-blur-sm border-b border-dark-700 flex items-center justify-between px-4 md:px-6 z-30">
        <div className="flex items-center gap-4">
          <Logo size="sm" />
          <div className="hidden sm:block h-6 w-px bg-dark-700" />
          <div className="hidden sm:flex items-center gap-3">
            <span className="text-sm text-dark-300 font-mono">{roomId}</span>
            <span className="text-dark-600">|</span>
            <span className="text-sm text-dark-300">{formatDuration(meetingDuration)}</span>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {isAdmin && (
            <button
              onClick={copyInviteLink}
              className="hidden sm:flex items-center gap-2 px-4 py-2 bg-dark-700 hover:bg-dark-600 text-white rounded-lg transition-colors text-sm"
            >
              {copied ? <Check size={16} /> : <Copy size={16} />}
              {copied ? 'Copied!' : 'Invite'}
            </button>
          )}
          <div className="flex items-center gap-1 px-3 py-2 bg-dark-800 rounded-lg">
            <Users size={16} className="text-dark-400" />
            <span className="text-sm font-medium text-white">{totalParticipants}</span>
          </div>
        </div>
      </header>

      {/* Invite prompt */}
      {showInvitePrompt && isAdmin && (
        <div className="fixed top-20 right-4 bg-dark-800 border border-dark-600 rounded-xl p-4 shadow-xl z-50 animate-fade-in">
          <p className="text-sm text-dark-300 mb-2">Share this link to invite participants:</p>
          <div className="flex items-center gap-2">
            <div className="px-3 py-2 bg-dark-900 rounded text-primary-400 text-sm font-mono truncate max-w-xs">
              {`${window.location.origin}/meeting/${roomId}?join=true`}
            </div>
            <button
              onClick={copyInviteLink}
              className="p-2 bg-primary-500 hover:bg-primary-600 rounded transition-colors shrink-0"
            >
              {copied ? <Check size={16} className="text-white" /> : <Copy size={16} className="text-white" />}
            </button>
          </div>
        </div>
      )}

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        <main className={`flex-1 flex items-center justify-center p-3 md:p-4 lg:p-6 transition-all duration-300 ${isAdmin && !panelCollapsed ? 'pr-80' : ''}`}>
          <div className={`grid gap-3 md:gap-4 w-full max-w-7xl ${
            totalParticipants <= 1 ? 'grid-cols-1 max-w-xl' :
            totalParticipants === 2 ? 'grid-cols-1 md:grid-cols-2 max-w-3xl' :
            totalParticipants <= 4 ? 'grid-cols-2' :
            totalParticipants <= 6 ? 'grid-cols-2 md:grid-cols-3' :
            'grid-cols-2 md:grid-cols-3 lg:grid-cols-4'
          }`}>
            <VideoFrame
              stream={actualLocalStream}
              participant={{
                id: 'local',
                displayName,
                isAdmin,
                cameraEnabled,
                microphoneEnabled,
              }}
              isLocal
            />
            {allParticipants.map(participant => (
              <VideoFrame
                key={participant.id}
                stream={participant.stream}
                participant={participant}
              />
            ))}
          </div>
        </main>

        {/* Transformation panel - ONLY for admin (host) */}
        {isAdmin && (
          <TransformPanel
            transformationSettings={transformationSettings}
            setTransformationSettings={setTransformationSettings}
            referenceImage={referenceImage}
            setReferenceImage={setReferenceImage}
            onBackgroundChange={updateBackground}
            isCollapsed={panelCollapsed}
            onToggleCollapse={() => setPanelCollapsed(!panelCollapsed)}
            statusMessage={statusMessage}
            modelLoadProgress={modelLoadProgress}
          />
        )}
      </div>

      {/* Footer controls */}
      <footer className="flex-shrink-0 h-16 md:h-20 bg-dark-900/90 backdrop-blur-sm border-t border-dark-700 flex items-center justify-center gap-3 md:gap-4 px-4 z-30">
        <button
          onClick={toggleMicrophone}
          className={`w-12 h-12 md:w-14 md:h-14 rounded-full flex items-center justify-center transition-all ${
            microphoneEnabled ? 'bg-dark-700 hover:bg-dark-600' : 'bg-red-600 hover:bg-red-700'
          }`}
        >
          {microphoneEnabled ? <Mic size={24} className="text-white" /> : <MicOff size={24} className="text-white" />}
        </button>
        <button
          onClick={toggleCamera}
          className={`w-12 h-12 md:w-14 md:h-14 rounded-full flex items-center justify-center transition-all ${
            cameraEnabled ? 'bg-dark-700 hover:bg-dark-600' : 'bg-red-600 hover:bg-red-700'
          }`}
        >
          {cameraEnabled ? <Video size={24} className="text-white" /> : <VideoOff size={24} className="text-white" />}
        </button>
        <button
          onClick={leaveMeeting}
          className="w-12 h-12 md:w-14 md:h-14 bg-red-600 hover:bg-red-700 rounded-full flex items-center justify-center transition-all"
        >
          <PhoneOff size={24} className="text-white" />
        </button>
        {isAdmin && (
          <button
            onClick={() => setPanelCollapsed(!panelCollapsed)}
            className="hidden md:flex w-12 h-12 md:w-14 md:h-14 bg-dark-700 hover:bg-dark-600 rounded-full items-center justify-center transition-all"
          >
            <PanelRightOpen size={24} className="text-white" />
          </button>
        )}
      </footer>
    </div>
  );
}
