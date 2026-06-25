export interface Participant {
  id: string;
  displayName: string;
  isAdmin: boolean;
  cameraEnabled?: boolean;
  microphoneEnabled?: boolean;
  stream?: MediaStream;
}

export interface Room {
  id: string;
  admin: string;
  participants: Map<string, Participant>;
}

export interface SignalData {
  type: 'offer' | 'answer' | 'ice-candidate';
  sdp?: RTCSessionDescriptionInit;
  candidate?: RTCIceCandidateInit;
}

export interface TransformationSettings {
  enabled: boolean;
  referenceImage: string | null;
  background: string;
  modelLoadProgress?: number;
}

export interface BackgroundOption {
  id: string;
  name: string;
  thumbnail: string;
  value: string;
}
