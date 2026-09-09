import type {
  HealthResponse,
  SystemResponse,
  AudioDevicesResponse,
  ModelResponse,
  MetricsResponse,
  RuntimeCommandResponse,
} from '../types/telemetry';

const API_BASE = '/api';

class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = 'ApiError';
  }
}

async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const url = `${API_BASE}${endpoint}`;
  const headers = {
    'Content-Type': 'application/json',
    ...(options.headers || {}),
  };

  const res = await fetch(url, { ...options, headers });
  if (!res.ok) {
    let errorDetail = res.statusText;
    try {
      const errData = await res.json();
      errorDetail = errData.detail || errData.message || errorDetail;
    } catch {
      // JSON parse fallback
    }
    throw new ApiError(res.status, errorDetail);
  }

  return (await res.json()) as T;
}

export const api = {
  getHealth: () => request<HealthResponse>('/health'),
  getSystem: () => request<SystemResponse>('/system'),
  getAudioDevices: () => request<AudioDevicesResponse>('/audio/devices'),
  getModel: () => request<ModelResponse>('/model'),
  getMetrics: () => request<MetricsResponse>('/metrics'),

  startRuntime: () =>
    request<RuntimeCommandResponse>('/runtime/start', {
      method: 'POST',
    }),

  stopRuntime: () =>
    request<RuntimeCommandResponse>('/runtime/stop', {
      method: 'POST',
    }),

  setBypass: (bypass: boolean) =>
    request<RuntimeCommandResponse>('/runtime/bypass', {
      method: 'POST',
      body: JSON.stringify({ bypass }),
    }),

  resetRuntime: () =>
    request<RuntimeCommandResponse>('/runtime/reset', {
      method: 'POST',
    }),
};
