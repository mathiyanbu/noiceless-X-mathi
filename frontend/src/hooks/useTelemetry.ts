import { useState, useEffect, useRef, useCallback } from 'react';
import type {
  MetricsResponse,
  SystemResponse,
  HealthResponse,
  ModelResponse,
  AudioDevicesResponse,
  WebSocketTelemetryPayload,
} from '../types/telemetry';
import { api } from '../services/api';

export function useTelemetry() {
  const [telemetry, setTelemetry] = useState<MetricsResponse | null>(null);
  const [system, setSystem] = useState<SystemResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [model, setModel] = useState<ModelResponse | null>(null);
  const [audioDevices, setAudioDevices] = useState<AudioDevicesResponse | null>(null);

  const [isWsConnected, setIsWsConnected] = useState(false);
  const [runtimeConnected, setRuntimeConnected] = useState(false);
  const [runtimeState, setRuntimeState] = useState<string>('offline');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const backoffRef = useRef(1000);

  // Periodic REST polling for system & hardware telemetry (CPU, RAM, Temp) every 1.5s
  const fetchSystemMetrics = useCallback(async () => {
    try {
      const sysData = await api.getSystem();
      setSystem(sysData);
    } catch {
      // System metrics fallback handled gracefully
    }
  }, []);

  const fetchHealth = useCallback(async () => {
    try {
      const h = await api.getHealth();
      setHealth(h);
      setRuntimeConnected(h.runtime_connected);
      if (h.runtime_state) {
        setRuntimeState(h.runtime_state);
      }
      if (!h.runtime_connected) {
        setErrorMessage(h.runtime_error || 'Runtime IPC disconnected');
      } else {
        setErrorMessage(null);
      }
    } catch (e: unknown) {
      setRuntimeConnected(false);
      setRuntimeState('offline');
      setErrorMessage(e instanceof Error ? e.message : 'Backend unreachable');
    }
  }, []);

  const fetchMetadata = useCallback(async () => {
    try {
      const [m, a] = await Promise.all([api.getModel(), api.getAudioDevices()]);
      setModel(m);
      setAudioDevices(a);
    } catch {
      // metadata load fallback
    }
  }, []);

  // WebSocket Connection Management
  const connectWebSocket = useCallback(() => {
    if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
      return;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    // When using Vite dev server with proxy, /ws is forwarded
    const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;

    try {
      const socket = new WebSocket(wsUrl);
      wsRef.current = socket;

      socket.onopen = () => {
        setIsWsConnected(true);
        backoffRef.current = 1000;
        fetchHealth();
      };

      socket.onmessage = (event) => {
        try {
          const payload: WebSocketTelemetryPayload = JSON.parse(event.data);
          if (payload.type === 'telemetry' && payload.data) {
            setTelemetry(payload.data);
            setRuntimeConnected(true);
            setErrorMessage(null);
            if (payload.data.fusion_mode === 'BYPASS') {
              setRuntimeState('bypass');
            }
          } else if (payload.type === 'telemetry_offline') {
            setRuntimeConnected(false);
            setRuntimeState('offline');
            setErrorMessage(payload.error || 'C++ runtime is offline');
          }
        } catch {
          // parse error
        }
      };

      socket.onclose = () => {
        setIsWsConnected(false);
        wsRef.current = null;
        // Exponential backoff reconnect
        reconnectTimeoutRef.current = setTimeout(() => {
          backoffRef.current = Math.min(backoffRef.current * 1.5, 10000);
          connectWebSocket();
        }, backoffRef.current);
      };

      socket.onerror = () => {
        socket.close();
      };
    } catch {
      // Socket instantiation error
    }
  }, [fetchHealth]);

  useEffect(() => {
    fetchHealth();
    fetchMetadata();
    fetchSystemMetrics();
    connectWebSocket();

    const sysInterval = setInterval(fetchSystemMetrics, 1500);
    const healthInterval = setInterval(fetchHealth, 3000);

    return () => {
      clearInterval(sysInterval);
      clearInterval(healthInterval);
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [fetchHealth, fetchMetadata, fetchSystemMetrics, connectWebSocket]);

  // Operator Controls
  const handleStart = async () => {
    setActionLoading('start');
    try {
      const res = await api.startRuntime();
      setRuntimeState(res.runtime_state || 'running');
      fetchHealth();
    } catch (e: unknown) {
      setErrorMessage(e instanceof Error ? e.message : 'Failed to start runtime');
    } finally {
      setActionLoading(null);
    }
  };

  const handleStop = async () => {
    setActionLoading('stop');
    try {
      const res = await api.stopRuntime();
      setRuntimeState(res.runtime_state || 'stopped');
      fetchHealth();
    } catch (e: unknown) {
      setErrorMessage(e instanceof Error ? e.message : 'Failed to stop runtime');
    } finally {
      setActionLoading(null);
    }
  };

  const handleBypass = async (bypassVal: boolean) => {
    setActionLoading('bypass');
    try {
      const res = await api.setBypass(bypassVal);
      setRuntimeState(bypassVal ? 'bypass' : (res.runtime_state || 'running'));
      fetchHealth();
    } catch (e: unknown) {
      setErrorMessage(e instanceof Error ? e.message : 'Failed to toggle bypass');
    } finally {
      setActionLoading(null);
    }
  };

  const handleReset = async () => {
    setActionLoading('reset');
    try {
      const res = await api.resetRuntime();
      setRuntimeState(res.runtime_state || 'ready');
      fetchHealth();
    } catch (e: unknown) {
      setErrorMessage(e instanceof Error ? e.message : 'Failed to reset runtime');
    } finally {
      setActionLoading(null);
    }
  };

  return {
    telemetry,
    system,
    health,
    model,
    audioDevices,
    isWsConnected,
    runtimeConnected,
    runtimeState,
    errorMessage,
    actionLoading,
    handleStart,
    handleStop,
    handleBypass,
    handleReset,
    refetchMetadata: fetchMetadata,
  };
}
