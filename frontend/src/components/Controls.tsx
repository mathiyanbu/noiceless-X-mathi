import React from 'react';
import { Play, Square, Shuffle, RotateCcw, AlertTriangle } from 'lucide-react';

interface ControlsProps {
  runtimeConnected: boolean;
  runtimeState: string;
  actionLoading: string | null;
  errorMessage: string | null;
  onStart: () => void;
  onStop: () => void;
  onBypass: (bypass: boolean) => void;
  onReset: () => void;
}

export const Controls: React.FC<ControlsProps> = ({
  runtimeConnected,
  runtimeState,
  actionLoading,
  errorMessage,
  onStart,
  onStop,
  onBypass,
  onReset,
}) => {
  const isRunning = runtimeState.toLowerCase() === 'running';
  const isBypass = runtimeState.toLowerCase() === 'bypass';

  return (
    <div className="glass-panel" style={{ padding: '20px', marginBottom: '20px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <h2 style={{ fontSize: '1rem', fontWeight: 700, letterSpacing: '-0.01em', marginBottom: '4px' }}>
            OPERATOR CONTROL PLANE
          </h2>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
            Real-time process state machine control via local IPC socket
          </p>
        </div>

        {/* Action Buttons */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
          {/* START */}
          <button
            className="ctrl-btn btn-start"
            onClick={onStart}
            disabled={!runtimeConnected || isRunning || actionLoading === 'start'}
          >
            <Play size={16} />
            <span>{actionLoading === 'start' ? 'Starting...' : 'START RUNTIME'}</span>
          </button>

          {/* STOP */}
          <button
            className="ctrl-btn btn-stop"
            onClick={onStop}
            disabled={!runtimeConnected || (!isRunning && !isBypass) || actionLoading === 'stop'}
          >
            <Square size={16} />
            <span>{actionLoading === 'stop' ? 'Stopping...' : 'STOP'}</span>
          </button>

          {/* BYPASS */}
          <button
            className={`ctrl-btn btn-bypass ${isBypass ? 'active' : ''}`}
            onClick={() => onBypass(!isBypass)}
            disabled={!runtimeConnected || actionLoading === 'bypass'}
          >
            <Shuffle size={16} />
            <span>{isBypass ? 'BYPASS ACTIVE' : 'BYPASS AI'}</span>
          </button>

          {/* RESET */}
          <button
            className="ctrl-btn btn-reset"
            onClick={onReset}
            disabled={!runtimeConnected || actionLoading === 'reset'}
          >
            <RotateCcw size={16} />
            <span>{actionLoading === 'reset' ? 'Resetting...' : 'RESET DSP'}</span>
          </button>
        </div>
      </div>

      {/* Offline / Error Alert Banner */}
      {errorMessage && (
        <div style={{
          marginTop: '16px',
          padding: '12px 16px',
          borderRadius: 'var(--radius-md)',
          background: 'rgba(239, 68, 68, 0.1)',
          border: '1px solid rgba(239, 68, 68, 0.25)',
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
          color: '#fca5a5',
          fontSize: '0.8125rem',
        }}>
          <AlertTriangle size={18} color="#ef4444" style={{ flexShrink: 0 }} />
          <div>
            <span style={{ fontWeight: 600 }}>Runtime Warning:</span> {errorMessage}
            {!runtimeConnected && (
              <div style={{ fontSize: '0.75rem', marginTop: '2px', color: 'var(--text-muted)' }}>
                Start the runtime with <code style={{ color: 'var(--accent-cyan)' }}>python sih26052.py --realtime</code> or C++ binary <code style={{ color: 'var(--accent-cyan)' }}>./build/sih26052 --realtime</code>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
