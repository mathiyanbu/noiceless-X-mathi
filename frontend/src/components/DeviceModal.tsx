import React from 'react';
import { X, Radio, Volume2, Mic, CheckCircle2, HardDrive } from 'lucide-react';
import type { AudioDevicesResponse, ModelResponse } from '../types/telemetry';

interface DeviceModalProps {
  isOpen: boolean;
  onClose: () => void;
  audioDevices: AudioDevicesResponse | null;
  model: ModelResponse | null;
}

export const DeviceModal: React.FC<DeviceModalProps> = ({
  isOpen,
  onClose,
  audioDevices,
  model,
}) => {
  if (!isOpen) return null;

  const cfg = audioDevices?.config;
  const devices = audioDevices?.devices || [];

  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      zIndex: 100,
      background: 'rgba(0, 0, 0, 0.75)',
      backdropFilter: 'blur(8px)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '20px',
    }}>
      <div className="glass-panel" style={{
        width: '100%',
        maxWidth: '720px',
        maxHeight: '90vh',
        overflowY: 'auto',
        padding: '24px',
        background: '#0d1117',
        border: '1px solid rgba(0, 240, 255, 0.3)',
        boxShadow: '0 20px 50px rgba(0, 0, 0, 0.8), 0 0 30px rgba(0, 240, 255, 0.15)',
      }}>
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <HardDrive size={20} color="var(--accent-cyan)" />
            <h2 style={{ fontSize: '1.125rem', fontWeight: 700, letterSpacing: '-0.01em' }}>
              HARDWARE ROUTING & AI ENGINE CONFIGURATION
            </h2>
          </div>
          <button
            onClick={onClose}
            style={{
              background: 'transparent',
              border: 'none',
              color: 'var(--text-muted)',
              cursor: 'pointer',
              padding: '4px',
            }}
          >
            <X size={20} />
          </button>
        </div>

        {/* Audio Configuration Parameters */}
        <div style={{ marginBottom: '24px' }}>
          <h3 style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--accent-cyan)', textTransform: 'uppercase', marginBottom: '12px' }}>
            ALSA Audio Pipeline Parameters
          </h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px', fontSize: '0.8125rem', fontFamily: 'var(--font-mono)' }}>
            <div style={{ padding: '10px', background: 'rgba(255, 255, 255, 0.02)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>Sample Rate</div>
              <div style={{ fontWeight: 600, color: '#fff' }}>{cfg?.sample_rate || 16000} Hz</div>
            </div>
            <div style={{ padding: '10px', background: 'rgba(255, 255, 255, 0.02)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>Hop / Frame Size</div>
              <div style={{ fontWeight: 600, color: '#fff' }}>{cfg?.hop_ms || 5} ms / {cfg?.frame_ms || 10} ms</div>
            </div>
            <div style={{ padding: '10px', background: 'rgba(255, 255, 255, 0.02)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>ALSA Period / Buffer</div>
              <div style={{ fontWeight: 600, color: '#fff' }}>{cfg?.period_size || 160} / {cfg?.buffer_size || 640}</div>
            </div>
          </div>
        </div>

        {/* Target Audio Routes */}
        <div style={{ marginBottom: '24px' }}>
          <h3 style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--accent-purple)', textTransform: 'uppercase', marginBottom: '12px' }}>
            Configured ALSA Audio Routes
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '0.8125rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 14px', background: 'rgba(255, 255, 255, 0.02)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Mic size={15} color="var(--accent-cyan)" />
                <span style={{ fontWeight: 600 }}>Primary Microphone (Headphone Mic):</span>
              </div>
              <code style={{ color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)' }}>{cfg?.primary_device || 'hw:CARD=Headset,DEV=0'}</code>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 14px', background: 'rgba(255, 255, 255, 0.02)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Radio size={15} color="var(--accent-purple)" />
                <span style={{ fontWeight: 600 }}>Reference Microphone (Error Mic):</span>
              </div>
              <code style={{ color: 'var(--accent-purple)', fontFamily: 'var(--font-mono)' }}>{cfg?.reference_device || 'hw:CARD=ErrorMic,DEV=0'}</code>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 14px', background: 'rgba(255, 255, 255, 0.02)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Volume2 size={15} color="var(--accent-emerald)" />
                <span style={{ fontWeight: 600 }}>Playback DAC Output:</span>
              </div>
              <code style={{ color: 'var(--accent-emerald)', fontFamily: 'var(--font-mono)' }}>{cfg?.output_device || 'hw:CARD=Headset,DEV=0'}</code>
            </div>
          </div>
        </div>

        {/* Enumerated Hardware Cards */}
        {devices.length > 0 && (
          <div style={{ marginBottom: '24px' }}>
            <h3 style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--accent-emerald)', textTransform: 'uppercase', marginBottom: '12px' }}>
              Detected ALSA Hardware Cards
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              {devices.map((dev, i) => (
                <div key={i} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '8px 12px', background: 'rgba(0, 0, 0, 0.3)', borderRadius: 'var(--radius-sm)', fontSize: '0.75rem', fontFamily: 'var(--font-mono)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <CheckCircle2 size={13} color="#10b981" />
                    <span>Card {dev.card_index}: {dev.name}</span>
                  </div>
                  <span style={{ color: 'var(--text-muted)' }}>{dev.identifier} ({dev.device_type})</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ONNX Model Architecture & Tensors */}
        <div>
          <h3 style={{ fontSize: '0.875rem', fontWeight: 600, color: '#c4b5fd', textTransform: 'uppercase', marginBottom: '12px' }}>
            Active ONNX Speech Enhancement Model
          </h3>
          <div style={{ padding: '12px', background: 'rgba(0, 0, 0, 0.3)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)', fontSize: '0.8125rem' }}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '10px', marginBottom: '10px' }}>
              <div>
                <span style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>Model Architecture:</span>
                <div style={{ fontWeight: 600 }}>{model?.model_name || 'ComplexCRN'}</div>
              </div>
              <div>
                <span style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>Quantization Precision:</span>
                <div style={{ fontWeight: 600, color: 'var(--accent-cyan)' }}>{model?.quantization || 'INT8 (ARM NEON SIMD)'}</div>
              </div>
              <div>
                <span style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>Execution Provider:</span>
                <div style={{ fontWeight: 600 }}>{model?.execution_provider || 'CPUExecutionProvider'}</div>
              </div>
              <div>
                <span style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>Intra-Op Worker Threads:</span>
                <div style={{ fontWeight: 600 }}>{model?.intra_op_threads || 2} Threads</div>
              </div>
            </div>

            {model?.input_shape && (
              <div style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: '8px', marginTop: '8px', fontSize: '0.7rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                <div>Input Tensors: {model.input_shape.join(' | ')}</div>
                <div style={{ marginTop: '2px' }}>Output Tensors: {model.output_shape.join(' | ')}</div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
