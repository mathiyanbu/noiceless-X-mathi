import React from 'react';
import { Cpu, Zap, Radio, Shield, Sliders } from 'lucide-react';
import type { MetricsResponse } from '../types/telemetry';

interface FusionStateCardProps {
  telemetry: MetricsResponse | null;
  runtimeConnected: boolean;
}

export const FusionStateCard: React.FC<FusionStateCardProps> = ({
  telemetry,
  runtimeConnected,
}) => {
  const isAvailable = runtimeConnected && telemetry !== null;

  const mode = isAvailable ? telemetry.fusion_mode : 'N/A';
  const lambda = isAvailable ? telemetry.current_lambda : null;
  const aiConfidence = isAvailable ? telemetry.ai_confidence : null;
  const impulseProb = isAvailable ? telemetry.impulse_probability : null;
  const impulseGain = isAvailable ? telemetry.impulse_envelope_gain : null;
  const vadProb = isAvailable ? telemetry.vad_probability : null;

  // Mode badge color
  const getModeColor = (m: string) => {
    switch (m.toUpperCase()) {
      case 'NORMAL': return { bg: 'rgba(16, 185, 129, 0.15)', text: '#34d399', border: 'rgba(16, 185, 129, 0.3)' };
      case 'LOW_CONFIDENCE': return { bg: 'rgba(245, 158, 11, 0.15)', text: '#fbbf24', border: 'rgba(245, 158, 11, 0.3)' };
      case 'IMPULSE': return { bg: 'rgba(239, 68, 68, 0.15)', text: '#f87171', border: 'rgba(239, 68, 68, 0.3)' };
      case 'DEGRADED': return { bg: 'rgba(239, 68, 68, 0.2)', text: '#fca5a5', border: 'rgba(239, 68, 68, 0.4)' };
      case 'NLMS_FAULT': return { bg: 'rgba(245, 158, 11, 0.2)', text: '#fde68a', border: 'rgba(245, 158, 11, 0.4)' };
      case 'BYPASS': return { bg: 'rgba(139, 92, 246, 0.15)', text: '#c4b5fd', border: 'rgba(139, 92, 246, 0.3)' };
      default: return { bg: 'rgba(255, 255, 255, 0.05)', text: 'var(--text-muted)', border: 'var(--border-subtle)' };
    }
  };

  const modeStyle = getModeColor(mode);

  return (
    <div className="glass-panel" style={{ padding: '20px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '18px' }}>
        <h3 style={{ fontSize: '0.9375rem', fontWeight: 700, letterSpacing: '-0.01em', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Sliders size={18} color="var(--accent-purple)" />
          PHASE 9 DUAL-MIC FUSION CONTROLLER
        </h3>

        {/* Current State Machine Mode */}
        <div style={{
          padding: '4px 12px',
          borderRadius: 'var(--radius-full)',
          background: modeStyle.bg,
          color: modeStyle.text,
          border: `1px solid ${modeStyle.border}`,
          fontSize: '0.75rem',
          fontWeight: 700,
          fontFamily: 'var(--font-mono)',
        }}>
          MODE: {mode}
        </div>
      </div>

      {/* Dynamic Mixing Lambda Slider / Visualizer */}
      <div style={{
        padding: '14px 16px',
        borderRadius: 'var(--radius-md)',
        background: 'rgba(0, 0, 0, 0.25)',
        border: '1px solid var(--border-subtle)',
        marginBottom: '16px',
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px', fontSize: '0.8125rem' }}>
          <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
            Dynamic Fusion Weight (&lambda;)
          </span>
          <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: lambda !== null ? 'var(--accent-cyan)' : 'var(--text-muted)' }}>
            {lambda !== null ? `\u03BB = ${lambda.toFixed(3)}` : 'N/A'}
          </span>
        </div>

        {/* Dual branch ratio meter */}
        <div style={{ height: '14px', borderRadius: 'var(--radius-sm)', background: 'rgba(255, 255, 255, 0.06)', position: 'relative', overflow: 'hidden' }}>
          {lambda !== null ? (
            <div style={{ display: 'flex', width: '100%', height: '100%' }}>
              {/* AI Branch Weight */}
              <div style={{
                width: `${lambda * 100}%`,
                background: 'linear-gradient(90deg, #8b5cf6, #00f0ff)',
                transition: 'width 0.1s ease',
              }} />
              {/* NLMS Branch Weight */}
              <div style={{
                width: `${(1 - lambda) * 100}%`,
                background: 'linear-gradient(90deg, #3b82f6, #10b981)',
                transition: 'width 0.1s ease',
              }} />
            </div>
          ) : (
            <div className="meter-na" style={{ fontSize: '0.65rem' }}>N/A</div>
          )}
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '6px', fontSize: '0.7rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
          <span>AI Branch: {lambda !== null ? `${(lambda * 100).toFixed(0)}%` : 'N/A'}</span>
          <span>NLMS Residual: {lambda !== null ? `${((1 - lambda) * 100).toFixed(0)}%` : 'N/A'}</span>
        </div>
      </div>

      {/* Grid of 4 Core Signal Indicators */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '12px' }}>
        {/* AI Confidence */}
        <div style={{
          padding: '12px 14px',
          borderRadius: 'var(--radius-md)',
          background: 'rgba(255, 255, 255, 0.02)',
          border: '1px solid var(--border-subtle)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
            <Cpu size={14} color="var(--accent-purple)" />
            <span>AI Confidence</span>
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.25rem', fontWeight: 700, color: aiConfidence !== null ? '#c4b5fd' : 'var(--text-muted)' }}>
            {aiConfidence !== null ? `${(aiConfidence * 100).toFixed(1)}%` : 'N/A'}
          </div>
        </div>

        {/* Impulse Noise Probability */}
        <div style={{
          padding: '12px 14px',
          borderRadius: 'var(--radius-md)',
          background: impulseProb && impulseProb > 0.5 ? 'rgba(239, 68, 68, 0.1)' : 'rgba(255, 255, 255, 0.02)',
          border: `1px solid ${impulseProb && impulseProb > 0.5 ? 'rgba(239, 68, 68, 0.3)' : 'var(--border-subtle)'}`,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
            <Zap size={14} color={impulseProb && impulseProb > 0.5 ? '#ef4444' : '#f59e0b'} />
            <span>Impulse Probability</span>
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.25rem', fontWeight: 700, color: impulseProb !== null ? (impulseProb > 0.5 ? '#f87171' : '#fde68a') : 'var(--text-muted)' }}>
            {impulseProb !== null ? `${(impulseProb * 100).toFixed(1)}%` : 'N/A'}
          </div>
        </div>

        {/* Impulse Protection Gain Envelope */}
        <div style={{
          padding: '12px 14px',
          borderRadius: 'var(--radius-md)',
          background: 'rgba(255, 255, 255, 0.02)',
          border: '1px solid var(--border-subtle)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
            <Shield size={14} color="var(--accent-emerald)" />
            <span>Protection Gain (g_I)</span>
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.25rem', fontWeight: 700, color: impulseGain !== null ? '#34d399' : 'var(--text-muted)' }}>
            {impulseGain !== null ? `${impulseGain.toFixed(2)}x` : 'N/A'}
          </div>
        </div>

        {/* Voice Activity Detection (VAD) */}
        <div style={{
          padding: '12px 14px',
          borderRadius: 'var(--radius-md)',
          background: 'rgba(255, 255, 255, 0.02)',
          border: '1px solid var(--border-subtle)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
            <Radio size={14} color="var(--accent-cyan)" />
            <span>VAD Speech Prob</span>
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.25rem', fontWeight: 700, color: vadProb !== null ? '#38bdf8' : 'var(--text-muted)' }}>
            {vadProb !== null ? `${(vadProb * 100).toFixed(1)}%` : 'N/A'}
          </div>
        </div>
      </div>
    </div>
  );
};
