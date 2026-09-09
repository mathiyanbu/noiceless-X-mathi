import React from 'react';
import { Clock, CheckCircle, AlertCircle } from 'lucide-react';
import type { MetricsResponse } from '../types/telemetry';

interface LatencyBreakdownProps {
  telemetry: MetricsResponse | null;
  runtimeConnected: boolean;
}

export const LatencyBreakdown: React.FC<LatencyBreakdownProps> = ({
  telemetry,
  runtimeConnected,
}) => {
  const isAvailable = runtimeConnected && telemetry !== null;

  const latencies = isAvailable ? telemetry.latencies : null;
  const rtf = isAvailable ? telemetry.rtf : null;
  const endToEndMs = latencies ? latencies.end_to_end_latency_ms : null;
  const totalProcessingUs = latencies ? latencies.total_processing_us : null;

  // Stages array for visualization
  const stages = [
    { label: 'Capture Dequeue', us: latencies?.capture_us, color: '#38bdf8' },
    { label: 'Preprocessing (DC+HPF)', us: latencies?.preprocessing_us, color: '#00f0ff' },
    { label: 'STFT Analysis', us: latencies?.stft_us, color: '#818cf8' },
    { label: 'AI Speech Enhancer', us: latencies?.ai_inference_us, color: '#a78bfa' },
    { label: 'iSTFT Synthesis', us: latencies?.istft_us, color: '#c084fc' },
    { label: 'NLMS Adaptive Canceller', us: latencies?.nlms_us, color: '#34d399' },
    { label: 'Multi-Mode Fusion', us: latencies?.fusion_us, color: '#f59e0b' },
    { label: 'Playback Queue Push', us: latencies?.playback_queue_us, color: '#ec4899' },
  ];

  // Hop deadline is 5000 us (5ms at 16kHz with 80 samples)
  const HOP_DEADLINE_US = 5000.0;
  const budgetConsumedPct = totalProcessingUs ? Math.min(100, (totalProcessingUs / HOP_DEADLINE_US) * 100) : 0;

  return (
    <div className="glass-panel" style={{ padding: '20px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px', flexWrap: 'wrap', gap: '8px' }}>
        <h3 style={{ fontSize: '0.9375rem', fontWeight: 700, letterSpacing: '-0.01em', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Clock size={18} color="var(--accent-cyan)" />
          STAGE LATENCIES & REAL-TIME FACTOR
        </h3>

        {/* RTF Status Pill */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          padding: '4px 12px',
          borderRadius: 'var(--radius-full)',
          background: rtf !== null && rtf < 0.5 ? 'rgba(16, 185, 129, 0.12)' : 'rgba(245, 158, 11, 0.12)',
          border: `1px solid ${rtf !== null && rtf < 0.5 ? 'rgba(16, 185, 129, 0.3)' : 'rgba(245, 158, 11, 0.3)'}`,
          fontFamily: 'var(--font-mono)',
          fontSize: '0.75rem',
          fontWeight: 700,
          color: rtf !== null && rtf < 0.5 ? '#34d399' : '#fbbf24',
        }}>
          {rtf !== null && rtf < 0.5 ? <CheckCircle size={14} /> : <AlertCircle size={14} />}
          <span>RTF: {rtf !== null ? `${rtf.toFixed(3)}x` : 'N/A'}</span>
          <span style={{ fontSize: '0.65rem', opacity: 0.7 }}>(&lt;0.50x Target)</span>
        </div>
      </div>

      {/* Top Headline Latency Metrics */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '12px', marginBottom: '16px' }}>
        {/* End to End Latency */}
        <div style={{
          padding: '12px 14px',
          borderRadius: 'var(--radius-md)',
          background: 'rgba(0, 0, 0, 0.25)',
          border: '1px solid var(--border-subtle)',
        }}>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '4px' }}>End-to-End Latency</div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.375rem', fontWeight: 700, color: endToEndMs !== null ? 'var(--accent-cyan)' : 'var(--text-muted)' }}>
            {endToEndMs !== null ? `${endToEndMs.toFixed(2)} ms` : 'N/A'}
          </div>
          <div style={{ fontSize: '0.6875rem', color: 'var(--text-muted)' }}>Target: &lt; 10.0 ms budget</div>
        </div>

        {/* Total DSP Hop Processing */}
        <div style={{
          padding: '12px 14px',
          borderRadius: 'var(--radius-md)',
          background: 'rgba(0, 0, 0, 0.25)',
          border: '1px solid var(--border-subtle)',
        }}>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '4px' }}>Hop Processing Time</div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.375rem', fontWeight: 700, color: totalProcessingUs !== null ? '#c4b5fd' : 'var(--text-muted)' }}>
            {totalProcessingUs !== null ? `${totalProcessingUs.toFixed(0)} \u03BCs` : 'N/A'}
          </div>
          <div style={{ fontSize: '0.6875rem', color: 'var(--text-muted)' }}>
            Budget: {budgetConsumedPct ? `${budgetConsumedPct.toFixed(1)}% of 5000 \u03BCs` : 'N/A'}
          </div>
        </div>
      </div>

      {/* Stage Breakdown List */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '4px' }}>
          Detailed Pipeline Stages (&mu;s)
        </div>

        {stages.map((stage) => {
          const hasVal = stage.us !== null && stage.us !== undefined && !isNaN(stage.us);
          const stagePct = (hasVal && totalProcessingUs && totalProcessingUs > 0)
            ? Math.min(100, (stage.us! / totalProcessingUs) * 100)
            : 0;

          return (
            <div key={stage.label} style={{ fontSize: '0.75rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '3px' }}>
                <span style={{ color: 'var(--text-secondary)' }}>{stage.label}</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: hasVal ? stage.color : 'var(--text-muted)' }}>
                  {hasVal ? `${stage.us!.toFixed(1)} \u03BCs` : 'N/A'}
                </span>
              </div>
              <div style={{ height: '5px', borderRadius: 'var(--radius-full)', background: 'rgba(255, 255, 255, 0.05)', overflow: 'hidden' }}>
                {hasVal && (
                  <div
                    style={{
                      width: `${stagePct}%`,
                      height: '100%',
                      background: stage.color,
                      borderRadius: 'var(--radius-full)',
                      transition: 'width 0.1s ease',
                    }}
                  />
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
