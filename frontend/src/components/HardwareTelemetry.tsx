import React from 'react';
import { Cpu, Thermometer, HardDrive, AlertOctagon, Layers, ArrowDownUp } from 'lucide-react';
import type { SystemResponse, MetricsResponse } from '../types/telemetry';

interface HardwareTelemetryProps {
  system: SystemResponse | null;
  telemetry: MetricsResponse | null;
  runtimeConnected: boolean;
}

export const HardwareTelemetry: React.FC<HardwareTelemetryProps> = ({
  system,
  telemetry,
  runtimeConnected,
}) => {
  const cpu = system?.cpu;
  const tempAvailable = system?.temperature_available;
  const tempC = tempAvailable ? system?.temperature_c : null;
  const memory = system?.memory;

  const alsaXruns = (runtimeConnected && telemetry) ? telemetry.alsa_xruns : null;
  const totalXruns = alsaXruns ? alsaXruns.primary + alsaXruns.reference + alsaXruns.playback : null;
  const droppedFrames = (runtimeConnected && telemetry) ? telemetry.dropped_frames : null;
  const processedFrames = (runtimeConnected && telemetry) ? telemetry.processed_frames : null;
  const driftMs = (runtimeConnected && telemetry) ? telemetry.drift_ms : null;
  const driftWarning = (runtimeConnected && telemetry) ? telemetry.drift_warning : false;

  return (
    <div className="glass-panel" style={{ padding: '20px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
        <h3 style={{ fontSize: '0.9375rem', fontWeight: 700, letterSpacing: '-0.01em', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <HardDrive size={18} color="var(--accent-cyan)" />
          HARDWARE TELEMETRY & XRUN MONITORS
        </h3>
        <span style={{ fontSize: '0.7rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
          SOURCE: {system?.source ? system.source.toUpperCase() : 'KERNEL'}
        </span>
      </div>

      {/* Hardware Stat Cards (CPU, Temp, Memory, XRUNs) */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px', marginBottom: '16px' }}>
        {/* CPU Overall */}
        <div style={{
          padding: '12px',
          borderRadius: 'var(--radius-md)',
          background: 'rgba(0, 0, 0, 0.25)',
          border: '1px solid var(--border-subtle)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '4px' }}>
            <Cpu size={14} color="var(--accent-cyan)" />
            <span>CPU Overall</span>
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.25rem', fontWeight: 700, color: cpu ? 'var(--text-primary)' : 'var(--text-muted)' }}>
            {cpu ? `${cpu.overall_pct.toFixed(1)}%` : 'N/A'}
          </div>
        </div>

        {/* Temperature */}
        <div style={{
          padding: '12px',
          borderRadius: 'var(--radius-md)',
          background: tempC && tempC > 75 ? 'rgba(239, 68, 68, 0.1)' : 'rgba(0, 0, 0, 0.25)',
          border: `1px solid ${tempC && tempC > 75 ? 'rgba(239, 68, 68, 0.3)' : 'var(--border-subtle)'}`,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '4px' }}>
            <Thermometer size={14} color={tempC && tempC > 75 ? '#ef4444' : '#f59e0b'} />
            <span>CPU Temp</span>
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.25rem', fontWeight: 700, color: tempC !== null ? (tempC > 75 ? '#f87171' : '#fbbf24') : 'var(--text-muted)' }}>
            {tempC !== null ? `${tempC.toFixed(1)} \u00B0C` : 'N/A'}
          </div>
        </div>

        {/* Memory */}
        <div style={{
          padding: '12px',
          borderRadius: 'var(--radius-md)',
          background: 'rgba(0, 0, 0, 0.25)',
          border: '1px solid var(--border-subtle)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '4px' }}>
            <HardDrive size={14} color="var(--accent-purple)" />
            <span>Memory RAM</span>
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1rem', fontWeight: 700, color: memory ? 'var(--text-primary)' : 'var(--text-muted)' }}>
            {memory ? `${memory.used_mb.toFixed(0)} MB` : 'N/A'}
          </div>
          <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>
            {memory ? `of ${memory.total_mb.toFixed(0)} MB` : 'N/A'}
          </div>
        </div>

        {/* ALSA Total XRUNs */}
        <div style={{
          padding: '12px',
          borderRadius: 'var(--radius-md)',
          background: totalXruns && totalXruns > 0 ? 'rgba(239, 68, 68, 0.1)' : 'rgba(0, 0, 0, 0.25)',
          border: `1px solid ${totalXruns && totalXruns > 0 ? 'rgba(239, 68, 68, 0.3)' : 'var(--border-subtle)'}`,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '4px' }}>
            <AlertOctagon size={14} color={totalXruns && totalXruns > 0 ? '#ef4444' : '#10b981'} />
            <span>ALSA XRUNs</span>
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.25rem', fontWeight: 700, color: totalXruns !== null ? (totalXruns > 0 ? '#f87171' : '#34d399') : 'var(--text-muted)' }}>
            {totalXruns !== null ? totalXruns : 'N/A'}
          </div>
        </div>
      </div>

      {/* Per-Core CPU Bars (4 Raspberry Pi Cores) */}
      {cpu && cpu.cores_pct && cpu.cores_pct.length > 0 && (
        <div style={{ marginBottom: '16px' }}>
          <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '8px', textTransform: 'uppercase' }}>
            Quad-Core Utilization (ARM Cortex)
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: `repeat(${cpu.cores_pct.length}, 1fr)`, gap: '8px' }}>
            {cpu.cores_pct.map((pct, idx) => (
              <div key={idx} style={{ padding: '8px', borderRadius: 'var(--radius-sm)', background: 'rgba(255, 255, 255, 0.02)', border: '1px solid var(--border-subtle)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.6875rem', marginBottom: '4px', fontFamily: 'var(--font-mono)' }}>
                  <span style={{ color: 'var(--text-muted)' }}>C{idx}</span>
                  <span style={{ color: 'var(--accent-cyan)' }}>{pct.toFixed(0)}%</span>
                </div>
                <div style={{ height: '4px', background: 'rgba(255, 255, 255, 0.06)', borderRadius: 'var(--radius-full)', overflow: 'hidden' }}>
                  <div style={{ width: `${Math.min(100, pct)}%`, height: '100%', background: 'linear-gradient(90deg, #38bdf8, #00f0ff)' }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Audio Stream Health: Drift, Frames, Dropped */}
      <div style={{
        padding: '12px 14px',
        borderRadius: 'var(--radius-md)',
        background: 'rgba(0, 0, 0, 0.25)',
        border: '1px solid var(--border-subtle)',
      }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px', textAlign: 'center' }}>
          {/* Dual-Mic Clock Drift */}
          <div>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '4px', fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '2px' }}>
              <ArrowDownUp size={12} />
              <span>Mic Clock Drift</span>
            </div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.9375rem', fontWeight: 700, color: driftWarning ? '#ef4444' : (driftMs !== null ? 'var(--text-primary)' : 'var(--text-muted)') }}>
              {driftMs !== null ? `${driftMs.toFixed(2)} ms` : 'N/A'}
            </div>
            <div style={{ fontSize: '0.65rem', color: driftWarning ? '#f87171' : 'var(--text-muted)' }}>
              {driftWarning ? 'DRIFT ALERT' : (driftMs !== null ? 'Sync OK' : 'N/A')}
            </div>
          </div>

          {/* Processed Audio Frames */}
          <div>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '4px', fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '2px' }}>
              <Layers size={12} />
              <span>Processed Frames</span>
            </div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.9375rem', fontWeight: 700, color: processedFrames !== null ? '#38bdf8' : 'var(--text-muted)' }}>
              {processedFrames !== null ? processedFrames.toLocaleString() : 'N/A'}
            </div>
            <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>
              {processedFrames !== null ? '5ms Hop Frames' : 'N/A'}
            </div>
          </div>

          {/* Dropped Frames */}
          <div>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '4px', fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '2px' }}>
              <AlertOctagon size={12} />
              <span>Dropped Frames</span>
            </div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.9375rem', fontWeight: 700, color: droppedFrames !== null ? (droppedFrames > 0 ? '#ef4444' : '#10b981') : 'var(--text-muted)' }}>
              {droppedFrames !== null ? droppedFrames : 'N/A'}
            </div>
            <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>
              {droppedFrames !== null ? (droppedFrames === 0 ? 'Zero Loss' : 'Buffer Overflow') : 'N/A'}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
