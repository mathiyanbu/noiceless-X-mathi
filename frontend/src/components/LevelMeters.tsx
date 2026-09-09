import React from 'react';
import { Mic, Radio, Volume2, HelpCircle } from 'lucide-react';
import type { MetricsResponse } from '../types/telemetry';

interface LevelMetersProps {
  telemetry: MetricsResponse | null;
  runtimeConnected: boolean;
}

interface MeterRowProps {
  label: string;
  sublabel: string;
  icon: React.ReactNode;
  levelDbfs: number | null | undefined;
  accentColor: string;
}

const MeterRow: React.FC<MeterRowProps> = ({
  label,
  sublabel,
  icon,
  levelDbfs,
  accentColor,
}) => {
  const isAvailable = levelDbfs !== null && levelDbfs !== undefined && !isNaN(levelDbfs);
  // Map dBFS from -60 dB to 0 dB to percentage 0% to 100%
  const pct = isAvailable ? Math.max(0, Math.min(100, ((levelDbfs + 60) / 60) * 100)) : 0;

  return (
    <div style={{ marginBottom: '14px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px', fontSize: '0.8125rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {icon}
          <div>
            <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{label}</span>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginLeft: '6px' }}>({sublabel})</span>
          </div>
        </div>
        <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, fontSize: '0.75rem', color: isAvailable ? accentColor : 'var(--text-muted)' }}>
          {isAvailable ? `${levelDbfs.toFixed(1)} dBFS` : 'N/A'}
        </div>
      </div>

      <div className="meter-track">
        {isAvailable ? (
          <div
            className="meter-fill"
            style={{
              width: `${pct}%`,
              background: `linear-gradient(90deg, #10b981 0%, #3b82f6 65%, #f59e0b 85%, #ef4444 100%)`,
            }}
          />
        ) : (
          <div className="meter-na">
            <span>MEASUREMENT N/A (Audio runs strictly in ALSA C++ threads)</span>
          </div>
        )}
      </div>
    </div>
  );
};

export const LevelMeters: React.FC<LevelMetersProps> = ({ telemetry, runtimeConnected }) => {
  const inputSnr = telemetry?.input_snr_db;
  const outputSnr = telemetry?.output_snr_db;

  return (
    <div className="glass-panel" style={{ padding: '20px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
        <h3 style={{ fontSize: '0.9375rem', fontWeight: 700, letterSpacing: '-0.01em', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Volume2 size={18} color="var(--accent-cyan)" />
          ACOUSTIC LEVEL & SNR GAIN
        </h3>
        <span style={{ fontSize: '0.7rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
          dBFS CALIBRATION
        </span>
      </div>

      {/* Primary Mic (Headphone Mic) */}
      <MeterRow
        label="Primary Headphone Mic"
        sublabel="Speech + Local Ambient"
        icon={<Mic size={15} color="var(--accent-cyan)" />}
        levelDbfs={runtimeConnected ? telemetry?.primary_level_dbfs : null}
        accentColor="var(--accent-cyan)"
      />

      {/* Reference Error Mic */}
      <MeterRow
        label="Reference Error Mic"
        sublabel="Acoustic Noise Corridor"
        icon={<Radio size={15} color="var(--accent-purple)" />}
        levelDbfs={runtimeConnected ? telemetry?.reference_level_dbfs : null}
        accentColor="var(--accent-purple)"
      />

      {/* Enhanced Playback Output */}
      <MeterRow
        label="Enhanced Playback DAC"
        sublabel="Fused Signal Output"
        icon={<Volume2 size={15} color="var(--accent-emerald)" />}
        levelDbfs={runtimeConnected ? telemetry?.output_level_dbfs : null}
        accentColor="var(--accent-emerald)"
      />

      {/* Signal-to-Noise Ratio (SNR) Metrics Card */}
      <div style={{
        marginTop: '18px',
        padding: '14px 16px',
        borderRadius: 'var(--radius-md)',
        background: 'rgba(0, 0, 0, 0.3)',
        border: '1px solid var(--border-subtle)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase' }}>
            SNR Metrics (Phase 13 Engine)
          </span>
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
            <HelpCircle size={12} />
            <span>Honest Evaluation</span>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px' }}>
          {/* Input SNR */}
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '2px' }}>Input SNR</div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1rem', fontWeight: 700, color: inputSnr !== null && inputSnr !== undefined ? '#38bdf8' : 'var(--text-muted)' }}>
              {inputSnr !== null && inputSnr !== undefined ? `${inputSnr.toFixed(1)} dB` : 'N/A'}
            </div>
          </div>

          {/* Output SNR */}
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '2px' }}>Output SNR</div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1rem', fontWeight: 700, color: outputSnr !== null && outputSnr !== undefined ? '#34d399' : 'var(--text-muted)' }}>
              {outputSnr !== null && outputSnr !== undefined ? `${outputSnr.toFixed(1)} dB` : 'N/A'}
            </div>
          </div>

          {/* SNR Gain Improvement */}
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '2px' }}>Net SNR Gain</div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1rem', fontWeight: 700, color: (outputSnr && inputSnr) ? '#a78bfa' : 'var(--text-muted)' }}>
              {(outputSnr && inputSnr) ? `+${(outputSnr - inputSnr).toFixed(1)} dB` : 'N/A'}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
