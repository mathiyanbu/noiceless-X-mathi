import React, { useState } from 'react';
import { useTelemetry } from './hooks/useTelemetry';
import { Header } from './components/Header';
import { Controls } from './components/Controls';
import { LevelMeters } from './components/LevelMeters';
import { FusionStateCard } from './components/FusionStateCard';
import { LatencyBreakdown } from './components/LatencyBreakdown';
import { HardwareTelemetry } from './components/HardwareTelemetry';
import { DeviceModal } from './components/DeviceModal';
import { Terminal } from 'lucide-react';

export const App: React.FC = () => {
  const {
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
  } = useTelemetry();

  const [isDeviceModalOpen, setIsDeviceModalOpen] = useState(false);

  return (
    <div style={{ minHeight: '100vh', padding: '24px 20px' }}>
      <div style={{ maxWidth: '1440px', margin: '0 auto' }}>
        {/* Header Bar */}
        <Header
          health={health}
          model={model}
          isWsConnected={isWsConnected}
          runtimeConnected={runtimeConnected}
          runtimeState={runtimeState}
          onOpenDevices={() => setIsDeviceModalOpen(true)}
        />

        {/* Operator Controls Plane */}
        <Controls
          runtimeConnected={runtimeConnected}
          runtimeState={runtimeState}
          actionLoading={actionLoading}
          errorMessage={errorMessage}
          onStart={handleStart}
          onStop={handleStop}
          onBypass={handleBypass}
          onReset={handleReset}
        />

        {/* 2-Column Responsive Dashboard Grid */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(480px, 1fr))',
          gap: '20px',
        }}>
          {/* Left Column: Acoustic Levels & Phase 9 Fusion Engine */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            <LevelMeters
              telemetry={telemetry}
              runtimeConnected={runtimeConnected}
            />

            <FusionStateCard
              telemetry={telemetry}
              runtimeConnected={runtimeConnected}
            />
          </div>

          {/* Right Column: Stage Latencies / RTF & Hardware Telemetry */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            <LatencyBreakdown
              telemetry={telemetry}
              runtimeConnected={runtimeConnected}
            />

            <HardwareTelemetry
              system={system}
              telemetry={telemetry}
              runtimeConnected={runtimeConnected}
            />
          </div>
        </div>

        {/* Subsystem Architecture Guarantee Footer */}
        <footer style={{
          marginTop: '32px',
          padding: '16px 20px',
          borderRadius: 'var(--radius-md)',
          background: 'rgba(0, 0, 0, 0.4)',
          border: '1px solid var(--border-subtle)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '12px',
          fontSize: '0.75rem',
          color: 'var(--text-muted)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Terminal size={14} color="var(--accent-cyan)" />
            <span>
              Real-time dual-mic audio processing executes strictly within C++ ALSA threads under <code style={{ color: '#fff' }}>SCHED_FIFO</code> priorities.
            </span>
          </div>

          <div style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>
            Zero-Mock Telemetry Link: 20 Hz IPC Loopback
          </div>
        </footer>

        {/* ALSA & Hardware Device Inspection Modal */}
        <DeviceModal
          isOpen={isDeviceModalOpen}
          onClose={() => setIsDeviceModalOpen(false)}
          audioDevices={audioDevices}
          model={model}
        />
      </div>
    </div>
  );
};

export default App;
