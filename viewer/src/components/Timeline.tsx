import type { TerminalState } from "@walle/contracts";
import type { Marker } from "../lib/derive";

export type PlaybackSpeed = 1 | 2 | 4;

interface TimelineProps {
  index: number;
  count: number;
  ticks: number[];
  markers: Marker[];
  playing: boolean;
  speed: PlaybackSpeed;
  terminal: TerminalState | null;
  battery: number;
  maxBattery: number;
  onIndexChange: (index: number) => void;
  onTogglePlay: () => void;
  onSpeedChange: (speed: PlaybackSpeed) => void;
}

const FAILURE_TERMINALS = new Set<TerminalState>([
  "EVE_LOCATE_FAILED",
  "RETURNED_NO_EVIDENCE",
  "BATTERY_EXHAUSTED_ABORTED",
  "TIMEOUT_ABORTED",
  "INVALID_ACTION_ABORTED",
]);

export function Timeline({
  index,
  count,
  ticks,
  markers,
  playing,
  speed,
  terminal,
  battery,
  maxBattery,
  onIndexChange,
  onTogglePlay,
  onSpeedChange,
}: TimelineProps) {
  const currentTick = ticks[index] ?? 0;
  const lastTick = ticks[count - 1] ?? 0;
  const pct = maxBattery > 0 ? (battery / maxBattery) * 100 : 0;
  const batteryClass = pct <= 15 ? "critical" : pct <= 35 ? "low" : "";
  const atEnd = index >= count - 1;

  return (
    <section className="card timeline-card" aria-label="Replay timeline">
      <div className="timeline-controls">
        <button
          type="button"
          onClick={() => onIndexChange(Math.max(0, index - 1))}
          disabled={index === 0}
          aria-label="Step back one event"
        >
          ⏮ Step
        </button>
        <button type="button" onClick={onTogglePlay} aria-label={playing ? "Pause replay" : "Play replay"}>
          {playing ? "⏸ Pause" : "▶ Play"}
        </button>
        <button
          type="button"
          onClick={() => onIndexChange(Math.min(count - 1, index + 1))}
          disabled={atEnd}
          aria-label="Step forward one event"
        >
          Step ⏭
        </button>

        <span className="tick-readout" aria-live="off">
          tick {currentTick} / {lastTick}
        </span>

        <div className="speed-group" role="group" aria-label="Playback speed">
          {([1, 2, 4] as const).map((s) => (
            <button
              key={s}
              type="button"
              aria-pressed={speed === s}
              onClick={() => onSpeedChange(s)}
            >
              {s}x
            </button>
          ))}
        </div>

        <div
          className="battery-track"
          role="meter"
          aria-label="Battery"
          aria-valuenow={battery}
          aria-valuemin={0}
          aria-valuemax={maxBattery}
        >
          <div className={`battery-fill ${batteryClass}`} style={{ width: `${pct}%` }} />
        </div>
        <span className="battery-label">
          {battery} / {maxBattery}
        </span>
      </div>

      <div className="scrubber-wrap">
        <div className="marker-track" aria-hidden="true">
          {markers.map((m) => {
            const left = count > 1 ? (m.index / (count - 1)) * 100 : 0;
            return (
              <div
                key={`${m.label}-${m.index}`}
                className="timeline-marker"
                style={{ left: `${left}%` }}
                title={`${m.label} at tick ${m.tick}`}
              >
                <span className="dot" />
                <span className="label">{m.label}</span>
              </div>
            );
          })}
        </div>
        <input
          className="scrubber"
          type="range"
          min={0}
          max={Math.max(0, count - 1)}
          step={1}
          value={index}
          onChange={(e) => onIndexChange(Number(e.target.value))}
          aria-label="Mission timeline scrubber"
          aria-valuetext={`Event ${index + 1} of ${count}, tick ${currentTick}`}
        />
      </div>

      {terminal !== null && atEnd && (
        <div
          className={`terminal-banner ${FAILURE_TERMINALS.has(terminal) ? "failure" : "success"}`}
          role="status"
        >
          <span>{FAILURE_TERMINALS.has(terminal) ? "MISSION ENDED" : "MISSION COMPLETE"}</span>
          <strong>{terminal}</strong>
        </div>
      )}
    </section>
  );
}
