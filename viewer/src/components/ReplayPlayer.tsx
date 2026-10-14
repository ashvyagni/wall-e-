import { useEffect, useMemo, useState } from "react";
import type { MissionConfig } from "@walle/contracts";
import { parseTrace } from "../lib/trace";
import { strategyMarkers, SUCCESS_TERMINALS } from "../lib/derive";
import { MapView } from "./MapView";
import { Timeline, type PlaybackSpeed } from "./Timeline";
import { SidePanel } from "./SidePanel";

interface ReplayPlayerProps {
  /** Raw JSONL trace text; every line is validated against traceEventSchema. */
  text: string;
  config?: MissionConfig | null;
  /** Starting event index (default 0). */
  initialIndex?: number;
}

const BASE_INTERVAL_MS = 550;

export function ReplayPlayer({ text, config = null, initialIndex = 0 }: ReplayPlayerProps) {
  const parsed = useMemo(() => parseTrace(text), [text]);
  const [index, setIndex] = useState(initialIndex);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<PlaybackSpeed>(1);

  useEffect(() => {
    setIndex(initialIndex);
    setPlaying(false);
  }, [text, initialIndex]);

  const events = parsed.ok ? parsed.events : [];
  const lastIndex = Math.max(0, events.length - 1);
  const markers = useMemo(() => strategyMarkers(events), [events]);

  useEffect(() => {
    if (!playing || events.length === 0) return;
    const interval = BASE_INTERVAL_MS / speed;
    const id = window.setInterval(() => {
      setIndex((i) => {
        if (i >= lastIndex) {
          setPlaying(false);
          return i;
        }
        return i + 1;
      });
    }, interval);
    return () => window.clearInterval(id);
  }, [playing, speed, events.length, lastIndex]);

  if (!parsed.ok) {
    return (
      <div className="error-state" role="alert" data-testid="trace-error">
        <h2>Trace failed validation</h2>
        <p>{parsed.error}</p>
        <p>Every line of a mission trace must be a valid TraceEvent under contract schema v1.0.</p>
      </div>
    );
  }

  const safeIndex = Math.min(index, lastIndex);
  const current = events[safeIndex];
  const maxBattery = current?.observation?.max_battery ?? config?.budget.battery ?? 1;
  const terminal = current?.terminal ?? null;
  const ticks = events.map((ev) => ev.tick);
  const atEnd = safeIndex >= lastIndex;

  const togglePlay = () => {
    if (!playing && safeIndex >= lastIndex) setIndex(0);
    setPlaying((p) => !p);
  };

  return (
    <div className="replay-main">
      <section className="card map-card" aria-label="Grid map">
        <MapView events={events} currentIndex={safeIndex} config={config} />
        <div className="decision-meta" style={{ marginTop: 12, alignItems: "center" }}>
          <span className="badge accent">
            event {safeIndex + 1} / {events.length}
          </span>
          <span className="badge">{current?.kind}</span>
          <span className="badge">agent {current?.agent_id}</span>
          {current?.action && (
            <span className="badge amber">
              {current.action.type}
              {current.action.direction ? `(${current.action.direction})` : ""}
              {current.action.position
                ? ` [${current.action.position[0]}, ${current.action.position[1]}]`
                : ""}
            </span>
          )}
        </div>
      </section>

      <Timeline
        index={safeIndex}
        count={events.length}
        ticks={ticks}
        markers={markers}
        playing={playing}
        speed={speed}
        terminal={terminal}
        battery={current?.battery ?? 0}
        maxBattery={maxBattery}
        onIndexChange={(i) => {
          setPlaying(false);
          setIndex(i);
        }}
        onTogglePlay={togglePlay}
        onSpeedChange={setSpeed}
      />

      {terminal !== null && atEnd && (
        <div
          className={`sr-only`}
          role="status"
          aria-live="polite"
        >
          Mission terminal: {SUCCESS_TERMINALS.includes(terminal) ? "success" : "failure"}, {terminal}
        </div>
      )}

      <SidePanel events={events} currentIndex={safeIndex} maxBattery={maxBattery} />
    </div>
  );
}
