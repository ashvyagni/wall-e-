import type { TraceEvent } from "@walle/contracts";
import { confidenceSeries } from "../lib/derive";

interface SidePanelProps {
  events: TraceEvent[];
  currentIndex: number;
  maxBattery: number;
}

const SPARK_W = 280;
const SPARK_H = 44;

export function SidePanel({ events, currentIndex, maxBattery }: SidePanelProps) {
  const current = events[currentIndex];
  const observation = current?.observation ?? null;
  const belief = current?.belief ?? null;
  const decision = current?.decision ?? null;
  const battery = current?.battery ?? 0;
  const pct = maxBattery > 0 ? (battery / maxBattery) * 100 : 0;
  const batteryClass = pct <= 15 ? "critical" : pct <= 35 ? "low" : "";

  const series = confidenceSeries(events);
  const sparkPoints = series
    .map(({ index, confidence }) => {
      const x = series.length > 1 ? (index / (events.length - 1)) * SPARK_W : 0;
      const y = SPARK_H - 4 - confidence * (SPARK_H - 8);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  const currentConf = belief?.confidence ?? 0;
  const currentX = series.length > 1 ? (currentIndex / (events.length - 1)) * SPARK_W : 0;
  const currentY = SPARK_H - 4 - currentConf * (SPARK_H - 8);

  return (
    <aside className="replay-side" aria-label="Telemetry side panel">
      <section className="card">
        <h2 className="card-title">Telemetry</h2>
        <div className="stat-grid">
          <div className="stat">
            <span className="label">Tick</span>
            <span className="value" data-testid="tick-value">
              {current?.tick ?? 0}
            </span>
          </div>
          <div className="stat">
            <span className="label">Confidence</span>
            <span className="value" data-testid="confidence-value">
              {belief ? belief.confidence.toFixed(2) : "—"}
            </span>
          </div>
          <div className="stat">
            <span className="label">Battery</span>
            <span className="value small" data-testid="battery-value">
              {battery} / {maxBattery}
            </span>
            <div className={`battery-fill ${batteryClass}`} style={{ position: "static", width: `${pct}%`, height: 6, borderRadius: 999, marginTop: 4 }} />
          </div>
          <div className="stat">
            <span className="label">Entropy</span>
            <span className="value small" data-testid="entropy-value">
              {belief ? `${belief.entropy_bits.toFixed(2)} bits` : "—"}
            </span>
          </div>
        </div>
        <div>
          <span className="label" style={{ fontSize: 10, color: "var(--text-faint)", letterSpacing: "0.1em", textTransform: "uppercase" }}>
            Confidence over time
          </span>
          <svg
            className="sparkline"
            viewBox={`0 0 ${SPARK_W} ${SPARK_H}`}
            preserveAspectRatio="none"
            role="img"
            aria-label={`Belief confidence sparkline, currently ${currentConf.toFixed(2)}`}
          >
            <polyline
              points={sparkPoints}
              fill="none"
              stroke="var(--accent)"
              strokeWidth={1.5}
              strokeOpacity={0.8}
            />
            <circle cx={currentX} cy={currentY} r={3} fill="var(--accent)" />
          </svg>
        </div>
      </section>

      <section className="card">
        <h2 className="card-title">Decision</h2>
        {decision ? (
          <>
            <div className="decision-meta">
              <span className="badge accent">{decision.action_type}</span>
              <span className="badge">planner: {decision.planner}</span>
            </div>
            <p className="decision-reason" data-testid="decision-reason">
              {decision.reason}
            </p>
            {decision.options_considered.length > 0 && (
              <p className="decision-reason" style={{ fontFamily: "var(--mono)", fontSize: 11 }}>
                options: {decision.options_considered.join(", ")}
              </p>
            )}
          </>
        ) : (
          <p className="decision-reason">No decision recorded at this tick.</p>
        )}
      </section>

      <section className="card">
        <h2 className="card-title">Last observation</h2>
        {observation ? (
          <div className="obs-rows">
            <div className="obs-row">
              <span className="k">Position</span>
              <span className="v">
                [{observation.position[0]}, {observation.position[1]}]
              </span>
            </div>
            <div className="obs-row">
              <span className="k">Signal strength</span>
              <span className="v" data-testid="signal-strength">
                {observation.signal ? observation.signal.strength.toFixed(2) : "no reading"}
              </span>
            </div>
            <div className="obs-row">
              <span className="k">Signal bearing</span>
              <span className="v" data-testid="signal-bearing">
                {observation.signal ? `${observation.signal.bearing_deg.toFixed(1)}°` : "—"}
              </span>
            </div>
            <div className="obs-row">
              <span className="k">Visible cells</span>
              <span className="v" data-testid="visible-cells">
                {observation.visible_cells.length}
              </span>
            </div>
            <div className="obs-row">
              <span className="k">Events</span>
              <span className="v" data-testid="obs-events">
                {observation.events.length === 0
                  ? "none"
                  : observation.events.map((e) => e.type).join(", ")}
              </span>
            </div>
            <div className="obs-row">
              <span className="k">Last action valid</span>
              <span className="v">{observation.last_action_valid ? "yes" : "no"}</span>
            </div>
          </div>
        ) : (
          <p className="decision-reason">No observation at this tick.</p>
        )}
      </section>
    </aside>
  );
}
