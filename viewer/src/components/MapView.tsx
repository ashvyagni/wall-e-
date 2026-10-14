import type { ReactElement } from "react";
import type { MissionConfig, TraceEvent } from "@walle/contracts";
import { agentPath, discoveredClues, inferBounds } from "../lib/derive";

const UNIT = 32;

interface MapViewProps {
  events: TraceEvent[];
  currentIndex: number;
  config: MissionConfig | null;
}

function toSvgX(x: number, bounds: { minX: number }): number {
  return (x - bounds.minX + 0.5) * UNIT;
}

function toSvgY(y: number, bounds: { minY: number; rows: number }): number {
  return (bounds.rows - (y - bounds.minY) - 0.5) * UNIT;
}

export function MapView({ events, currentIndex, config }: MapViewProps) {
  const current = events[currentIndex];
  const bounds = inferBounds(events, config);
  const width = bounds.cols * UNIT;
  const height = bounds.rows * UNIT;

  const path = agentPath(events, currentIndex);
  const pathPoints = path.map(([x, y]) => `${toSvgX(x, bounds)},${toSvgY(y, bounds)}`).join(" ");

  const obstacles = config?.map.obstacles ?? [];
  const rechargeStations = config?.map.recharge_stations ?? [];
  const home = config?.map.home ?? null;

  const clues = discoveredClues(events, currentIndex);
  const belief = current?.belief ?? null;
  const position = current?.observation?.position ?? null;
  const bearing = current?.observation?.signal?.bearing_deg ?? null;

  const gridLines: ReactElement[] = [];
  for (let c = 0; c <= bounds.cols; c += 1) {
    gridLines.push(
      <line
        key={`v${c}`}
        x1={c * UNIT}
        y1={0}
        x2={c * UNIT}
        y2={height}
        stroke="var(--border)"
        strokeWidth={0.5}
      />,
    );
  }
  for (let r = 0; r <= bounds.rows; r += 1) {
    gridLines.push(
      <line
        key={`h${r}`}
        x1={0}
        y1={r * UNIT}
        x2={width}
        y2={r * UNIT}
        stroke="var(--border)"
        strokeWidth={0.5}
      />,
    );
  }

  // Bearing arrow: clockwise from north, +y is north in world space.
  const arrowLen = UNIT * 2.2;
  const arrowDx = position && bearing !== null ? Math.sin((bearing * Math.PI) / 180) * arrowLen : 0;
  const arrowDy = position && bearing !== null ? Math.cos((bearing * Math.PI) / 180) * arrowLen : 0;

  return (
    <svg
      className="map-svg"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={
        position
          ? `Grid map, agent at cell ${position[0]},${position[1]}`
          : "Grid map"
      }
    >
      <defs>
        <radialGradient id="belief-heat">
          <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.55" />
          <stop offset="55%" stopColor="var(--accent)" stopOpacity="0.18" />
          <stop offset="100%" stopColor="var(--accent)" stopOpacity="0" />
        </radialGradient>
        <filter id="agent-glow" x="-80%" y="-80%" width="260%" height="260%">
          <feGaussianBlur stdDeviation="3.2" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      <rect x={0} y={0} width={width} height={height} fill="var(--bg)" />
      {gridLines}

      {/* Belief heat spots */}
      {belief?.top_k.map(([cell, prob], i) => (
        <circle
          key={`heat-${i}-${cell[0]}-${cell[1]}`}
          cx={toSvgX(cell[0], bounds)}
          cy={toSvgY(cell[1], bounds)}
          r={UNIT * 0.95}
          fill="url(#belief-heat)"
          opacity={Math.min(1, Math.max(0.1, prob * 1.4))}
        />
      ))}

      {obstacles.map(([x, y]) => (
        <rect
          key={`obs-${x}-${y}`}
          x={toSvgX(x, bounds) - UNIT / 2}
          y={toSvgY(y, bounds) - UNIT / 2}
          width={UNIT}
          height={UNIT}
          fill="#1a2130"
          stroke="var(--border-strong)"
          strokeWidth={1}
          rx={3}
        />
      ))}

      {rechargeStations.map(([x, y]) => (
        <g key={`recharge-${x}-${y}`}>
          <rect
            x={toSvgX(x, bounds) - UNIT / 2 + 3}
            y={toSvgY(y, bounds) - UNIT / 2 + 3}
            width={UNIT - 6}
            height={UNIT - 6}
            rx={4}
            fill="var(--amber-dim)"
            stroke="var(--amber)"
            strokeWidth={1}
          />
          <text
            x={toSvgX(x, bounds)}
            y={toSvgY(y, bounds)}
            textAnchor="middle"
            dominantBaseline="central"
            fontSize={13}
            fill="var(--amber)"
            fontFamily="var(--mono)"
          >
            R
          </text>
        </g>
      ))}

      {home && (
        <g key={`home-${home[0]}-${home[1]}`}>
          <rect
            x={toSvgX(home[0], bounds) - UNIT / 2 + 3}
            y={toSvgY(home[1], bounds) - UNIT / 2 + 3}
            width={UNIT - 6}
            height={UNIT - 6}
            rx={4}
            fill="var(--success-dim)"
            stroke="var(--success)"
            strokeWidth={1}
          />
          <text
            x={toSvgX(home[0], bounds)}
            y={toSvgY(home[1], bounds)}
            textAnchor="middle"
            dominantBaseline="central"
            fontSize={12}
            fill="var(--success)"
            fontFamily="var(--mono)"
          >
            H
          </text>
        </g>
      )}

      {/* Agent path */}
      {pathPoints !== "" && (
        <polyline
          points={pathPoints}
          fill="none"
          stroke="var(--accent)"
          strokeWidth={2}
          strokeOpacity={0.75}
          strokeLinejoin="round"
          strokeLinecap="round"
        />
      )}

      {/* Discovered clue markers */}
      {clues.map((clue) => (
        <g key={`clue-${clue.clueId}`}>
          <circle
            cx={toSvgX(clue.pos[0], bounds)}
            cy={toSvgY(clue.pos[1], bounds)}
            r={5}
            fill="var(--amber)"
            stroke="#0b0d12"
            strokeWidth={1.5}
          />
          <text
            x={toSvgX(clue.pos[0], bounds) + 9}
            y={toSvgY(clue.pos[1], bounds) - 8}
            fontSize={9}
            fontFamily="var(--mono)"
            fill="var(--amber)"
          >
            {clue.kind.replace("_", " ")}
          </text>
        </g>
      ))}

      {/* Signal bearing arrow */}
      {position && bearing !== null && (
        <g aria-hidden="true">
          <line
            x1={toSvgX(position[0], bounds)}
            y1={toSvgY(position[1], bounds)}
            x2={toSvgX(position[0], bounds) + arrowDx}
            y2={toSvgY(position[1], bounds) - arrowDy}
            stroke="var(--amber)"
            strokeWidth={2}
            strokeDasharray="5 4"
            markerEnd="url(#arrowhead)"
          />
          <defs>
            <marker id="arrowhead" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
              <path d="M0,0 L7,3 L0,6 Z" fill="var(--amber)" />
            </marker>
          </defs>
        </g>
      )}

      {/* Agent marker */}
      {position && (
        <g filter="url(#agent-glow)">
          <circle
            data-testid="agent-marker"
            data-cell={`${position[0]},${position[1]}`}
            cx={toSvgX(position[0], bounds)}
            cy={toSvgY(position[1], bounds)}
            r={UNIT * 0.32}
            fill="var(--accent)"
            className="agent-glow"
            stroke="#eaffff"
            strokeWidth={1.5}
          >
            <title>{`Agent at [${position[0]}, ${position[1]}]`}</title>
          </circle>
        </g>
      )}
    </svg>
  );
}
