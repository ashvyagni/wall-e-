import type { MissionConfig, TraceEvent, TerminalState } from "@walle/contracts";

export interface Bounds {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  cols: number;
  rows: number;
}

export interface Marker {
  index: number;
  tick: number;
  label: string;
}

export interface ClueMarker {
  clueId: string;
  kind: string;
  pos: [number, number];
  tick: number;
  description: string;
}

const MARKER_ACTIONS = ["scan", "recharge", "declare_found", "return_home"] as const;

/** Map bounds: prefer mission config; otherwise infer from observed positions. */
export function inferBounds(events: TraceEvent[], config: MissionConfig | null): Bounds {
  if (config) {
    return {
      minX: 0,
      minY: 0,
      maxX: config.map.width - 1,
      maxY: config.map.height - 1,
      cols: config.map.width,
      rows: config.map.height,
    };
  }
  let minX = 0;
  let minY = 0;
  let maxX = 0;
  let maxY = 0;
  for (const ev of events) {
    const cells = [
      ...(ev.observation?.position ? [ev.observation.position] : []),
      ...(ev.belief ? [ev.belief.map_pos, ...ev.belief.top_k.map(([p]) => p)] : []),
    ];
    for (const [x, y] of cells) {
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
    }
  }
  return { minX, minY, maxX, maxY, cols: maxX - minX + 1, rows: maxY - minY + 1 };
}

/** First-occurrence markers for strategy-shifting action types. */
export function strategyMarkers(events: TraceEvent[]): Marker[] {
  const seen = new Set<string>();
  const markers: Marker[] = [];
  events.forEach((ev, index) => {
    const actionType = ev.decision?.action_type ?? ev.action?.type;
    if (!actionType || actionType === "start") return;
    if (!(MARKER_ACTIONS as readonly string[]).includes(actionType)) return;
    if (seen.has(actionType)) return;
    seen.add(actionType);
    const label =
      actionType === "declare_found" ? "declare" : actionType === "return_home" ? "return" : actionType;
    markers.push({ index, tick: ev.tick, label });
  });
  return markers;
}

/** Agent path — observation positions up to and including `upToIndex`. */
export function agentPath(events: TraceEvent[], upToIndex: number): Array<[number, number]> {
  const path: Array<[number, number]> = [];
  for (let i = 0; i <= upToIndex && i < events.length; i += 1) {
    const pos = events[i]?.observation?.position;
    if (pos) path.push([pos[0], pos[1]]);
  }
  return path;
}

/** Discovered clues up to and including `upToIndex`. */
export function discoveredClues(events: TraceEvent[], upToIndex: number): ClueMarker[] {
  const clues = new Map<string, ClueMarker>();
  for (let i = 0; i <= upToIndex && i < events.length; i += 1) {
    for (const ev2 of events[i]?.observation?.events ?? []) {
      clues.set(ev2.clue_id, {
        clueId: ev2.clue_id,
        kind: ev2.kind,
        pos: [ev2.pos[0], ev2.pos[1]],
        tick: ev2.tick,
        description: ev2.description,
      });
    }
  }
  return [...clues.values()];
}

export function confidenceSeries(events: TraceEvent[]): Array<{ index: number; confidence: number }> {
  const series: Array<{ index: number; confidence: number }> = [];
  events.forEach((ev, index) => {
    if (ev.belief) series.push({ index, confidence: ev.belief.confidence });
  });
  return series;
}

export function lastTerminal(events: TraceEvent[]): TerminalState | null {
  for (let i = events.length - 1; i >= 0; i -= 1) {
    const terminal = events[i]?.terminal;
    if (terminal) return terminal;
  }
  return null;
}

export const SUCCESS_TERMINALS: readonly TerminalState[] = ["EVE_LOCATED_SUCCESS"];
