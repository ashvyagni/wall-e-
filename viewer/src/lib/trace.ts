import { traceEventSchema, type TraceEvent } from "@walle/contracts";

export type ParseTraceResult =
  | { ok: true; events: TraceEvent[] }
  | { ok: false; error: string; line: number };

/**
 * Parse a JSONL mission trace — one TraceEvent per line — validating every
 * line against the frozen v1.0 contract schema. Blank lines are skipped.
 */
export function parseTrace(text: string): ParseTraceResult {
  const events: TraceEvent[] = [];
  const lines = text.split("\n");

  for (let i = 0; i < lines.length; i += 1) {
    const trimmed = (lines[i] ?? "").trim();
    if (trimmed === "") continue;

    let json: unknown;
    try {
      json = JSON.parse(trimmed);
    } catch {
      return { ok: false, error: `Line ${i + 1}: invalid JSON — ${trimmed.slice(0, 80)}`, line: i + 1 };
    }

    const result = traceEventSchema.safeParse(json);
    if (!result.success) {
      const issue = result.error.issues[0];
      const path = issue?.path.join(".") ?? "";
      const detail = issue?.message ?? "schema violation";
      return {
        ok: false,
        error: `Line ${i + 1}: ${detail}${path ? ` (at ${path})` : ""}`,
        line: i + 1,
      };
    }
    events.push(result.data);
  }

  if (events.length === 0) {
    return { ok: false, error: "Trace contains no events.", line: 0 };
  }
  return { ok: true, events };
}
