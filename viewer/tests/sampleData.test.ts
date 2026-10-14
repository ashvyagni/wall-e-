import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { missionConfigSchema, missionReportSchema, traceEventSchema } from "@walle/contracts";
import { describe, expect, it } from "vitest";
import { parseTrace } from "../src/lib/trace";
import { strategyMarkers } from "../src/lib/derive";

const samplePath = resolve(process.cwd(), "public/samples/sample_basic.jsonl");
const reportPath = resolve(process.cwd(), "public/samples/sample_report.json");
const configPath = resolve(process.cwd(), "public/samples/sample_config.json");

const sampleText = readFileSync(samplePath, "utf8");
const reportJson = JSON.parse(readFileSync(reportPath, "utf8")) as unknown;
const configJson = JSON.parse(readFileSync(configPath, "utf8")) as unknown;

const lines = sampleText.trim().split("\n");

describe("sample_basic.jsonl", () => {
  it("has one event per line and every line validates against traceEventSchema", () => {
    expect(lines.length).toBe(41);
    lines.forEach((line, i) => {
      const result = traceEventSchema.safeParse(JSON.parse(line));
      if (!result.success) {
        throw new Error(`line ${i + 1} invalid: ${JSON.stringify(result.error.issues[0])}`);
      }
      expect(result.success).toBe(true);
    });
  });

  it("parses as a whole via parseTrace", () => {
    const parsed = parseTrace(sampleText);
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;
    expect(parsed.events.length).toBe(41);
    expect(parsed.events[0]?.tick).toBe(0);
    expect(parsed.events[40]?.tick).toBe(40);
  });

  it("tells the required story: signal rises 0.12→0.78, confidence 0.05→0.62, entropy 9.1→5.7", () => {
    const parsed = parseTrace(sampleText);
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;
    const first = parsed.events[0];
    const last = parsed.events.at(-1);
    if (!first || !last) throw new Error("unexpected empty trace");
    expect(first?.observation?.signal?.strength).toBe(0.12);
    expect(last?.observation?.signal?.strength).toBe(0.78);
    expect(first?.belief?.confidence).toBe(0.05);
    expect(last?.belief?.confidence).toBe(0.62);
    expect(first?.belief?.entropy_bits).toBeCloseTo(9.1, 5);
    expect(last?.belief?.entropy_bits).toBeCloseTo(5.7, 5);

    let prevSignal = -1;
    let prevConf = -1;
    let prevEntropy = Infinity;
    for (const ev of parsed.events) {
      const s = ev.observation?.signal?.strength ?? 0;
      const c = ev.belief?.confidence ?? 0;
      const h = ev.belief?.entropy_bits ?? 0;
      expect(s).toBeGreaterThanOrEqual(prevSignal - 0.05);
      expect(c).toBeGreaterThanOrEqual(prevConf - 0.01);
      expect(h).toBeLessThanOrEqual(prevEntropy + 0.05);
      prevSignal = s;
      prevConf = c;
      prevEntropy = h;
    }
  });

  it("starts at [1,1], moves north-east, and ends adjacent to [17,12]", () => {
    const parsed = parseTrace(sampleText);
    if (!parsed.ok) throw new Error("parse failed");
    const positions = parsed.events.map((ev) => ev.observation?.position);
    expect(positions[0]).toEqual([1, 1]);
    const finalPos = positions[positions.length - 1];
    expect(finalPos).toEqual([17, 11]);
    expect(finalPos?.[0]).toBeGreaterThan(1);
    expect(finalPos?.[1]).toBeGreaterThan(1);
    const last = parsed.events.at(-1);
    if (!last) throw new Error("unexpected empty trace");
    expect(last.kind).toBe("terminal");
    expect(last.action?.type).toBe("declare_found");
    expect(last.action?.position).toEqual([17, 12]);
    expect(last.terminal).toBe("EVE_LOCATED_SUCCESS");
  });

  it("contains exactly one clue_discovered event and it is at tick 18", () => {
    const parsed = parseTrace(sampleText);
    if (!parsed.ok) throw new Error("parse failed");
    const clueEvents = parsed.events.flatMap((ev) => ev.observation?.events ?? []);
    expect(clueEvents.length).toBe(1);
    expect(clueEvents[0]?.tick).toBe(18);
    expect(clueEvents[0]?.kind).toBe("footprint");
  });

  it("has a decision with reason text on every non-initial event", () => {
    const parsed = parseTrace(sampleText);
    if (!parsed.ok) throw new Error("parse failed");
    for (const ev of parsed.events) {
      expect(ev.decision).not.toBeNull();
      expect((ev.decision?.reason ?? "").length).toBeGreaterThan(20);
    }
  });

  it("keeps battery consistent with action costs (move 1, scan 2, declare 1)", () => {
    const parsed = parseTrace(sampleText);
    if (!parsed.ok) throw new Error("parse failed");
    const costs: Record<string, number> = { move: 1, scan: 2, recharge: 0, return_home: 0, declare_found: 1 };
    for (let i = 1; i < parsed.events.length; i += 1) {
      const prev = parsed.events[i - 1];
      const ev = parsed.events[i];
      const actionType = ev?.action?.type ?? "";
      const expected = (prev?.battery ?? 0) - (costs[actionType] ?? 0);
      expect(ev?.battery).toBe(expected);
    }
  });

  it("exposes scan and declare strategy-change markers", () => {
    const parsed = parseTrace(sampleText);
    if (!parsed.ok) throw new Error("parse failed");
    const markers = strategyMarkers(parsed.events);
    const labels = markers.map((m) => m.label);
    expect(labels).toContain("scan");
    expect(labels).toContain("declare");
    expect(labels).not.toContain("recharge");
  });
});

describe("sample_report.json", () => {
  it("validates against missionReportSchema", () => {
    expect(missionReportSchema.safeParse(reportJson).success).toBe(true);
  });
});

describe("sample_config.json", () => {
  it("validates against missionConfigSchema", () => {
    expect(missionConfigSchema.safeParse(configJson).success).toBe(true);
  });
});

describe("parseTrace error handling", () => {
  it("rejects invalid JSON with a line number", () => {
    const result = parseTrace("\nnot json");
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.line).toBe(2);
    expect(result.error).toContain("invalid JSON");
  });

  it("rejects schema violations with a line number", () => {
    const result = parseTrace('{"schema_version":"1.0","seq":-1}');
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.line).toBe(1);
  });

  it("rejects an empty trace", () => {
    const result = parseTrace("\n\n");
    expect(result.ok).toBe(false);
  });
});
