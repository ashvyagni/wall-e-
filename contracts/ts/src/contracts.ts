/**
 * Zod mirror of the frozen Python contract schemas (schema_version 1.0).
 *
 * Source of truth: src/walle/contracts.py + contracts/SEMANTICS.md.
 * Every fixture in contracts/fixtures/*.json must validate against BOTH the
 * pydantic models (Python tests) and these zod schemas (vitest) — a divergence
 * fails CI in either language.
 */

import { z } from "zod";

export const SCHEMA_VERSION = "1.0" as const;

const pos = z.tuple([z.number().int(), z.number().int()]);

export const terminalStateSchema = z.enum([
  "EVE_LOCATED_SUCCESS",
  "EVE_LOCATE_FAILED",
  "RETURNED_NO_EVIDENCE",
  "BATTERY_EXHAUSTED_ABORTED",
  "TIMEOUT_ABORTED",
  "INVALID_ACTION_ABORTED",
]);

export const actionTypeSchema = z.enum([
  "move",
  "scan",
  "recharge",
  "return_home",
  "declare_found",
]);

export const moveDirectionSchema = z.enum(["north", "south", "east", "west"]);
export const signalStatusSchema = z.enum(["ok", "degraded", "failed"]);
export const clueKindSchema = z.enum([
  "footprint",
  "scorch_mark",
  "plant_sample",
  "waste_scrap",
  "signal_mirror",
]);

// --- MissionConfig ---------------------------------------------------------

export const mapConfigSchema = z
  .object({
    width: z.number().int().min(2).max(512),
    height: z.number().int().min(2).max(512),
    cell_size: z.number().positive(),
    obstacles: z.array(pos),
    home: pos,
    recharge_stations: z.array(pos),
  })
  .strict();

export const budgetConfigSchema = z
  .object({
    battery: z.number().int().positive(),
    max_ticks: z.number().int().positive(),
    action_costs: z.record(z.string(), z.number().int().min(0)),
    recharge_ticks: z.number().int().min(1),
    reserve_margin: z.number().int().min(0),
  })
  .strict()
  .superRefine((val, ctx) => {
    const known = ["move", "scan", "recharge", "return_home", "declare_found"];
    for (const k of Object.keys(val.action_costs)) {
      if (!known.includes(k)) {
        ctx.addIssue({ code: z.ZodIssueCode.custom, message: `unknown action cost: ${k}` });
      }
    }
    for (const k of known) {
      if (!(k in val.action_costs)) {
        ctx.addIssue({ code: z.ZodIssueCode.custom, message: `missing action cost: ${k}` });
      }
    }
  });

export const signalSensorConfigSchema = z
  .object({
    model: z.literal("log_normal_path_loss"),
    reference_distance: z.number().positive(),
    reference_power: z.number().positive(),
    path_loss_exponent: z.number().positive(),
    noise_std_db: z.number().min(0),
    bearing_noise_std_deg: z.number().min(0),
    max_range: z.number().int().positive(),
    scan_noise_scale: z.number().positive(),
  })
  .strict();

export const sensorsConfigSchema = z
  .object({
    signal: signalSensorConfigSchema,
    local_view_radius: z.number().int().min(0).max(16),
    scan_radius: z.number().int().min(1).max(32),
    failure_rate: z.number().min(0).max(1),
  })
  .strict();

export const missionConfigSchema = z
  .object({
    schema_version: z.literal("1.0"),
    mission_id: z.string().min(1),
    seed: z.number().int(),
    map: mapConfigSchema,
    budget: budgetConfigSchema,
    sensors: sensorsConfigSchema,
    target: z
      .object({
        placement: pos.nullable(),
        contact_radius: z.number().int().min(0).max(8),
        status: z.enum(["dormant", "active"]),
      })
      .strict(),
    clues: z
      .object({
        placements: z.array(z.tuple([pos, clueKindSchema])),
        seed_count: z.number().int().min(0).max(64),
        detection_radius: z.number().int().min(1).max(16),
        spawn_radius: z.number().int().min(0).max(64).default(3),
      })
      .strict(),
    decoys: z
      .object({
        count: z.number().int().min(0).max(16),
        seed: z.number().int(),
        placements: z.array(pos),
        strength_floor: z.number().min(0).max(1),
        min_clue_distance: z.number().int().min(0).max(64).default(4),
      })
      .strict(),
    agents: z
      .array(
        z
          .object({
            id: z.string().min(1),
            start: pos.nullable(),
          })
          .strict(),
      )
      .min(1),
    planner: z.enum(["classical", "llm"]),
  })
  .strict();

// --- Action ----------------------------------------------------------------

export const actionSchema = z
  .object({
    schema_version: z.literal("1.0"),
    type: actionTypeSchema,
    direction: moveDirectionSchema.nullable(),
    position: pos.nullable(),
  })
  .strict()
  .superRefine((val, ctx) => {
    if (val.type === "move" && val.direction === null) {
      ctx.addIssue({ code: z.ZodIssueCode.custom, message: "move requires direction" });
    }
    if (val.type !== "move" && val.direction !== null) {
      ctx.addIssue({ code: z.ZodIssueCode.custom, message: `${val.type} must not carry direction` });
    }
    if (val.type === "declare_found" && val.position === null) {
      ctx.addIssue({ code: z.ZodIssueCode.custom, message: "declare_found requires position" });
    }
    if (val.type !== "declare_found" && val.position !== null) {
      ctx.addIssue({ code: z.ZodIssueCode.custom, message: `${val.type} must not carry position` });
    }
  });

// --- Observation ------------------------------------------------------------

export const signalReadingSchema = z
  .object({
    bearing_deg: z.number(),
    strength: z.number().min(0).max(1),
    measured_from: pos,
    timestamp_tick: z.number().int(),
  })
  .strict();

export const visibleCellSchema = z
  .object({
    pos: pos,
    kind: z.enum(["free", "obstacle", "recharge", "home"]),
  })
  .strict();

export const clueEventSchema = z
  .object({
    type: z.literal("clue_discovered"),
    clue_id: z.string().min(1),
    kind: clueKindSchema,
    pos: pos,
    description: z.string(),
    tick: z.number().int(),
  })
  .strict();

export const observationSchema = z
  .object({
    schema_version: z.literal("1.0"),
    tick: z.number().int().min(0),
    agent_id: z.string().min(1),
    position: pos,
    signal: signalReadingSchema.nullable(),
    visible_cells: z.array(visibleCellSchema),
    events: z.array(clueEventSchema),
    battery: z.number().int().min(0),
    max_battery: z.number().int().positive(),
    max_ticks: z.number().int().positive(),
    sensor_status: z.record(z.string(), signalStatusSchema),
    last_action_valid: z.boolean(),
    last_action_error: z.string().nullable(),
  })
  .strict();

// --- TraceEvent -------------------------------------------------------------

export const beliefSummarySchema = z
  .object({
    map_pos: pos,
    confidence: z.number().min(0).max(1),
    entropy_bits: z.number().min(0),
    top_k: z.array(z.tuple([pos, z.number()])),
  })
  .strict();

export const decisionRecordSchema = z
  .object({
    planner: z.string(),
    action_type: z.string(),
    reason: z.string(),
    options_considered: z.array(z.string()),
    extras: z.record(z.string(), z.unknown()),
  })
  .strict();

export const traceEventSchema = z
  .object({
    schema_version: z.literal("1.0"),
    seq: z.number().int().min(0),
    tick: z.number().int().min(0),
    agent_id: z.string().min(1),
    kind: z.enum(["initial", "step", "terminal"]),
    action: actionSchema.nullable(),
    observation: observationSchema.nullable(),
    belief: beliefSummarySchema.nullable(),
    decision: decisionRecordSchema.nullable(),
    battery: z.number().int().min(0),
    terminal: terminalStateSchema.nullable(),
  })
  .strict();

// --- MissionReport ----------------------------------------------------------

export const missionReportSchema = z
  .object({
    schema_version: z.literal("1.0"),
    mission_id: z.string().min(1),
    seed: z.number().int(),
    planner: z.string(),
    generated_by: z.string(),
    outcome: z
      .object({
        terminal: terminalStateSchema,
        successful: z.boolean(),
        steps: z.number().int().min(0),
        ticks_used: z.number().int().min(0),
        battery_used: z.number().int().min(0),
        battery_remaining: z.number().int().min(0),
      })
      .strict(),
    evidence: z.array(
      z
        .object({
          clue_id: z.string(),
          kind: clueKindSchema,
          pos: pos,
          description: z.string(),
          tick: z.number().int().min(0),
        })
        .strict(),
    ),
    inference: z
      .object({
        eve_location: pos,
        status: z.enum(["located", "estimated", "unknown"]),
        confidence: z.number().min(0).max(1),
        recommended_next_step: z.string(),
      })
      .strict(),
    strategy: z
      .object({
        phases: z.array(
          z
            .object({
              name: z.string(),
              start_tick: z.number().int().min(0),
              end_tick: z.number().int().min(0),
              note: z.string(),
            })
            .strict(),
        ),
        recharges: z.number().int().min(0),
        scans: z.number().int().min(0),
        turn_back_tick: z.number().int().nullable(),
      })
      .strict(),
    evaluation: z
      .object({
        true_eve_pos: pos,
        localization_error: z.number().min(0),
        clues_total: z.number().int().min(0),
        clues_found: z.number().int().min(0),
        invalid_action_count: z.number().int().min(0),
        failed_claims: z.number().int().min(0),
      })
      .strict(),
    trace_path: z.string().min(1),
    notes: z.array(z.string()),
  })
  .strict();

// --- Fixture registry (mirrors FIXTURE_MAP in Python) -----------------------

export const fixtureSchemas = {
  "mission_config.json": missionConfigSchema,
  "observation.json": observationSchema,
  "action_move.json": actionSchema,
  "action_declare.json": actionSchema,
  "trace_event.json": traceEventSchema,
  "mission_report.json": missionReportSchema,
} as const;

export type MissionConfig = z.infer<typeof missionConfigSchema>;
export type Observation = z.infer<typeof observationSchema>;
export type Action = z.infer<typeof actionSchema>;
export type TraceEvent = z.infer<typeof traceEventSchema>;
export type MissionReport = z.infer<typeof missionReportSchema>;
export type TerminalState = z.infer<typeof terminalStateSchema>;
