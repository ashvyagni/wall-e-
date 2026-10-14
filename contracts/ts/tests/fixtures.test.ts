import { readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { fixtureSchemas, terminalStateSchema } from "../src/contracts";

const here = dirname(fileURLToPath(import.meta.url));
const fixturesDir = join(here, "..", "..", "..", "contracts", "fixtures");

function loadFixture(name: string): unknown {
  return JSON.parse(readFileSync(join(fixturesDir, name), "utf-8"));
}

describe("cross-language contract compatibility", () => {
  const onDisk = readdirSync(fixturesDir)
    .filter((f) => f.endsWith(".json"))
    .sort();
  const registered = Object.keys(fixtureSchemas).sort();

  it("fixture registry matches Python FIXTURE_MAP", () => {
    expect(onDisk).toEqual(registered);
  });

  for (const [name, schema] of Object.entries(fixtureSchemas)) {
    it(`${name} validates against zod mirror`, () => {
      const result = schema.safeParse(loadFixture(name));
      if (!result.success) {
        throw new Error(JSON.stringify(result.error.issues, null, 2));
      }
      expect(result.success).toBe(true);
    });

    it(`${name} rejects unknown fields (parity with pydantic extra=forbid)`, () => {
      const data = loadFixture(name) as Record<string, unknown>;
      const result = schema.safeParse({ ...data, __smuggled_field__: 1 });
      expect(result.success).toBe(false);
    });
  }

  it("schema_version is 1.0 on every versioned document", () => {
    for (const name of registered) {
      const data = loadFixture(name) as Record<string, unknown>;
      expect(data.schema_version, name).toBe("1.0");
    }
  });

  it("terminal state enum matches Python TerminalState exactly", () => {
    expect(terminalStateSchema.options).toEqual([
      "EVE_LOCATED_SUCCESS",
      "EVE_LOCATE_FAILED",
      "RETURNED_NO_EVIDENCE",
      "BATTERY_EXHAUSTED_ABORTED",
      "TIMEOUT_ABORTED",
      "INVALID_ACTION_ABORTED",
    ]);
  });
});
