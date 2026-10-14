import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { missionConfigSchema, type MissionConfig } from "@walle/contracts";
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ReplayPlayer } from "../src/components/ReplayPlayer";

const sampleText = readFileSync(resolve(process.cwd(), "public/samples/sample_basic.jsonl"), "utf8");
const configJson = missionConfigSchema.parse(
  JSON.parse(readFileSync(resolve(process.cwd(), "public/samples/sample_config.json"), "utf8")) as unknown,
);
const config: MissionConfig = configJson;

function scrubTo(index: number) {
  const scrubber = screen.getByLabelText("Mission timeline scrubber");
  fireEvent.change(scrubber, { target: { value: String(index) } });
}

describe("ReplayPlayer", () => {
  it("renders the agent position at tick 0", () => {
    render(<ReplayPlayer text={sampleText} config={config} />);
    const agent = screen.getByTestId("agent-marker");
    expect(agent.getAttribute("data-cell")).toBe("1,1");
  });

  it("renders the agent position after scrubbing to a later tick", () => {
    render(<ReplayPlayer text={sampleText} config={config} />);
    scrubTo(20);
    const agent = screen.getByTestId("agent-marker");
    expect(agent.getAttribute("data-cell")).toBe("3,8");
  });

  it("updates the battery bar as the trace advances", () => {
    render(<ReplayPlayer text={sampleText} config={config} />);
    const meter = screen.getByRole("meter", { name: "Battery" });
    expect(meter.getAttribute("aria-valuenow")).toBe("220");
    expect(screen.getByTestId("battery-value").textContent).toContain("220 / 220");

    scrubTo(40);
    expect(meter.getAttribute("aria-valuenow")).toBe("177");
    expect(screen.getByTestId("battery-value").textContent).toContain("177 / 220");
  });

  it("steps forward one event with the step button", () => {
    render(<ReplayPlayer text={sampleText} config={config} />);
    fireEvent.click(screen.getByLabelText("Step forward one event"));
    // tick 1 is a scan — agent holds position [1,1]
    expect(screen.getByTestId("agent-marker").getAttribute("data-cell")).toBe("1,1");
    expect(screen.getByTestId("tick-value").textContent).toBe("1");
    fireEvent.click(screen.getByLabelText("Step forward one event"));
    expect(screen.getByTestId("agent-marker").getAttribute("data-cell")).toBe("1,2");
    expect(screen.getByTestId("tick-value").textContent).toBe("2");
  });

  it("shows the terminal banner on the last tick", () => {
    render(<ReplayPlayer text={sampleText} config={config} />);
    expect(screen.queryByText("EVE_LOCATED_SUCCESS")).toBeNull();
    scrubTo(40);
    const banner = screen.getByText("EVE_LOCATED_SUCCESS");
    expect(banner).toBeTruthy();
    expect(banner.closest(".terminal-banner")).toBeTruthy();
  });

  it("shows the decision reason verbatim from the trace", () => {
    render(<ReplayPlayer text={sampleText} config={config} />);
    scrubTo(24);
    const reason = screen.getByTestId("decision-reason").textContent ?? "";
    expect(reason).toContain("Signal strength climbing");
    expect(reason).toContain("east");
  });

  it("shows an error state for an invalid trace", () => {
    render(<ReplayPlayer text={'{"schema_version":"1.0"}'} />);
    const alert = screen.getByRole("alert");
    expect(alert.getAttribute("data-testid")).toBe("trace-error");
    expect(alert.textContent).toContain("Trace failed validation");
  });

  it("shows an error state for structurally broken JSON", () => {
    render(<ReplayPlayer text={"this is not jsonl"} />);
    expect(screen.getByTestId("trace-error").textContent).toContain("invalid JSON");
  });
});
