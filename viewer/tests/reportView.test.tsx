import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { missionConfigSchema, missionReportSchema, type MissionConfig, type MissionReport } from "@walle/contracts";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ReportView } from "../src/components/ReportView";
import { ConfigView } from "../src/components/ConfigView";

const report: MissionReport = missionReportSchema.parse(
  JSON.parse(readFileSync(resolve(process.cwd(), "public/samples/sample_report.json"), "utf8")) as unknown,
);
const config: MissionConfig = missionConfigSchema.parse(
  JSON.parse(readFileSync(resolve(process.cwd(), "public/samples/sample_config.json"), "utf8")) as unknown,
);

describe("ReportView", () => {
  it("renders the outcome terminal state", () => {
    render(<ReportView report={report} />);
    expect(screen.getByText("EVE_LOCATED_SUCCESS")).toBeTruthy();
    expect(screen.getByText("156")).toBeTruthy();
  });

  it("renders every evidence entry with description and position", () => {
    render(<ReportView report={report} />);
    const list = screen.getByTestId("evidence-list");
    expect(list.textContent).toContain("Tread pattern matches EVE's hull unit.");
    expect(list.textContent).toContain("Fresh scorch mark on the hull plating");
    expect(list.textContent).toContain("[3, 8]");
    expect(screen.getByText("scorch mark")).toBeTruthy();
  });

  it("renders the inference with a confidence meter", () => {
    render(<ReportView report={report} />);
    const meter = screen.getByTestId("confidence-meter");
    expect(meter.textContent).toContain("87%");
    expect(screen.getByText(/return to home with evidence log/)).toBeTruthy();
  });

  it("renders strategy phases and labels evaluation as evaluator-only", () => {
    render(<ReportView report={report} />);
    expect(screen.getByText("signal acquisition")).toBeTruthy();
    expect(screen.getByText("gradient follow")).toBeTruthy();
    expect(screen.getByText("evaluator-only")).toBeTruthy();
    expect(screen.getAllByText("[17, 12]").length).toBeGreaterThanOrEqual(2);
  });
});

describe("ConfigView", () => {
  it("renders map, budget, and sensor readouts from the config", () => {
    render(<ConfigView config={config} />);
    expect(screen.getByText("20 × 15 cells")).toBeTruthy();
    expect(screen.getByText("220")).toBeTruthy();
    expect(screen.getByText("log_normal_path_loss")).toBeTruthy();
    expect(screen.getAllByText("[1, 1]").length).toBeGreaterThan(0);
  });
});
