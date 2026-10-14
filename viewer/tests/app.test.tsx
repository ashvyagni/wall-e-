import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { App } from "../src/App";

describe("App shell", () => {
  it("renders the mission control header with contract version", async () => {
    render(<App />);
    expect(screen.getByText("WALL-E Mission Control")).toBeTruthy();
    expect(screen.getByText("contract v1.0")).toBeTruthy();
    await waitFor(() => {
      expect(screen.queryByText("Loading samples…")).toBeNull();
    });
  });

  it("exposes the three mission view tabs", async () => {
    render(<App />);
    expect(screen.getByRole("tab", { name: "Replay" })).toBeTruthy();
    expect(screen.getByRole("tab", { name: "Report" })).toBeTruthy();
    expect(screen.getByRole("tab", { name: "Config" })).toBeTruthy();
    await waitFor(() => {
      expect(screen.queryByText("Loading samples…")).toBeNull();
    });
  });
});
