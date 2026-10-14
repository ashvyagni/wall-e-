import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { App } from "../src/App";

describe("App shell", () => {
  it("renders the mission control header with contract version", () => {
    render(<App />);
    expect(screen.getByText("WALL-E Mission Control")).toBeTruthy();
    expect(screen.getByText("contract v1.0")).toBeTruthy();
  });
});
