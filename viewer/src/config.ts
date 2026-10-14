/** Deployment configuration — never hardcode URLs in components. */

export const apiHttpUrl: string = import.meta.env.VITE_API_HTTP_URL ?? "http://localhost:8000";
export const apiWsUrl: string = import.meta.env.VITE_API_WS_URL ?? "ws://localhost:8000";

/** Relative paths to offline sample assets served from public/. */
export const SAMPLE_PATHS = {
  trace: "/samples/sample_basic.jsonl",
  report: "/samples/sample_report.json",
  config: "/samples/sample_config.json",
} as const;
