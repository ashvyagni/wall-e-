/** Deployment configuration — never hardcode URLs in components. */

export const apiHttpUrl: string = import.meta.env.VITE_API_HTTP_URL ?? "http://localhost:8000";
export const apiWsUrl: string = import.meta.env.VITE_API_WS_URL ?? "ws://localhost:8000";
