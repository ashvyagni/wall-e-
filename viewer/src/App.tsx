import { SCHEMA_VERSION } from "@walle/contracts";

export function App() {
  return (
    <main className="shell">
      <header className="shell-header">
        <h1>WALL-E Mission Control</h1>
        <span className="badge">contract v{SCHEMA_VERSION}</span>
      </header>
      <p className="placeholder">Replay viewer and live lab land here (UI track, H6+).</p>
    </main>
  );
}
