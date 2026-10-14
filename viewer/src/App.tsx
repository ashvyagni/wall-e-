import { useCallback, useEffect, useMemo, useRef, useState, type DragEvent, type KeyboardEvent } from "react";
import { SCHEMA_VERSION, missionConfigSchema, missionReportSchema, type MissionConfig, type MissionReport } from "@walle/contracts";
import { SAMPLE_PATHS } from "./config";
import { parseTrace } from "./lib/trace";
import { lastTerminal, SUCCESS_TERMINALS } from "./lib/derive";
import { ReplayPlayer } from "./components/ReplayPlayer";
import { ReportView } from "./components/ReportView";
import { ConfigView } from "./components/ConfigView";

type Tab = "replay" | "report" | "config";

const TABS: ReadonlyArray<{ id: Tab; label: string }> = [
  { id: "replay", label: "Replay" },
  { id: "report", label: "Report" },
  { id: "config", label: "Config" },
];

interface LoadedSamples {
  traceText: string;
  report: MissionReport | null;
  config: MissionConfig | null;
  error: string | null;
}

export function App() {
  const [tab, setTab] = useState<Tab>("replay");
  const [samples, setSamples] = useState<LoadedSamples | null>(null);
  const [customTrace, setCustomTrace] = useState<{ text: string; name: string } | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const [traceRes, reportRes, configRes] = await Promise.all([
          fetch(SAMPLE_PATHS.trace),
          fetch(SAMPLE_PATHS.report),
          fetch(SAMPLE_PATHS.config),
        ]);
        if (!traceRes.ok) throw new Error(`trace sample: HTTP ${traceRes.status}`);
        const traceText = await traceRes.text();

        let report: MissionReport | null = null;
        if (reportRes.ok) {
          const parsedReport = missionReportSchema.safeParse(await reportRes.json());
          report = parsedReport.success ? parsedReport.data : null;
        }

        let config: MissionConfig | null = null;
        if (configRes.ok) {
          const parsedConfig = missionConfigSchema.safeParse(await configRes.json());
          config = parsedConfig.success ? parsedConfig.data : null;
        }

        if (!cancelled) setSamples({ traceText, report, config, error: null });
      } catch (err) {
        if (!cancelled) {
          setSamples({
            traceText: "",
            report: null,
            config: null,
            error: err instanceof Error ? err.message : "failed to load samples",
          });
        }
      }
    }
    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  const activeTraceText = customTrace?.text ?? samples?.traceText ?? "";
  const parsedTrace = useMemo(() => parseTrace(activeTraceText), [activeTraceText]);
  const terminal = parsedTrace.ok ? lastTerminal(parsedTrace.events) : null;
  const missionId = samples?.config?.mission_id ?? samples?.report?.mission_id ?? "unknown";

  const handleFile = useCallback((file: File) => {
    file.text().then((text) => {
      setCustomTrace({ text, name: file.name });
      setTab("replay");
    });
  }, []);

  const onDrop = useCallback(
    (e: DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  const onTabKeyDown = useCallback((e: KeyboardEvent) => {
    const idx = TABS.findIndex((t) => t.id === tab);
    if (e.key === "ArrowRight") {
      const next = TABS[(idx + 1) % TABS.length];
      if (next) setTab(next.id);
    } else if (e.key === "ArrowLeft") {
      const prev = TABS[(idx - 1 + TABS.length) % TABS.length];
      if (prev) setTab(prev.id);
    }
  }, [tab]);

  return (
    <main className="shell">
      <header className="shell-header">
        <div className="brand">
          <h1>WALL-E Mission Control</h1>
          <span className="badge">contract v{SCHEMA_VERSION}</span>
        </div>
        <div className="header-meta">
          <span className="mission-id">mission: {missionId}</span>
          {terminal !== null && (
            <span className={`badge ${SUCCESS_TERMINALS.includes(terminal) ? "success" : "danger"}`}>
              {terminal}
            </span>
          )}
        </div>
      </header>

      <div className="tabs" role="tablist" aria-label="Mission views" onKeyDown={onTabKeyDown}>
        {TABS.map((t) => (
          <button
            key={t.id}
            type="button"
            role="tab"
            id={`tab-${t.id}`}
            className="tab"
            aria-selected={tab === t.id}
            aria-controls={`panel-${t.id}`}
            tabIndex={tab === t.id ? 0 : -1}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div className="tab-panels">
        {samples === null && (
          <p className="loading" role="status">
            Loading samples…
          </p>
        )}

        {samples?.error && (
          <div className="error-state" role="alert">
            <h2>Failed to load offline samples</h2>
            <p>{samples.error}</p>
          </div>
        )}

        {samples && !samples.error && (
          <>
            <div
              role="tabpanel"
              id="panel-replay"
              aria-labelledby="tab-replay"
              hidden={tab !== "replay"}
            >
              {tab === "replay" && (
                <div className="replay-main">
                  <div className="source-bar">
                    <span className="badge">
                      source: {customTrace ? customTrace.name : SAMPLE_PATHS.trace}
                    </span>
                    {customTrace && (
                      <button type="button" onClick={() => setCustomTrace(null)}>
                        Restore sample
                      </button>
                    )}
                    <label
                      className={`dropzone ${dragOver ? "dragover" : ""}`}
                      onDragOver={(e) => {
                        e.preventDefault();
                        setDragOver(true);
                      }}
                      onDragLeave={() => setDragOver(false)}
                      onDrop={onDrop}
                    >
                      Drop a .jsonl trace here or
                      <input
                        ref={fileInputRef}
                        className="file-input"
                        type="file"
                        accept=".jsonl,.json,application/json"
                        aria-label="Load trace file"
                        onChange={(e) => {
                          const file = e.target.files?.[0];
                          if (file) handleFile(file);
                        }}
                      />
                      <button type="button" onClick={() => fileInputRef.current?.click()} style={{ marginLeft: 6 }}>
                        browse…
                      </button>
                    </label>
                  </div>
                  <ReplayPlayer text={activeTraceText} config={samples.config} />
                </div>
              )}
            </div>

            <div
              role="tabpanel"
              id="panel-report"
              aria-labelledby="tab-report"
              hidden={tab !== "report"}
            >
              {tab === "report" &&
                (samples.report ? (
                  <ReportView report={samples.report} />
                ) : (
                  <div className="error-state" role="alert">
                    <h2>No mission report available</h2>
                    <p>The sample report failed validation or could not be loaded.</p>
                  </div>
                ))}
            </div>

            <div
              role="tabpanel"
              id="panel-config"
              aria-labelledby="tab-config"
              hidden={tab !== "config"}
            >
              {tab === "config" &&
                (samples.config ? (
                  <ConfigView config={samples.config} />
                ) : (
                  <div className="error-state" role="alert">
                    <h2>No mission config available</h2>
                    <p>The sample config failed validation or could not be loaded.</p>
                  </div>
                ))}
            </div>
          </>
        )}
      </div>
    </main>
  );
}
