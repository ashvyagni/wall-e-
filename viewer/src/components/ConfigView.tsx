import type { MissionConfig } from "@walle/contracts";

interface ConfigViewProps {
  config: MissionConfig;
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <tr>
      <th scope="row">{k}</th>
      <td>{v}</td>
    </tr>
  );
}

export function ConfigView({ config }: ConfigViewProps) {
  const { map, budget, sensors } = config;

  return (
    <div className="view-stack">
      <section className="card">
        <h2 className="card-title">Mission</h2>
        <table className="readout-table">
          <tbody>
            <Row k="Mission ID" v={config.mission_id} />
            <Row k="Schema version" v={config.schema_version} />
            <Row k="Seed" v={String(config.seed)} />
            <Row k="Planner" v={config.planner} />
            <Row
              k="Agents"
              v={config.agents.map((a) => `${a.id}${a.start ? ` @ [${a.start[0]}, ${a.start[1]}]` : ""}`).join(", ")}
            />
          </tbody>
        </table>
      </section>

      <div className="readout-cols">
        <section className="card">
          <h2 className="card-title">Map</h2>
          <table className="readout-table">
            <tbody>
              <Row k="Dimensions" v={`${map.width} × ${map.height} cells`} />
              <Row k="Cell size" v={`${map.cell_size} m`} />
              <Row k="Home" v={`[${map.home[0]}, ${map.home[1]}]`} />
              <Row
                k="Recharge stations"
                v={map.recharge_stations.map((s) => `[${s[0]}, ${s[1]}]`).join(", ") || "none"}
              />
              <Row k="Obstacles" v={`${map.obstacles.length} cells`} />
            </tbody>
          </table>
        </section>

        <section className="card">
          <h2 className="card-title">Budget</h2>
          <table className="readout-table">
            <tbody>
              <Row k="Battery" v={String(budget.battery)} />
              <Row k="Max ticks" v={String(budget.max_ticks)} />
              <Row
                k="Action costs"
                v={Object.entries(budget.action_costs)
                  .map(([k, v]) => `${k}: ${v}`)
                  .join(" · ")}
              />
              <Row k="Recharge ticks" v={String(budget.recharge_ticks)} />
              <Row k="Reserve margin" v={String(budget.reserve_margin)} />
            </tbody>
          </table>
        </section>

        <section className="card">
          <h2 className="card-title">Sensors</h2>
          <table className="readout-table">
            <tbody>
              <Row k="Signal model" v={sensors.signal.model} />
              <Row k="Reference distance" v={`${sensors.signal.reference_distance} m`} />
              <Row k="Reference power" v={String(sensors.signal.reference_power)} />
              <Row k="Path loss exponent" v={String(sensors.signal.path_loss_exponent)} />
              <Row k="Noise σ" v={`${sensors.signal.noise_std_db} dB`} />
              <Row k="Bearing noise σ" v={`${sensors.signal.bearing_noise_std_deg}°`} />
              <Row k="Max range" v={`${sensors.signal.max_range} cells`} />
              <Row k="Scan noise scale" v={String(sensors.signal.scan_noise_scale)} />
              <Row k="Local view radius" v={String(sensors.local_view_radius)} />
              <Row k="Scan radius" v={String(sensors.scan_radius)} />
              <Row k="Failure rate" v={String(sensors.failure_rate)} />
            </tbody>
          </table>
        </section>
      </div>
    </div>
  );
}
