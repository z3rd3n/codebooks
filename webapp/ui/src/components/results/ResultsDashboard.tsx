import { useState } from "react";
import type { RunResult } from "../../api/types";
import { MetricCards } from "./MetricCards";
import { SeVsSnrChart } from "./SeVsSnrChart";
import { OverheadBar } from "./OverheadBar";
import { Heatmap } from "../Heatmap";
import { BeamGridPlot } from "./BeamGridPlot";
import { PmiTable } from "./PmiTable";
import { CodeSnippet } from "./CodeSnippet";
import { Panel } from "./Panel";
import { fmtSeconds } from "../../utils/format";
import { Icon } from "../Icon";

interface ResultsDashboardProps {
  result: RunResult;
  fieldDescriptions?: Record<string, string>;
}

export function ResultsDashboard({ result, fieldDescriptions }: ResultsDashboardProps) {
  const [layerIdx, setLayerIdx] = useState(0);
  const metrics = result.metrics;
  const viz = result.viz;
  const layers = viz?.precoder ?? [];
  const activeLayer = layers[layerIdx] ?? layers[0] ?? null;

  return (
    <div className="col col-gap-4">
      <div className="row row-gap-3" style={{ justifyContent: "space-between" }}>
        <div className="row row-gap-2 text-sm text-muted">
          <Icon name="check" size={14} />
          {result.scheme_name ?? "Run complete"} &middot; {fmtSeconds(result.seconds)}
        </div>
      </div>

      {metrics && <MetricCards metrics={metrics} />}

      {metrics && metrics.snr_db?.length > 0 && (
        <Panel
          title="Spectral efficiency vs SNR"
          subtitle="Achieved SE from the reported PMI, compared to the ideal-eigenvector upper bound. Shaded area is the gap you pay for quantized feedback."
          about="The eigen bound assumes the gNB knows the exact channel eigenvectors; the achieved curve uses the quantized PMI this codebook actually reports. A smaller shaded gap means less capacity lost to feedback compression."
        >
          <SeVsSnrChart snrDb={metrics.snr_db} se={metrics.se} seUpperBound={metrics.se_upper_bound} />
        </Panel>
      )}

      {metrics?.overhead_bits && Object.keys(metrics.overhead_bits).length > 0 && (
        <Panel
          title="Overhead breakdown"
          subtitle="How the total feedback report is spent across PMI fields, from one representative drop."
          about="Each segment is one part of the PMI report (e.g. beam selection, amplitude, phase). Hover a segment for its bit count and a plain-language description. Wider segments cost more feedback bits."
        >
          <OverheadBar overheadBits={metrics.overhead_bits} fieldDescriptions={fieldDescriptions} />
        </Panel>
      )}

      {(viz?.channel || layers.length > 0) && (
        <Panel
          title="Channel and precoder heatmaps"
          subtitle="Magnitude of the channel matrix |H| and the reconstructed precoder |W| for the selected layer, both across frequency (N3) and antenna ports."
          about="|H| shows how the channel's strength varies by frequency unit and port before any compression. |W| shows what the codebook actually reconstructs from the quantized PMI for one transmit layer — compare the two to see how faithfully the codebook captures the channel's structure."
          right={
            layers.length > 1 ? (
              <div className="tabs" style={{ borderBottom: "none", marginBottom: 0 }}>
                {layers.map((l, i) => (
                  <button
                    key={l.layer}
                    className={`tab${i === layerIdx ? " active" : ""}`}
                    onClick={() => setLayerIdx(i)}
                    style={{ padding: "6px 10px" }}
                  >
                    Layer {l.layer}
                  </button>
                ))}
              </div>
            ) : undefined
          }
        >
          <div className="grid-2">
            {viz?.channel && (
              <div>
                <div className="text-sm text-muted mt-2" style={{ marginBottom: 8 }}>
                  |H| — channel magnitude
                </div>
                <Heatmap
                  matrix={viz.channel.abs}
                  rowsLabel={viz.channel.rows}
                  colsLabel={viz.channel.cols}
                  valueLabel="|H|"
                />
              </div>
            )}
            {activeLayer && (
              <div>
                <div className="text-sm text-muted mt-2" style={{ marginBottom: 8 }}>
                  |W| — precoder magnitude (layer {activeLayer.layer})
                </div>
                <Heatmap matrix={activeLayer.abs} rowsLabel="N3 frequency units" colsLabel="P ports" valueLabel="|W|" />
              </div>
            )}
          </div>
        </Panel>
      )}

      {viz?.beam_grid && (
        <Panel
          title="Selected DFT beams"
          subtitle="Which oversampled DFT beam indices this drop's PMI selected, out of the full beam grid."
          about="Every dot is one candidate beam in the oversampled DFT codebook grid; highlighted dots are the beams actually selected and reported in the PMI for this drop."
        >
          <BeamGridPlot beamGrid={viz.beam_grid} />
        </Panel>
      )}

      {result.pmi?.fields && result.pmi.fields.length > 0 && (
        <Panel
          title="PMI report"
          subtitle="The individual fields that make up the fed-back precoder matrix indicator for this drop."
        >
          <PmiTable fields={result.pmi.fields} />
        </Panel>
      )}

      {result.python_snippet && (
        <Panel title="Python equivalent" subtitle="Copy-pasteable code that reproduces this run outside the UI.">
          <CodeSnippet code={result.python_snippet} />
        </Panel>
      )}
    </div>
  );
}
