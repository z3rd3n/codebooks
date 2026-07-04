import type { AntennaPair, AntennaSpec } from "../../api/types";

interface AntennaPickerProps {
  spec: AntennaSpec | null | undefined;
  value: { n1: number; n2: number; ng?: number } | null;
  onChange: (pair: AntennaPair) => void;
}

function pairKey(p: { n1: number; n2: number; ng?: number | null }): string {
  return `${p.ng ?? 1}-${p.n1}-${p.n2}`;
}

/** Small SVG visualization of an N1 x N2 dual-polarized antenna grid. */
export function AntennaGridSvg({ n1, n2, ng = 1, size = 72 }: { n1: number; n2: number; ng?: number; size?: number }) {
  const cols = n1;
  const rows = n2 * ng;
  const maxDim = Math.max(cols, rows, 1);
  const cell = Math.min(14, Math.max(6, Math.floor((size - 8) / maxDim)));
  const w = cols * cell + 8;
  const h = rows * cell + 8;

  const dots: JSX.Element[] = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const cx = 4 + c * cell + cell / 2;
      const cy = 4 + r * cell + cell / 2;
      dots.push(
        <g key={`${r}-${c}`}>
          <circle cx={cx - cell * 0.14} cy={cy} r={Math.max(1.3, cell * 0.18)} fill="var(--accent)" />
          <circle cx={cx + cell * 0.14} cy={cy} r={Math.max(1.3, cell * 0.18)} fill="var(--series-2)" />
        </g>,
      );
    }
  }

  return (
    <svg
      width={w}
      height={h}
      viewBox={`0 0 ${w} ${h}`}
      role="img"
      aria-label={`${n1} by ${n2} dual-polarized antenna grid${ng > 1 ? `, ${ng} panels` : ""}`}
    >
      {dots}
    </svg>
  );
}

/** Card grid / single fixed display for choosing an antenna geometry. */
export function AntennaPicker({ spec, value, onChange }: AntennaPickerProps) {
  if (!spec || !spec.pairs?.length) {
    return <div className="text-sm text-muted">Antenna geometry unavailable.</div>;
  }

  if (spec.mode === "fixed" || spec.pairs.length === 1) {
    const p = spec.pairs[0];
    return (
      <div className="choice-card selected" style={{ cursor: "default" }}>
        <div className="row row-gap-3">
          <AntennaGridSvg n1={p.n1} n2={p.n2} ng={p.ng ?? 1} />
          <div>
            <div className="choice-card-label">
              N1={p.n1}, N2={p.n2}
              {p.ng ? `, Ng=${p.ng}` : ""}
            </div>
            <div className="choice-card-desc">{p.ports} ports</div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="choice-grid" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))" }}>
      {spec.pairs.map((p) => {
        const selected = value ? pairKey(p) === pairKey(value) : false;
        return (
          <button
            key={pairKey(p)}
            type="button"
            className={`choice-card${selected ? " selected" : ""}`}
            onClick={() => onChange(p)}
            aria-pressed={selected}
          >
            <div className="row row-gap-3">
              <AntennaGridSvg n1={p.n1} n2={p.n2} ng={p.ng ?? 1} size={56} />
              <div>
                <div className="choice-card-label">
                  {p.n1}&times;{p.n2}
                  {p.ng ? ` ×${p.ng} panels` : ""}
                </div>
                <div className="choice-card-desc">{p.ports} ports</div>
              </div>
            </div>
          </button>
        );
      })}
    </div>
  );
}
