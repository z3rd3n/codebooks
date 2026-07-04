/**
 * Number formatting: 3 significant digits, units always shown.
 */

export function fmtNum(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  if (value === 0) return "0";
  const abs = Math.abs(value);
  if (abs >= 1e6 || (abs < 1e-4 && abs > 0)) {
    return value.toExponential(digits - 1);
  }
  const magnitude = Math.floor(Math.log10(abs));
  const decimals = Math.max(0, digits - 1 - magnitude);
  return value.toFixed(Math.min(decimals, 6)).replace(/\.?0+$/, (m) =>
    m === "." ? "" : m.replace(/0+$/, "").replace(/\.$/, ""),
  );
}

export function fmtBits(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${fmtNum(value)} bit${Math.abs(value - 1) < 1e-9 ? "" : "s"}`;
}

export function fmtSE(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${fmtNum(value)} bit/s/Hz`;
}

export function fmtDb(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${fmtNum(value)} dB`;
}

export function fmtPct(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${fmtNum(value * 100)}%`;
}

export function fmtSeconds(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  if (value < 1) return `${Math.round(value * 1000)} ms`;
  return `${fmtNum(value)} s`;
}

/** Clamp + round-trip helper for controlled numeric inputs. */
export function clamp(value: number, min?: number | null, max?: number | null): number {
  let v = value;
  if (min !== null && min !== undefined) v = Math.max(min, v);
  if (max !== null && max !== undefined) v = Math.min(max, v);
  return v;
}
