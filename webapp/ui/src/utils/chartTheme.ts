/** Reads live CSS variable values so ECharts options match the active theme. */
export function cssVar(name: string): string {
  if (typeof window === "undefined") return "#000000";
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export const seriesColors = () => [
  cssVar("--series-1"),
  cssVar("--series-2"),
  cssVar("--series-3"),
  cssVar("--series-4"),
  cssVar("--series-5"),
  cssVar("--series-6"),
];

export function chartTextStyle() {
  return {
    color: cssVar("--text-secondary"),
    fontFamily: "var(--font-ui)",
    fontSize: 12,
  };
}

export function chartGrid(overrides: Record<string, unknown> = {}) {
  return {
    left: 48,
    right: 20,
    top: 28,
    bottom: 40,
    containLabel: true,
    ...overrides,
  };
}

export function axisLineStyle() {
  return { lineStyle: { color: cssVar("--border-strong") } };
}
