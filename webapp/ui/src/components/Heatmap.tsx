import { useMemo } from "react";
import type { EChartsOption } from "echarts";
import { Chart } from "./Chart";
import { cssVar, chartTextStyle } from "../utils/chartTheme";

interface HeatmapProps {
  matrix: number[][];
  rowsLabel?: string | null;
  colsLabel?: string | null;
  height?: number;
  valueLabel?: string;
}

/** Renders a 2D matrix (rows x cols) as an ECharts heatmap series. */
export function Heatmap({
  matrix,
  rowsLabel,
  colsLabel,
  height = 260,
  valueLabel = "value",
}: HeatmapProps) {
  const option = useMemo<EChartsOption>(() => {
    const nRows = matrix.length;
    const nCols = nRows > 0 ? matrix[0]?.length ?? 0 : 0;

    const data: [number, number, number][] = [];
    let max = -Infinity;
    let min = Infinity;
    for (let r = 0; r < nRows; r++) {
      for (let c = 0; c < nCols; c++) {
        const v = matrix[r]?.[c] ?? 0;
        data.push([c, r, v]);
        if (v > max) max = v;
        if (v < min) min = v;
      }
    }
    if (!Number.isFinite(max)) max = 1;
    if (!Number.isFinite(min)) min = 0;

    return {
      textStyle: chartTextStyle(),
      tooltip: {
        position: "top",
        formatter: (p) => {
          const arr = (p as { data: number[] }).data;
          return `${colsLabel ?? "col"} ${arr[0]}, ${rowsLabel ?? "row"} ${arr[1]}<br/>${valueLabel}: ${arr[2].toFixed(3)}`;
        },
      },
      grid: { left: 56, right: 16, top: 12, bottom: 36, containLabel: true },
      xAxis: {
        type: "category",
        data: Array.from({ length: nCols }, (_, i) => String(i)),
        name: colsLabel ?? undefined,
        nameLocation: "middle",
        nameGap: 28,
        splitArea: { show: true },
        axisLabel: { color: cssVar("--text-muted"), fontSize: 10, interval: Math.ceil(nCols / 16) },
        axisLine: { lineStyle: { color: cssVar("--border-strong") } },
      },
      yAxis: {
        type: "category",
        data: Array.from({ length: nRows }, (_, i) => String(i)),
        name: rowsLabel ?? undefined,
        nameLocation: "middle",
        nameGap: 36,
        splitArea: { show: true },
        axisLabel: { color: cssVar("--text-muted"), fontSize: 10, interval: Math.ceil(nRows / 12) },
        axisLine: { lineStyle: { color: cssVar("--border-strong") } },
      },
      visualMap: {
        min,
        max,
        calculable: false,
        show: false,
        inRange: {
          color: [cssVar("--bg-sunken"), cssVar("--accent")],
        },
      },
      series: [
        {
          type: "heatmap",
          data,
          progressive: 2000,
          itemStyle: { borderColor: cssVar("--bg-raised"), borderWidth: 1 },
        },
      ],
    };
  }, [matrix, rowsLabel, colsLabel, valueLabel]);

  if (!matrix.length || !matrix[0]?.length) {
    return (
      <div className="empty-state" style={{ padding: "24px 0" }}>
        <span className="text-sm">No matrix data available.</span>
      </div>
    );
  }

  return <Chart option={option} height={height} ariaLabel={valueLabel} />;
}
