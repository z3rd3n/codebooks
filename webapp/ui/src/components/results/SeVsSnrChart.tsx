import { useMemo } from "react";
import type { EChartsOption } from "echarts";
import { Chart } from "../Chart";
import { cssVar, chartTextStyle, chartGrid } from "../../utils/chartTheme";

interface SeVsSnrChartProps {
  snrDb: number[];
  se: number[];
  seUpperBound: number[];
  height?: number;
}

export function SeVsSnrChart({ snrDb, se, seUpperBound, height = 300 }: SeVsSnrChartProps) {
  const option = useMemo<EChartsOption>(() => {
    const accent = cssVar("--accent");
    const bound = cssVar("--series-bound");

    // "gap" series: the bound minus the achieved SE, stacked on top of SE, to shade the difference.
    const gap = snrDb.map((_, i) => Math.max(0, (seUpperBound[i] ?? 0) - (se[i] ?? 0)));

    return {
      textStyle: chartTextStyle(),
      tooltip: {
        trigger: "axis",
        valueFormatter: (v) => `${Number(v).toFixed(3)} bit/s/Hz`,
      },
      legend: {
        data: ["Achieved SE", "Eigen bound"],
        top: 0,
        textStyle: chartTextStyle(),
      },
      grid: chartGrid(),
      xAxis: {
        type: "category",
        data: snrDb.map((v) => v.toFixed(0)),
        name: "SNR (dB)",
        nameLocation: "middle",
        nameGap: 28,
        axisLabel: { color: cssVar("--text-muted") },
        axisLine: { lineStyle: { color: cssVar("--border-strong") } },
      },
      yAxis: {
        type: "value",
        name: "bit/s/Hz",
        nameLocation: "middle",
        nameGap: 40,
        axisLabel: { color: cssVar("--text-muted") },
        splitLine: { lineStyle: { color: cssVar("--border") } },
      },
      series: [
        {
          name: "Achieved SE",
          type: "line",
          data: se,
          symbolSize: 6,
          lineStyle: { color: accent, width: 2.5 },
          itemStyle: { color: accent },
          z: 3,
        },
        {
          name: "Achieved SE (base)",
          type: "line",
          data: se,
          stack: "gap",
          symbol: "none",
          lineStyle: { opacity: 0 },
          areaStyle: { opacity: 0 },
          silent: true,
          tooltip: { show: false },
          z: 1,
        },
        {
          name: "Gap to bound",
          type: "line",
          data: gap,
          stack: "gap",
          symbol: "none",
          lineStyle: { opacity: 0 },
          areaStyle: { color: accent, opacity: 0.08 },
          z: 1,
        },
        {
          name: "Eigen bound",
          type: "line",
          data: seUpperBound,
          symbol: "none",
          lineStyle: { color: bound, width: 1.5, type: "dashed" },
          z: 2,
        },
      ],
    };
  }, [snrDb, se, seUpperBound]);

  if (!snrDb?.length) {
    return <div className="empty-state" style={{ padding: 24 }}>No SNR sweep data.</div>;
  }

  return <Chart option={option} height={height} ariaLabel="Spectral efficiency vs SNR" />;
}
