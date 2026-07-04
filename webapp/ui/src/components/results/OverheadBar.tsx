import { useMemo } from "react";
import type { EChartsOption } from "echarts";
import { Chart } from "../Chart";
import { seriesColors, chartTextStyle, cssVar } from "../../utils/chartTheme";

interface OverheadBarProps {
  overheadBits: Record<string, number>;
  fieldDescriptions?: Record<string, string>;
  height?: number;
}

export function OverheadBar({ overheadBits, fieldDescriptions, height = 140 }: OverheadBarProps) {
  const entries = useMemo(() => Object.entries(overheadBits ?? {}), [overheadBits]);

  const option = useMemo<EChartsOption>(() => {
    const colors = seriesColors();
    const total = entries.reduce((sum, [, v]) => sum + v, 0) || 1;

    return {
      textStyle: chartTextStyle(),
      tooltip: {
        trigger: "item",
        formatter: (p) => {
          const item = p as { seriesName: string; value: number };
          const pct = ((item.value / total) * 100).toFixed(1);
          const desc = fieldDescriptions?.[item.seriesName];
          return `<strong>${item.seriesName}</strong>: ${item.value} bits (${pct}%)${desc ? `<br/><span style="opacity:.75">${desc}</span>` : ""}`;
        },
      },
      legend: {
        top: 0,
        textStyle: chartTextStyle(),
        type: "scroll",
      },
      grid: { left: 8, right: 8, top: 40, bottom: 8, containLabel: true },
      xAxis: {
        type: "value",
        show: false,
      },
      yAxis: {
        type: "category",
        data: ["Feedback report"],
        show: false,
      },
      series: entries.map(([name, value], i) => ({
        name,
        type: "bar",
        stack: "total",
        data: [value],
        itemStyle: { color: colors[i % colors.length] },
        barWidth: 42,
        label: {
          show: value / total > 0.06,
          formatter: () => `${value}`,
          color: cssVar("--text-inverse"),
          fontSize: 11,
        },
      })),
    };
  }, [entries, fieldDescriptions]);

  if (!entries.length) {
    return <div className="empty-state" style={{ padding: 24 }}>No overhead breakdown available.</div>;
  }

  return <Chart option={option} height={height} ariaLabel="PMI overhead breakdown" />;
}
