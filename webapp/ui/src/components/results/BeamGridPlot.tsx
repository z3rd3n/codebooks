import { useMemo } from "react";
import type { EChartsOption } from "echarts";
import type { BeamGrid } from "../../api/types";
import { Chart } from "../Chart";
import { cssVar, chartTextStyle } from "../../utils/chartTheme";

interface BeamGridPlotProps {
  beamGrid: BeamGrid;
  height?: number;
}

export function BeamGridPlot({ beamGrid, height = 280 }: BeamGridPlotProps) {
  const option = useMemo<EChartsOption>(() => {
    const { g1, g2, selected } = beamGrid;
    const allPoints: [number, number][] = [];
    for (let l = 0; l < g1; l++) {
      for (let m = 0; m < g2; m++) {
        allPoints.push([l, m]);
      }
    }
    const selectedSet = new Set(selected.map(([l, m]) => `${l},${m}`));
    const unselected = allPoints.filter(([l, m]) => !selectedSet.has(`${l},${m}`));

    return {
      textStyle: chartTextStyle(),
      tooltip: {
        formatter: (p) => {
          const d = (p as { data: number[] }).data;
          return `beam (${d[0]}, ${d[1]})`;
        },
      },
      grid: { left: 48, right: 20, top: 20, bottom: 44, containLabel: true },
      xAxis: {
        type: "value",
        name: "beam index l1",
        nameLocation: "middle",
        nameGap: 28,
        min: -0.5,
        max: g1 - 0.5,
        axisLabel: { color: cssVar("--text-muted") },
        splitLine: { lineStyle: { color: cssVar("--border") } },
      },
      yAxis: {
        type: "value",
        name: "beam index l2",
        nameLocation: "middle",
        nameGap: 34,
        min: -0.5,
        max: g2 - 0.5,
        axisLabel: { color: cssVar("--text-muted") },
        splitLine: { lineStyle: { color: cssVar("--border") } },
      },
      series: [
        {
          name: "Available beams",
          type: "scatter",
          data: unselected,
          symbolSize: 6,
          itemStyle: { color: cssVar("--border-strong") },
        },
        {
          name: "Selected beams",
          type: "scatter",
          data: selected,
          symbolSize: 14,
          itemStyle: { color: cssVar("--accent") },
        },
      ],
    };
  }, [beamGrid]);

  return <Chart option={option} height={height} ariaLabel="Selected DFT beam grid" />;
}
