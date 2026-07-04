import { useEffect, useRef } from "react";
import * as echarts from "echarts";
import type { EChartsOption } from "echarts";

interface ChartProps {
  option: EChartsOption;
  height?: number | string;
  loading?: boolean;
  ariaLabel?: string;
}

/** Thin wrapper around echarts (imported directly, no echarts-for-react). */
export function Chart({ option, height = 320, loading = false, ariaLabel }: ChartProps) {
  const ref = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    if (!ref.current) return;
    const theme = document.documentElement.getAttribute("data-theme");
    const chart = echarts.init(ref.current, theme === "dark" ? "dark" : undefined, {
      renderer: "svg",
    });
    chartRef.current = chart;

    const onResize = () => chart.resize();
    window.addEventListener("resize", onResize);

    const observer = new ResizeObserver(() => chart.resize());
    observer.observe(ref.current);

    return () => {
      window.removeEventListener("resize", onResize);
      observer.disconnect();
      chart.dispose();
      chartRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    chartRef.current?.setOption(option, true);
  }, [option]);

  useEffect(() => {
    if (loading) chartRef.current?.showLoading({ text: "" });
    else chartRef.current?.hideLoading();
  }, [loading]);

  return (
    <div
      ref={ref}
      role="img"
      aria-label={ariaLabel}
      style={{ width: "100%", height }}
    />
  );
}
