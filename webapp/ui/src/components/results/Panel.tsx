import type { ReactNode } from "react";
import { Icon } from "../Icon";

interface PanelProps {
  title: string;
  subtitle: string;
  about?: string;
  children: ReactNode;
  right?: ReactNode;
}

/** Standard result panel: title, "what am I looking at" subtitle, content,
 * and an optional "About this chart" collapsible explanation. */
export function Panel({ title, subtitle, about, children, right }: PanelProps) {
  return (
    <div className="panel">
      <div className="panel-head row" style={{ justifyContent: "space-between", alignItems: "flex-start" }}>
        <div>
          <div className="panel-title">{title}</div>
          <div className="panel-subtitle">{subtitle}</div>
        </div>
        {right}
      </div>
      {children}
      {about && (
        <details className="about-toggle">
          <summary>
            <Icon name="info" size={14} /> About this chart
          </summary>
          <div className="about-toggle-body">{about}</div>
        </details>
      )}
    </div>
  );
}
