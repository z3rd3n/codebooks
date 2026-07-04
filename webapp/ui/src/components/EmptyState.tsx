import type { ReactNode } from "react";
import { Icon, type IconName } from "./Icon";

interface EmptyStateProps {
  title: string;
  body?: string;
  icon?: IconName;
  action?: ReactNode;
}

export function EmptyState({ title, body, icon = "empty", action }: EmptyStateProps) {
  return (
    <div className="empty-state">
      <Icon name={icon} size={32} />
      <div className="empty-state-title">{title}</div>
      {body ? <div className="text-sm">{body}</div> : null}
      {action}
    </div>
  );
}
