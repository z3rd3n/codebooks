import type { ReactNode } from "react";

const RELEASE_CLASS: Record<string, string> = {
  R15: "badge-r15",
  R16: "badge-r16",
  R17: "badge-r17",
  R18: "badge-r18",
  R19: "badge-r19",
};

export function ReleaseBadge({ release }: { release: string }) {
  const cls = RELEASE_CLASS[release] ?? "badge-neutral";
  return <span className={`badge ${cls}`}>{release}</span>;
}

export function ReleaseDot({ release }: { release: string }) {
  const dotCls = `dot-${release.toLowerCase()}`;
  return (
    <span
      aria-hidden
      style={{
        display: "inline-block",
        width: 8,
        height: 8,
        borderRadius: "50%",
      }}
      className={dotCls}
    />
  );
}

export function Badge({ children }: { children: ReactNode }) {
  return <span className="badge badge-neutral">{children}</span>;
}
