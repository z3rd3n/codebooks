interface SkeletonProps {
  width?: string | number;
  height?: string | number;
  radius?: string | number;
  style?: React.CSSProperties;
}

export function Skeleton({ width = "100%", height = 16, radius, style }: SkeletonProps) {
  return (
    <div
      className="skeleton"
      style={{ width, height, borderRadius: radius, ...style }}
    />
  );
}

export function SkeletonCard() {
  return (
    <div className="card">
      <Skeleton height={18} width="55%" />
      <div style={{ marginTop: 10 }}>
        <Skeleton height={12} width="90%" />
      </div>
      <div style={{ marginTop: 6 }}>
        <Skeleton height={12} width="75%" />
      </div>
      <div style={{ marginTop: 14, display: "flex", gap: 6 }}>
        <Skeleton height={20} width={56} radius={999} />
        <Skeleton height={20} width={56} radius={999} />
      </div>
    </div>
  );
}

export function SkeletonGrid({ count = 6 }: { count?: number }) {
  return (
    <div className="card-grid">
      {Array.from({ length: count }, (_, i) => (
        <SkeletonCard key={i} />
      ))}
    </div>
  );
}
