import type { PmiField } from "../../api/types";

interface PmiTableProps {
  fields: PmiField[];
}

export function PmiTable({ fields }: PmiTableProps) {
  if (!fields?.length) {
    return <div className="empty-state" style={{ padding: 24 }}>No PMI fields reported.</div>;
  }
  return (
    <table className="table">
      <thead>
        <tr>
          <th>Field</th>
          <th>Bits</th>
          <th>Summary</th>
          <th>Description</th>
        </tr>
      </thead>
      <tbody>
        {fields.map((f, i) => (
          <tr key={`${f.name}-${i}`}>
            <td style={{ fontWeight: 600, fontFamily: "var(--font-mono)", fontSize: 12.5 }}>{f.name}</td>
            <td className="tabular-nums">{f.bits ?? "—"}</td>
            <td>{f.value}</td>
            <td className="text-secondary">{f.description || "—"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
