import { useState } from "react";
import { Icon } from "../Icon";

interface CodeSnippetProps {
  code: string;
  collapsible?: boolean;
}

export function CodeSnippet({ code, collapsible = true }: CodeSnippetProps) {
  const [copied, setCopied] = useState(false);
  const [open, setOpen] = useState(!collapsible);

  async function copy() {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // clipboard API unavailable — silently ignore
    }
  }

  return (
    <div className="code-block">
      <div className="code-block-head">
        {collapsible ? (
          <button className="btn btn-ghost btn-sm" onClick={() => setOpen((o) => !o)}>
            <Icon name={open ? "chevron-down" : "chevron-right"} size={14} />
            Python equivalent
          </button>
        ) : (
          <span className="text-sm text-secondary">Python equivalent</span>
        )}
        <button className="btn btn-ghost btn-sm" onClick={copy}>
          <Icon name={copied ? "check" : "copy"} size={14} />
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      {open && <pre><code>{code}</code></pre>}
    </div>
  );
}
