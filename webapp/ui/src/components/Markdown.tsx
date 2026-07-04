import type { ComponentPropsWithoutRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

interface MarkdownProps {
  text: string;
  className?: string;
  /** Render relative/in-repo links (e.g. `foo.md`, `../src/x.py`) as plain
   * text instead of broken navigations. External http(s) links stay clickable
   * and open in a new tab. Used by the Deep dive tab. */
  flattenRelativeLinks?: boolean;
}

function isExternal(href: string): boolean {
  return /^(https?:)?\/\//i.test(href) || href.startsWith("mailto:");
}

/** Renders markdown + GFM tables + inline/block KaTeX math ($...$ / $$...$$). */
export function Markdown({ text, className, flattenRelativeLinks }: MarkdownProps) {
  const components = flattenRelativeLinks
    ? {
        a({ href, children, ...rest }: ComponentPropsWithoutRef<"a">) {
          if (href && isExternal(href)) {
            return (
              <a href={href} target="_blank" rel="noopener noreferrer" {...rest}>
                {children}
              </a>
            );
          }
          // In-repo / relative reference: show the text, but not as a link.
          return <span className="doc-ref">{children}</span>;
        },
      }
    : undefined;

  return (
    <div className={`markdown-body${className ? ` ${className}` : ""}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={components}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}
