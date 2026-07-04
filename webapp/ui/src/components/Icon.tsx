import type { SVGProps } from "react";

export type IconName =
  | "home"
  | "library"
  | "playground"
  | "compare"
  | "figures"
  | "glossary"
  | "chevron-left"
  | "chevron-right"
  | "chevron-down"
  | "sun"
  | "moon"
  | "search"
  | "check"
  | "close"
  | "warning"
  | "info"
  | "external"
  | "copy"
  | "download"
  | "play"
  | "spinner"
  | "empty"
  | "arrow-right"
  | "plus"
  | "grid"
  | "antenna"
  | "expand";

interface IconProps extends SVGProps<SVGSVGElement> {
  name: IconName;
  size?: number;
}

const paths: Record<IconName, JSX.Element> = {
  home: (
    <path
      d="M3 10.5 10 4l7 6.5M5 9v7h10V9M8.5 16v-4h3v4"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  ),
  library: (
    <g strokeLinecap="round" strokeLinejoin="round">
      <rect x="3.5" y="3.5" width="4" height="13" rx="0.5" />
      <rect x="8.5" y="3.5" width="4" height="13" rx="0.5" />
      <path d="M14 5.2 17 4.5l2 12-3 0.7" />
    </g>
  ),
  playground: (
    <g strokeLinecap="round" strokeLinejoin="round">
      <circle cx="10" cy="10" r="6.5" />
      <path d="M10 6.5v3.5l2.4 1.4" />
    </g>
  ),
  compare: (
    <g strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 15V7a1 1 0 0 1 1-1h3v9" />
      <path d="M12 15V4h3a1 1 0 0 1 1 1v10" />
      <path d="M3 15h14" />
    </g>
  ),
  figures: (
    <g strokeLinecap="round" strokeLinejoin="round">
      <rect x="3.5" y="3.5" width="13" height="13" rx="1.5" />
      <path d="M6.5 13 9 9.5l2 2.2L14 8" />
    </g>
  ),
  glossary: (
    <g strokeLinecap="round" strokeLinejoin="round">
      <path d="M5 3.5h8a1 1 0 0 1 1 1V17l-5-2.3L4 17V4.5a1 1 0 0 1 1-1Z" />
    </g>
  ),
  "chevron-left": <path d="M12 4.5 6.5 10l5.5 5.5" strokeLinecap="round" strokeLinejoin="round" />,
  "chevron-right": <path d="M8 4.5 13.5 10 8 15.5" strokeLinecap="round" strokeLinejoin="round" />,
  "chevron-down": <path d="M4.5 7.5 10 13l5.5-5.5" strokeLinecap="round" strokeLinejoin="round" />,
  sun: (
    <g strokeLinecap="round">
      <circle cx="10" cy="10" r="3.4" />
      <path d="M10 2.5v2M10 15.5v2M17.5 10h-2M4.5 10h-2M15.3 4.7l-1.4 1.4M6.1 13.9l-1.4 1.4M15.3 15.3l-1.4-1.4M6.1 6.1 4.7 4.7" />
    </g>
  ),
  moon: <path d="M16 12.5A6.8 6.8 0 0 1 7.5 4 6.8 6.8 0 1 0 16 12.5Z" strokeLinejoin="round" />,
  search: (
    <g strokeLinecap="round" strokeLinejoin="round">
      <circle cx="8.8" cy="8.8" r="5.3" />
      <path d="m16 16-3.5-3.5" />
    </g>
  ),
  check: <path d="M4.5 10.5 8 14l7.5-8" strokeLinecap="round" strokeLinejoin="round" />,
  close: <path d="M5 5l10 10M15 5 5 15" strokeLinecap="round" strokeLinejoin="round" />,
  warning: (
    <g strokeLinecap="round" strokeLinejoin="round">
      <path d="M10 3.5 17.5 16h-15L10 3.5Z" />
      <path d="M10 8.3v3.4" />
      <circle cx="10" cy="14" r="0.6" fill="currentColor" stroke="none" />
    </g>
  ),
  info: (
    <g strokeLinecap="round" strokeLinejoin="round">
      <circle cx="10" cy="10" r="7" />
      <path d="M10 9v4.2" />
      <circle cx="10" cy="6.6" r="0.6" fill="currentColor" stroke="none" />
    </g>
  ),
  external: (
    <g strokeLinecap="round" strokeLinejoin="round">
      <path d="M8.5 4.5H15.5V11.5" />
      <path d="M15.5 4.5 8 12" />
      <path d="M12.5 9.5V15a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V8a1 1 0 0 1 1-1h5.5" />
    </g>
  ),
  copy: (
    <g strokeLinecap="round" strokeLinejoin="round">
      <rect x="7.5" y="7.5" width="9" height="9" rx="1.3" />
      <path d="M4.5 12.5v-7a1 1 0 0 1 1-1h7" />
    </g>
  ),
  download: (
    <g strokeLinecap="round" strokeLinejoin="round">
      <path d="M10 3.5v9.5M6.2 9.3 10 13l3.8-3.7" />
      <path d="M4 15.5h12" />
    </g>
  ),
  play: <path d="M6.5 4.5 15 10l-8.5 5.5V4.5Z" strokeLinejoin="round" />,
  spinner: (
    <g strokeLinecap="round">
      <path d="M10 3.5v3" opacity="1" />
      <path d="M14.5 5.5 12.6 7.4" opacity="0.85" />
      <path d="M16.5 10h-3" opacity="0.7" />
      <path d="M14.5 14.5 12.6 12.6" opacity="0.55" />
      <path d="M10 16.5v-3" opacity="0.4" />
      <path d="M5.5 14.5l1.9-1.9" opacity="0.3" />
      <path d="M3.5 10h3" opacity="0.2" />
      <path d="M5.5 5.5l1.9 1.9" opacity="0.12" />
    </g>
  ),
  empty: (
    <g strokeLinecap="round" strokeLinejoin="round">
      <rect x="3.5" y="6.5" width="13" height="10" rx="1.5" />
      <path d="M3.5 10h13" />
      <path d="M7 3.5 9 6.5M13 3.5 11 6.5" />
    </g>
  ),
  "arrow-right": <path d="M4 10h11.5M11 5.5 15.5 10 11 14.5" strokeLinecap="round" strokeLinejoin="round" />,
  plus: <path d="M10 4.5v11M4.5 10h11" strokeLinecap="round" />,
  grid: (
    <g>
      <rect x="3.5" y="3.5" width="5" height="5" rx="0.8" />
      <rect x="11.5" y="3.5" width="5" height="5" rx="0.8" />
      <rect x="3.5" y="11.5" width="5" height="5" rx="0.8" />
      <rect x="11.5" y="11.5" width="5" height="5" rx="0.8" />
    </g>
  ),
  antenna: (
    <g strokeLinecap="round" strokeLinejoin="round">
      <path d="M10 3v6M6 6l4-3 4 3" />
      <circle cx="10" cy="11" r="1" fill="currentColor" stroke="none" />
      <path d="M10 12v5M6 17h8" />
    </g>
  ),
  expand: (
    <g strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 8V4h4M16 8V4h-4M4 12v4h4M16 12v4h-4" />
    </g>
  ),
};

export function Icon({ name, size = 16, ...rest }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 20 20"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.6}
      aria-hidden="true"
      focusable="false"
      {...rest}
    >
      {paths[name]}
    </svg>
  );
}
