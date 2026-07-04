import { useState, type ReactNode } from "react";
import { NavLink } from "react-router-dom";
import { Icon, type IconName } from "./Icon";
import { useTheme } from "../hooks/useTheme";

interface NavItem {
  to: string;
  label: string;
  icon: IconName;
  end?: boolean;
}

const NAV_ITEMS: NavItem[] = [
  { to: "/", label: "Overview", icon: "home", end: true },
  { to: "/codebooks", label: "Library", icon: "library" },
  { to: "/playground", label: "Playground", icon: "playground" },
  { to: "/compare", label: "Compare", icon: "compare" },
  { to: "/figures", label: "Figure Lab", icon: "figures" },
  { to: "/glossary", label: "Glossary", icon: "glossary" },
];

const SIDEBAR_STORAGE_KEY = "csi-studio-sidebar-collapsed";

export function AppShell({ children }: { children: ReactNode }) {
  const [collapsed, setCollapsed] = useState<boolean>(() => {
    if (typeof window === "undefined") return false;
    return window.localStorage.getItem(SIDEBAR_STORAGE_KEY) === "1";
  });
  const [theme, toggleTheme] = useTheme();

  function toggleCollapsed() {
    setCollapsed((c) => {
      const next = !c;
      window.localStorage.setItem(SIDEBAR_STORAGE_KEY, next ? "1" : "0");
      return next;
    });
  }

  return (
    <div className="app-shell">
      <aside className={`sidebar${collapsed ? " collapsed" : ""}`}>
        <div className="sidebar-brand">
          <div className="sidebar-brand-mark">CS</div>
          {!collapsed && <span className="sidebar-brand-text">CSI Codebook Studio</span>}
        </div>
        <nav className="sidebar-nav" aria-label="Primary">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) => `sidebar-link${isActive ? " active" : ""}`}
              title={collapsed ? item.label : undefined}
            >
              <Icon name={item.icon} />
              {!collapsed && <span>{item.label}</span>}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-footer">
          <button
            className="theme-toggle"
            onClick={toggleTheme}
            aria-label={theme === "light" ? "Switch to dark theme" : "Switch to light theme"}
          >
            <Icon name={theme === "light" ? "moon" : "sun"} />
            {!collapsed && <span>{theme === "light" ? "Dark" : "Light"}</span>}
          </button>
          <button
            className="sidebar-toggle"
            onClick={toggleCollapsed}
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            <Icon name={collapsed ? "chevron-right" : "chevron-left"} />
          </button>
        </div>
      </aside>
      <div className="main-column">{children}</div>
    </div>
  );
}
