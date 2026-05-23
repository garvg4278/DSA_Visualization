"use client";
import * as React from "react";
import { cn } from "@/lib/utils";
import { C } from "@/lib/utils";

// ── Button ────────────────────────────────────────────────────────────────────
type ButtonVariant = "default"|"secondary"|"ghost"|"danger"|"outline"|"success";
type ButtonSize = "xs"|"sm"|"md"|"default"|"lg"|"xl"|"icon"|"icon-sm"|"icon-lg";

const variantStyles: Record<ButtonVariant, React.CSSProperties> = {
  default:   { background: C.accent, color: "#fff" },
  secondary: { background: C.surfaceRaised, border: `1px solid ${C.border}`, color: C.text2 },
  ghost:     { background: "transparent", color: C.text3 },
  danger:    { background: `${C.red}18`, border: `1px solid ${C.red}50`, color: C.red },
  outline:   { background: "transparent", border: `1px solid ${C.border}`, color: C.text2 },
  success:   { background: `${C.green}15`, border: `1px solid ${C.green}40`, color: C.green },
};

const sizeStyles: Record<ButtonSize, React.CSSProperties> = {
  xs:        { height: 24, padding: "0 8px", fontSize: 11, borderRadius: 6 },
  sm:        { height: 28, padding: "0 12px", fontSize: 12, borderRadius: 8 },
  md:        { height: 32, padding: "0 14px", fontSize: 13, borderRadius: 8 },
  default:   { height: 36, padding: "0 16px", fontSize: 14, borderRadius: 8 },
  lg:        { height: 40, padding: "0 20px", fontSize: 15, borderRadius: 10 },
  xl:        { height: 48, padding: "0 24px", fontSize: 16, borderRadius: 12 },
  icon:      { height: 32, width: 32, padding: 0, fontSize: 14, borderRadius: 8 },
  "icon-sm": { height: 28, width: 28, padding: 0, fontSize: 13, borderRadius: 7 },
  "icon-lg": { height: 40, width: 40, padding: 0, fontSize: 15, borderRadius: 10 },
};

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  loading?: boolean;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ variant = "default", size = "default", loading, children, disabled, style, className, ...props }, ref) => (
    <button
      ref={ref}
      disabled={disabled || loading}
      style={{
        display: "inline-flex", alignItems: "center", justifyContent: "center",
        gap: 6, fontFamily: "'Outfit', sans-serif", fontWeight: 600,
        cursor: "pointer", border: "none", transition: "all 0.15s",
        ...variantStyles[variant], ...sizeStyles[size], ...style,
      }}
      {...props}
    >
      {loading && (
        <svg className="animate-spin" style={{ width: 14, height: 14 }} viewBox="0 0 24 24" fill="none">
          <circle opacity="0.25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path opacity="0.75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
      )}
      {children}
    </button>
  )
);
Button.displayName = "Button";

// ── Badge ─────────────────────────────────────────────────────────────────────
type BadgeVariant = "default"|"green"|"amber"|"red"|"blue"|"purple"|"muted";

const badgeColors: Record<BadgeVariant, { bg: string; text: string; border: string }> = {
  default: { bg: `${C.accent}18`, text: C.accent, border: `${C.accent}30` },
  green:   { bg: `${C.green}18`,  text: C.green,  border: `${C.green}30` },
  amber:   { bg: `${C.amber}18`,  text: C.amber,  border: `${C.amber}30` },
  red:     { bg: `${C.red}18`,    text: C.red,    border: `${C.red}30` },
  blue:    { bg: `${C.blue}18`,   text: C.blue,   border: `${C.blue}30` },
  purple:  { bg: `${C.purple}18`, text: C.purple, border: `${C.purple}30` },
  muted:   { bg: C.surfaceRaised, text: C.text3,  border: C.border },
};

export function Badge({ variant = "default", children, style, ...props }: React.HTMLAttributes<HTMLSpanElement> & { variant?: BadgeVariant }) {
  const { bg, text, border } = badgeColors[variant];
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 4, borderRadius: 99,
      fontFamily: "'JetBrains Mono', monospace", fontSize: 10, fontWeight: 600,
      letterSpacing: "0.08em", textTransform: "uppercase", padding: "2px 8px",
      background: bg, color: text, border: `1px solid ${border}`, ...style }} {...props}>
      {children}
    </span>
  );
}

// ── Card ──────────────────────────────────────────────────────────────────────
export function Card({ className, children, style, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div style={{ borderRadius: 12, background: C.surface, border: `1px solid ${C.border}`, ...style }}
      className={className} {...props}>{children}</div>
  );
}

export function CardHeader({ style, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div style={{ padding: "12px 16px", borderBottom: `1px solid ${C.border}`, ...style }} {...props} />;
}
export function CardContent({ style, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div style={{ padding: 16, ...style }} {...props} />;
}

// ── Skeleton ──────────────────────────────────────────────────────────────────
export function Skeleton({ style, className }: { style?: React.CSSProperties; className?: string }) {
  return <div className={cn("animate-shimmer", className)} style={{ borderRadius: 8, background: C.surfaceRaised, ...style }} />;
}

// ── Kbd ───────────────────────────────────────────────────────────────────────
export function Kbd({ children }: { children: React.ReactNode }) {
  return (
    <kbd style={{ display: "inline-flex", alignItems: "center", padding: "1px 6px", borderRadius: 5,
      border: `1px solid ${C.border}`, background: C.surfaceRaised,
      fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: C.text3 }}>
      {children}
    </kbd>
  );
}

// ── Tooltip ───────────────────────────────────────────────────────────────────
export function Tooltip({ content, children }: { content: React.ReactNode; children: React.ReactNode }) {
  return <div className="relative group" style={{ display: "inline-flex" }}>
    {children}
    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1.5 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50"
      style={{ background: C.surfaceOverlay, border: `1px solid ${C.border}`, borderRadius: 6,
        padding: "4px 8px", fontSize: 11, color: C.text2, whiteSpace: "nowrap",
        fontFamily: "'JetBrains Mono', monospace" }}>
      {content}
    </div>
  </div>;
}
