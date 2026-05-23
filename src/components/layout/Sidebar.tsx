"use client";
import { useState } from "react";
import Link from "next/link";
import Image from "next/image";
import { usePathname } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import {
  ArrowUpDown, Search, Network, GitBranch, TableProperties,
  ChevronDown, Home, GitCompare, Terminal, Layers, Menu, X,
} from "lucide-react";
import { ALGORITHMS_BY_CATEGORY, CATEGORY_CONFIG, APP_CONFIG } from "@/constants";
import { C } from "@/lib/utils";
import type { AlgorithmCategory } from "@/types";

const CATEGORY_ICONS: Record<AlgorithmCategory, React.ReactNode> = {
  sorting:     <ArrowUpDown size={13} />,
  searching:   <Search size={13} />,
  graph:       <Network size={13} />,
  trees:       <GitBranch size={13} />,
  dp:          <TableProperties size={13} />,
  recursion:   <Layers size={13} />,
  backtracking:<Terminal size={13} />,
};
const CATEGORIES: AlgorithmCategory[] = ["sorting", "searching", "graph", "trees", "dp"];

function CategorySection({ category, defaultOpen = false }: { category: AlgorithmCategory; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  const pathname = usePathname();
  const config = CATEGORY_CONFIG[category];
  const algorithms = ALGORITHMS_BY_CATEGORY[category] ?? [];

  return (
    <div>
      <button
        onClick={() => setOpen(!open)}
        style={{
          width: "100%", display: "flex", alignItems: "center", justifyContent: "space-between",
          padding: "6px 10px", borderRadius: 7, background: "transparent", border: "none",
          cursor: "pointer",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 6, color: config.color }}>
          {CATEGORY_ICONS[category]}
          <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.12em", textTransform: "uppercase", color: C.text3 }}>
            {config.label}
          </span>
        </div>
        <ChevronDown
          size={11}
          style={{ color: C.text4, transition: "transform 0.2s", transform: open ? "rotate(180deg)" : "rotate(0deg)" }}
        />
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.18 }}
            style={{ overflow: "hidden", marginLeft: 8, borderLeft: `1px solid ${C.border}`, paddingLeft: 8, paddingBottom: 2 }}
          >
            {algorithms.map(algo => {
              const href = `/visualizer/${category}/${algo.id}`;
              const active = pathname === href;
              return (
                <Link key={algo.id} href={href} style={{
                  display: "flex", alignItems: "center", gap: 7, padding: "5px 8px", borderRadius: 6,
                  textDecoration: "none", fontSize: 12, fontWeight: 500, marginBottom: 1,
                  background: active ? `${C.accent}12` : "transparent",
                  color: active ? C.accent : C.text3,
                  transition: "all 0.1s",
                }}>
                  <span style={{ width: 5, height: 5, borderRadius: "50%", flexShrink: 0, background: active ? config.color : C.text4 }} />
                  {algo.name}
                </Link>
              );
            })}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function SidebarContent({ onClose }: { onClose?: () => void }) {
  const pathname = usePathname();
  const navLinks = [
    { href: "/",          icon: <Home size={13} />,       label: "Home" },
    { href: "/compare",   icon: <GitCompare size={13} />, label: "Compare" },
    { href: "/playground",icon: <Terminal size={13} />,   label: "Playground" },
  ];

  return (
    <>
      {/* Logo */}
      <div style={{ padding: "14px 14px 12px", borderBottom: `1px solid ${C.border}`, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <Link href="/" style={{ display: "flex", alignItems: "center", gap: 9, textDecoration: "none" }}>
          {/* Real Logo */}
          <div style={{ width: 30, height: 30, borderRadius: 8, overflow: "hidden", flexShrink: 0, background: C.surface }}>
            <Image
              src="/assets/logo/Favicon.png"
              alt="AlgoVista Logo"
              width={30}
              height={30}
              style={{ objectFit: "cover", width: "100%", height: "100%" }}
              priority
            />
          </div>
          <div>
            <div className="gradient-text" style={{ fontSize: 14, fontWeight: 800, lineHeight: 1 }}>
              {APP_CONFIG.name}
            </div>
            <div style={{ fontSize: 9, color: C.text4, fontFamily: "'JetBrains Mono', monospace", marginTop: 1 }}>
              v{APP_CONFIG.version}
            </div>
          </div>
        </Link>
        {onClose && (
          <button onClick={onClose} style={{ background: "transparent", border: "none", cursor: "pointer", color: C.text3, padding: 4 }}>
            <X size={16} />
          </button>
        )}
      </div>

      {/* Nav links */}
      <div style={{ flex: 1, overflowY: "auto", padding: "10px 8px", display: "flex", flexDirection: "column", gap: 14 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
          {navLinks.map(({ href, icon, label }) => {
            const active = pathname === href;
            return (
              <Link key={href} href={href} onClick={onClose} style={{
                display: "flex", alignItems: "center", gap: 8, padding: "7px 10px", borderRadius: 8,
                textDecoration: "none", fontSize: 13, fontWeight: 500,
                background: active ? `${C.accent}12` : "transparent",
                color: active ? C.accent : C.text3,
                transition: "all 0.1s",
              }}>
                <span style={{ opacity: active ? 1 : 0.7 }}>{icon}</span>
                {label}
              </Link>
            );
          })}
        </div>

        <div style={{ height: 1, background: C.border }} />

        <div>
          <p style={{ fontSize: 9, fontWeight: 700, letterSpacing: "0.14em", textTransform: "uppercase", color: C.text4, padding: "0 10px", marginBottom: 7 }}>
            Algorithms
          </p>
          <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
            {CATEGORIES.map((cat, i) => (
              <CategorySection key={cat} category={cat} defaultOpen={i === 0} />
            ))}
          </div>
        </div>
      </div>

      {/* Keyboard shortcuts hint */}
      <div style={{ padding: "8px 12px 10px", borderTop: `1px solid ${C.border}` }}>
        <div style={{ fontSize: 10, color: C.text4, display: "flex", alignItems: "center", gap: 5, flexWrap: "wrap" }}>
          <span>Keys:</span>
          {["Space", "← →", "R", "+/-"].map(k => (
            <kbd key={k} style={{
              background: C.surfaceRaised, border: `1px solid ${C.border}`,
              borderRadius: 4, padding: "1px 5px", fontFamily: "monospace", fontSize: 9,
            }}>{k}</kbd>
          ))}
        </div>
      </div>
    </>
  );
}

export function Sidebar() {
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <>
      {/* Desktop sidebar */}
      <aside
        className="desktop-sidebar"
        style={{
          width: 245, flexShrink: 0, height: "100vh", position: "sticky", top: 0,
          display: "flex", flexDirection: "column",
          background: C.bgSubtle, borderRight: `1px solid ${C.border}`, overflow: "hidden",
        }}
      >
        <SidebarContent />
      </aside>

      {/* Mobile hamburger */}
      <button
        onClick={() => setMobileOpen(true)}
        className="mobile-menu-btn"
        style={{
          position: "fixed", top: 12, left: 12, zIndex: 200,
          width: 36, height: 36, borderRadius: 9,
          alignItems: "center", justifyContent: "center",
          background: C.surface, border: `1px solid ${C.border}`, cursor: "pointer", color: C.text2,
        }}
      >
        <Menu size={18} />
      </button>

      {/* Mobile overlay */}
      <AnimatePresence>
        {mobileOpen && (
          <>
            <motion.div
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              onClick={() => setMobileOpen(false)}
              style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.65)", zIndex: 300 }}
            />
            <motion.aside
              initial={{ x: -260 }} animate={{ x: 0 }} exit={{ x: -260 }}
              transition={{ type: "spring", damping: 25, stiffness: 200 }}
              style={{
                position: "fixed", top: 0, left: 0, bottom: 0, width: 260, zIndex: 400,
                display: "flex", flexDirection: "column",
                background: C.bgSubtle, borderRight: `1px solid ${C.border}`,
              }}
            >
              <SidebarContent onClose={() => setMobileOpen(false)} />
            </motion.aside>
          </>
        )}
      </AnimatePresence>
    </>
  );
}
