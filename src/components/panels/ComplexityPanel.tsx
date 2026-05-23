"use client";
import { motion } from "framer-motion";
import { TrendingUp, HardDrive, Zap, Clock, Info } from "lucide-react";
import { C } from "@/lib/utils";
import type { AlgorithmMeta } from "@/types";

function complexityColor(c: string) {
  if (c.includes("1")) return C.green;
  if (c.includes("log")) return C.green;
  if (c.includes("√")) return C.blue;
  if (c === "O(n)") return C.blue;
  if (c.includes("n log") || c.includes("V+E")) return C.amber;
  if (c.includes("n²") || c.includes("nm") || c.includes("nW") || c.includes("nk")) return C.red;
  if (c.includes("2ⁿ") || c.includes("n!")) return "hsl(346 87% 50%)";
  return C.text2;
}

function complexityLabel(c: string) {
  if (c === "O(1)") return "Constant";
  if (c === "O(log n)") return "Logarithmic";
  if (c === "O(√n)") return "Sub-linear";
  if (c === "O(n)") return "Linear";
  if (c === "O(n log n)") return "Linearithmic";
  if (c.includes("n²")) return "Quadratic";
  if (c.includes("2ⁿ")) return "Exponential";
  if (c.includes("n!")) return "Factorial";
  return "Polynomial";
}

function complexityPct(c: string) {
  if (c.includes("1")) return 5;
  if (c.includes("log")) return 15;
  if (c.includes("√")) return 25;
  if (c === "O(n)") return 38;
  if (c.includes("n log") || c.includes("V+E")) return 55;
  if (c.includes("nk") || c.includes("nW") || c.includes("nm")) return 68;
  if (c.includes("n²")) return 80;
  if (c.includes("2ⁿ")) return 93;
  if (c.includes("n!")) return 100;
  return 50;
}

function ComplexityRow({ label, value, icon, delay=0 }: { label: string; value: string; icon: React.ReactNode; delay?: number }) {
  const color = complexityColor(value);
  const pct = complexityPct(value);
  return (
    <motion.div initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }}
      transition={{ delay, duration: 0.3 }} style={{ display: "flex", flexDirection: "column", gap: 5 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, color: C.text3 }}>
          <span style={{ color: C.text4 }}>{icon}</span>{label}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 11, color: C.text4 }}>{complexityLabel(value)}</span>
          <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 13, fontWeight: 700, color }}>{value}</span>
        </div>
      </div>
      <div style={{ height: 3, borderRadius: 99, background: C.surfaceRaised, overflow: "hidden" }}>
        <motion.div style={{ height: "100%", borderRadius: 99, background: color }}
          initial={{ width: 0 }} animate={{ width: `${pct}%` }}
          transition={{ delay: delay + 0.1, duration: 0.6, ease: [0.16, 1, 0.3, 1] }} />
      </div>
    </motion.div>
  );
}

export function ComplexityPanel({ algo }: { algo: AlgorithmMeta }) {
  const { complexity, stable, inPlace, tags } = algo;
  const chips = [
    ...(stable !== undefined ? [{ label: stable ? "Stable" : "Unstable", color: stable ? C.green : C.text3 }] : []),
    ...(inPlace !== undefined ? [{ label: inPlace ? "In-place" : "Extra space", color: inPlace ? C.blue : C.text3 }] : []),
    ...tags.slice(0, 3).map((t) => ({ label: t, color: C.text3 })),
  ];

  return (
    <div style={{ borderRadius: 12, background: C.surface, border: `1px solid ${C.border}`, overflow: "hidden" }}>
      <div style={{ padding: "10px 14px", borderBottom: `1px solid ${C.border}`, background: C.bgSubtle,
        display: "flex", alignItems: "center", gap: 7 }}>
        <TrendingUp size={13} color={C.text3} />
        <span style={{ fontSize: 12, fontWeight: 600, color: C.text2 }}>Complexity Analysis</span>
      </div>
      <div style={{ padding: 14, display: "flex", flexDirection: "column", gap: 14 }}>
        <div>
          <p style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase",
            color: C.text4, marginBottom: 10 }}>Time Complexity</p>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <ComplexityRow label="Best Case"   value={complexity.time.best}    icon={<Zap size={11}/>}        delay={0} />
            <ComplexityRow label="Average"     value={complexity.time.average} icon={<Clock size={11}/>}      delay={0.05} />
            <ComplexityRow label="Worst Case"  value={complexity.time.worst}   icon={<TrendingUp size={11}/>} delay={0.1} />
          </div>
        </div>
        <div style={{ height: 1, background: C.borderSubtle }} />
        <ComplexityRow label="Space (aux)" value={complexity.space} icon={<HardDrive size={11}/>} delay={0.15} />
        <div style={{ height: 1, background: C.borderSubtle }} />
        <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
          {chips.map(({ label, color }) => (
            <span key={label} style={{ fontSize: 10, padding: "3px 8px", borderRadius: 99,
              background: C.surfaceRaised, border: `1px solid ${C.border}`, color,
              fontFamily: "'JetBrains Mono', monospace", fontWeight: 600 }}>
              {label}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

export function DescriptionPanel({ algo }: { algo: AlgorithmMeta }) {
  return (
    <div style={{ borderRadius: 12, background: C.surface, border: `1px solid ${C.border}`, overflow: "hidden" }}>
      <div style={{ padding: "10px 14px", borderBottom: `1px solid ${C.border}`, background: C.bgSubtle,
        display: "flex", alignItems: "center", gap: 7 }}>
        <Info size={13} color={C.text3} />
        <span style={{ fontSize: 12, fontWeight: 600, color: C.text2 }}>About</span>
      </div>
      <div style={{ padding: 14, display: "flex", flexDirection: "column", gap: 12 }}>
        <p style={{ fontSize: 13, color: C.text2, lineHeight: 1.7 }}>{algo.description}</p>
        <div style={{ borderRadius: 8, background: `${C.accent}08`, border: `1px solid ${C.accent}20`, padding: 12 }}>
          <p style={{ fontSize: 10, fontWeight: 700, color: C.accent, marginBottom: 5, textTransform: "uppercase", letterSpacing: "0.08em" }}>Key Insight</p>
          <p style={{ fontSize: 12, color: C.text3, lineHeight: 1.6 }}>{algo.keyInsight}</p>
        </div>
        {algo.useCases.length > 0 && (
          <div>
            <p style={{ fontSize: 10, fontWeight: 700, color: C.text4, textTransform: "uppercase",
              letterSpacing: "0.1em", marginBottom: 7 }}>Use Cases</p>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
              {algo.useCases.map((uc) => (
                <span key={uc} style={{ fontSize: 11, padding: "3px 8px", borderRadius: 99,
                  background: C.surfaceRaised, border: `1px solid ${C.border}`, color: C.text3 }}>{uc}</span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
