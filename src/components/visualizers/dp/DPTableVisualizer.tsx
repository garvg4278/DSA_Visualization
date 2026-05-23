"use client";
import { motion } from "framer-motion";
import { C } from "@/lib/utils";
import type { DPStep } from "@/types";

function getCellStyle(row: number, col: number, step: DPStep | null) {
  if (!step) return { bg: C.surfaceRaised, color: C.text3, ring: false };
  const [ar, ac] = step.activeCell;
  const isActive = ar === row && ac === col;
  const isHighlighted = step.highlightedCells.some(([r,c]) => r===row && c===col);
  if (isActive) return step.type === "dp-match"
    ? { bg: `${C.green}22`, color: C.green, ring: true }
    : { bg: `${C.accent}22`, color: C.accent, ring: true };
  if (isHighlighted) return { bg: `${C.amber}15`, color: C.amber, ring: false };
  const val = step.table[row]?.[col];
  if (val !== -1 && val !== undefined && val !== null) return { bg: C.surface, color: C.text3, ring: false };
  return { bg: C.surfaceRaised, color: C.text4, ring: false };
}

interface DPTableVisualizerProps {
  step: DPStep | null;
  rowLabels?: string[];
  colLabels?: string[];
  maxRows?: number;
  maxCols?: number;
}

export function DPTableVisualizer({ step, rowLabels=[], colLabels=[], maxRows=10, maxCols=12 }: DPTableVisualizerProps) {
  if (!step || step.table.length === 0) {
    return (
      <div style={{ height: 200, borderRadius: 12, background: C.surface, border: `1px solid ${C.border}`,
        display: "flex", alignItems: "center", justifyContent: "center",
        color: C.text4, fontSize: 13, fontFamily: "'JetBrains Mono', monospace" }}>
        Press Visualize to begin
      </div>
    );
  }

  const rows = Math.min(step.table.length, maxRows);
  const cols = Math.min(step.table[0]?.length ?? 0, maxCols);
  const cellSize = Math.max(34, Math.min(52, Math.floor(600 / (cols + 1))));
  const fs = cellSize < 40 ? 10 : 12;

  return (
    <div style={{ borderRadius: 12, background: C.surface, border: `1px solid ${C.border}`, padding: 16, overflowX: "auto" }}>
      <table style={{ borderCollapse: "collapse" }}>
        <thead>
          <tr>
            <th style={{ width: cellSize * 0.7, height: cellSize * 0.6, border: `1px solid ${C.border}`, fontSize: fs, color: C.text4, background: C.bgSubtle }} />
            {Array.from({ length: cols }).map((_, j) => (
              <th key={j} style={{ width: cellSize, height: cellSize * 0.7, border: `1px solid ${C.border}`,
                textAlign: "center", fontSize: fs, fontFamily: "'JetBrains Mono', monospace",
                color: C.text3, background: C.bgSubtle, fontWeight: 600 }}>
                {colLabels[j] ?? j}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {Array.from({ length: rows }).map((_, i) => (
            <tr key={i}>
              <td style={{ width: cellSize * 0.7, height: cellSize, border: `1px solid ${C.border}`,
                textAlign: "center", fontSize: fs, fontFamily: "'JetBrains Mono', monospace",
                color: C.text3, background: C.bgSubtle, fontWeight: 600 }}>
                {rowLabels[i] ?? i}
              </td>
              {Array.from({ length: cols }).map((_, j) => {
                const value = step.table[i]?.[j];
                const { bg, color, ring } = getCellStyle(i, j, step);
                const isEmpty = value === -1 || value === undefined || value === null;
                return (
                  <td key={j} style={{ width: cellSize, height: cellSize, border: `1px solid ${C.border}`, padding: 0, position: "relative" }}>
                    <motion.div animate={{ background: bg }} transition={{ duration: 0.18 }}
                      style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center",
                        boxShadow: ring ? `inset 0 0 0 2px ${color}` : undefined }}>
                      {!isEmpty && (
                        <motion.span key={value} initial={{ scale: 0.6, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}
                          transition={{ duration: 0.18, ease: [0.16,1,0.3,1] }}
                          style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: fs, fontWeight: 600, color }}>
                          {value}
                        </motion.span>
                      )}
                    </motion.div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      {/* Legend */}
      <div style={{ display: "flex", gap: 14, marginTop: 12, paddingTop: 10, borderTop: `1px solid ${C.borderSubtle}` }}>
        {[{ color: C.accent, label: "Active" }, { color: C.green, label: "Match" }, { color: C.amber, label: "Referenced" }, { color: C.text3, label: "Computed" }].map(({ color, label }) => (
          <div key={label} style={{ display: "flex", alignItems: "center", gap: 5 }}>
            <div style={{ width: 9, height: 9, borderRadius: 2, background: color }} />
            <span style={{ fontSize: 10, color: C.text4, fontFamily: "'JetBrains Mono', monospace" }}>{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function FibonacciTable({ step, n }: { step: DPStep | null; n: number }) {
  if (!step || step.table.length === 0) {
    return (
      <div style={{ height: 120, borderRadius: 12, background: C.surface, border: `1px solid ${C.border}`,
        display: "flex", alignItems: "center", justifyContent: "center",
        color: C.text4, fontSize: 13, fontFamily: "'JetBrains Mono', monospace" }}>
        Press Visualize
      </div>
    );
  }
  const row = step.table[0] ?? [];
  const [, ac] = step.activeCell;
  return (
    <div style={{ borderRadius: 12, background: C.surface, border: `1px solid ${C.border}`, padding: 20, overflowX: "auto" }}>
      <div style={{ display: "flex", gap: 6, alignItems: "flex-end" }}>
        {row.map((val, i) => {
          const isActive = ac === i;
          const isHighlighted = step.highlightedCells.some(([, c]) => c === i);
          const isFilled = val !== -1 && val !== undefined;
          return (
            <div key={i} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 5 }}>
              <motion.div animate={{
                background: isActive ? `${C.accent}25` : isHighlighted ? `${C.amber}15` : isFilled ? C.surfaceRaised : C.surface,
                borderColor: isActive ? C.accent : isHighlighted ? C.amber : C.border,
                color: isActive ? C.accent : isHighlighted ? C.amber : C.text2,
              }} style={{ width: 44, height: 44, borderRadius: 8, border: `1px solid`,
                display: "flex", alignItems: "center", justifyContent: "center",
                fontFamily: "'JetBrains Mono', monospace", fontWeight: 700, fontSize: 13 }}>
                {isFilled ? val : ""}
              </motion.div>
              <span style={{ fontSize: 9, fontFamily: "'JetBrains Mono', monospace", color: C.text4 }}>F({i})</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
