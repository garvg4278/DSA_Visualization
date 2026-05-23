"use client";
import { useMemo } from "react";
import { motion } from "framer-motion";
import { C } from "@/lib/utils";
import type { ArrayStep, StepType } from "@/types";

function getBarColor(index: number, step: ArrayStep | null): { bg: string; glow: boolean } {
  if (!step) return { bg: C.vizDefault, glow: false };
  const { type, indices } = step;
  const aux = step.auxiliaryData as Record<string, unknown> | undefined;
  const sortedIndices: number[] = Array.isArray(aux?.sortedIndices) ? (aux.sortedIndices as number[]) : [];
  const sortedFrom = typeof aux?.sortedFrom === "number" ? aux.sortedFrom : -1;
  const pivotIdx = typeof aux?.pivot === "number" ? aux.pivot : -1;
  const currentExtreme = typeof aux?.currentExtreme === "number" ? aux.currentExtreme : -1;
  const foundAt = typeof aux?.foundAt === "number" ? aux.foundAt : -1;
  if (type === "sorted" && indices.includes(index)) return { bg: C.vizSorted, glow: true };
  if (sortedIndices.includes(index)) return { bg: C.vizSorted, glow: false };
  if (sortedFrom >= 0 && index >= sortedFrom) return { bg: C.vizSorted, glow: false };
  if ((type === "found" && foundAt === index) || (type === "found" && indices.includes(index))) return { bg: C.vizFound, glow: true };
  if (type === "not-found") return { bg: "hsl(220 14% 22%)", glow: false };
  if (pivotIdx === index) return { bg: C.vizPivot, glow: true };
  if (currentExtreme === index) return { bg: C.vizPivot, glow: false };
  if (!indices.includes(index)) return { bg: C.vizDefault, glow: false };
  const colorMap: Partial<Record<StepType, { bg: string; glow: boolean }>> = {
    swap:      { bg: C.vizSwap,    glow: true },
    compare:   { bg: C.vizCompare, glow: false },
    pivot:     { bg: C.vizPivot,   glow: true },
    merge:     { bg: C.blue,       glow: false },
    partition: { bg: C.purple,     glow: false },
    highlight: { bg: C.vizCompare, glow: false },
    overwrite: { bg: C.blue,       glow: false },
  };
  return colorMap[type as StepType] ?? { bg: C.vizDefault, glow: false };
}

const LEGEND = [
  { color: C.vizDefault, label: "Default" },
  { color: C.vizCompare, label: "Comparing" },
  { color: C.vizSwap,    label: "Swap" },
  { color: C.vizPivot,   label: "Pivot" },
  { color: C.vizSorted,  label: "Sorted" },
  { color: C.vizFound,   label: "Found" },
];

interface ArrayVisualizerProps {
  step: ArrayStep | null;
  baseArray?: number[];
  showIndices?: boolean;
  showValues?: boolean;
  height?: number;
}

export function ArrayVisualizer({ step, baseArray = [], showIndices = false, showValues = true, height = 280 }: ArrayVisualizerProps) {
  const array = step?.array ?? baseArray;
  const maxVal = useMemo(() => Math.max(...array, 1), [array]);

  if (array.length === 0) {
    return (
      <div style={{ height, borderRadius: 12, background: C.surface, border: `1px solid ${C.border}`,
        display: "flex", alignItems: "center", justifyContent: "center",
        color: C.text4, fontSize: 13, fontFamily: "'JetBrains Mono', monospace" }}>
        Configure inputs and press Visualize
      </div>
    );
  }

  const barAreaH = height - (showValues ? 28 : 12) - (showIndices ? 18 : 0) - 24;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      {/* Chart */}
      <div style={{ height, borderRadius: 12, background: C.surface, border: `1px solid ${C.border}`,
        padding: "12px 14px 8px", display: "flex", flexDirection: "column", position: "relative", overflow: "hidden" }}>
        {/* Subtle grid lines */}
        {[25, 50, 75].map(p => (
          <div key={p} style={{ position: "absolute", left: 14, right: 14,
            bottom: `calc(${p}% * ${barAreaH / height} + 8px + ${showIndices ? 18 : 0}px)`,
            height: 1, background: `${C.border}50`, pointerEvents: "none" }} />
        ))}

        {/* Bars container */}
        <div style={{ display: "flex", alignItems: "flex-end", gap: 2, flex: 1, position: "relative", zIndex: 1 }}>
          {array.map((value, index) => {
            const { bg, glow } = getBarColor(index, step);
            const barH = Math.max((value / maxVal) * barAreaH, 4);
            const fontSize = Math.max(9, Math.min(12, Math.floor(560 / array.length / 1.8)));

            return (
              <div key={index} style={{ display: "flex", flexDirection: "column", alignItems: "center",
                justifyContent: "flex-end", flex: 1, minWidth: 0, height: "100%" }}>
                {showValues && array.length <= 22 && (
                  <span style={{ fontFamily: "'JetBrains Mono', monospace", color: C.text3,
                    marginBottom: 3, fontSize, lineHeight: 1 }}>{value}</span>
                )}
                <motion.div layout transition={{ layout: { duration: 0.22, ease: [0.4,0,0.2,1] } }}
                  style={{ width: "100%", borderRadius: "3px 3px 0 0", position: "relative",
                    background: bg, height: barH, minHeight: 4,
                    boxShadow: glow ? `0 0 10px ${bg}70, 0 -2px 14px ${bg}30` : undefined }}>
                  <div style={{ position: "absolute", inset: 0, background: "linear-gradient(to bottom, rgba(255,255,255,0.1), transparent)",
                    borderRadius: "3px 3px 0 0", pointerEvents: "none" }} />
                </motion.div>
                {showIndices && (
                  <span style={{ fontFamily: "'JetBrains Mono', monospace", color: C.text4,
                    marginTop: 3, fontSize: Math.max(8, fontSize - 1) }}>{index}</span>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Legend */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: "6px 14px", paddingLeft: 2 }}>
        {LEGEND.map(({ color, label }) => (
          <div key={label} style={{ display: "flex", alignItems: "center", gap: 5 }}>
            <div style={{ width: 9, height: 9, borderRadius: 2, background: color, flexShrink: 0 }} />
            <span style={{ fontSize: 10, color: C.text4, fontFamily: "'JetBrains Mono', monospace" }}>{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
