"use client";
import { motion, AnimatePresence } from "framer-motion";
import { C } from "@/lib/utils";
import { useVisualizerStore } from "@/stores/visualizerStore";
import { useStepDescription } from "@/hooks/useVisualizer";

export function StepDescriptionBar({ style }: { style?: React.CSSProperties }) {
  const playbackState = useVisualizerStore((s) => s.playbackState);
  const currentStep = useVisualizerStore((s) => s.currentStep);
  const description = useStepDescription();
  const isRunning = playbackState === "playing";

  const dotColor = {
    idle: C.text4, playing: C.green, paused: C.amber, finished: C.green,
  }[playbackState] ?? C.text4;

  return (
    <div style={{ display: "flex", alignItems: "flex-start", gap: 10, padding: "10px 14px",
      borderRadius: 10, background: C.surface, border: `1px solid ${C.border}`,
      minHeight: 44, ...style }}>
      <span style={{ marginTop: 5, width: 8, height: 8, borderRadius: "50%", flexShrink: 0,
        background: dotColor, boxShadow: isRunning ? `0 0 8px ${C.green}` : "none",
        animation: isRunning ? "pulse-subtle 1.5s ease-in-out infinite" : "none" }} />
      <AnimatePresence mode="wait">
        <motion.p key={`${currentStep}-${description.slice(0, 15)}`}
          initial={{ opacity: 0, y: 3 }} animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -3 }} transition={{ duration: 0.12 }}
          style={{ fontSize: 12.5, fontFamily: "'JetBrains Mono', monospace",
            color: C.text2, lineHeight: 1.6, flex: 1 }}>
          {description}
        </motion.p>
      </AnimatePresence>
    </div>
  );
}
