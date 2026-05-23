"use client";

import { useCallback, useEffect, useRef } from "react";
import { useVisualizerStore } from "@/stores/visualizerStore";
import { ALGORITHM_MAP } from "@/constants";
import { generateRandomArray } from "@/utils";

export function useAlgorithmMeta(id: string | null) {
  return id ? ALGORITHM_MAP.get(id) ?? null : null;
}

export function usePlaybackControls() {
  const store = useVisualizerStore();
  const {
    playbackState, currentStep, speed,
    play, pause, reset, stepForward, stepBackward, setSpeed,
  } = store;

  const totalSteps = useVisualizerStore((s) => {
    switch (s.activeStepType) {
      case "array": return s.arraySteps.length;
      case "graph": return s.graphSteps.length;
      case "dp":    return s.dpSteps.length;
      case "tree":  return s.treeSteps.length;
      default:      return 0;
    }
  });

  const progress = totalSteps === 0 ? 0 : (currentStep / Math.max(totalSteps - 1, 1)) * 100;
  const isPlaying = playbackState === "playing";
  const isFinished = playbackState === "finished";
  const canStepForward = currentStep < totalSteps - 1;
  const canStepBackward = currentStep > 0;

  const togglePlay = useCallback(() => {
    const s = useVisualizerStore.getState();
    if (s.playbackState === "playing") s.pause();
    else s.play();
  }, []);

  // Use refs so the keyboard handler always has fresh callbacks
  const actionsRef = useRef({ togglePlay, stepForward, stepBackward, reset, setSpeed, speed });
  actionsRef.current = { togglePlay, stepForward, stepBackward, reset, setSpeed, speed };

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      const { togglePlay, stepForward, stepBackward, reset, setSpeed, speed } = actionsRef.current;
      switch (e.key) {
        case " ":          e.preventDefault(); togglePlay(); break;
        case "ArrowRight": e.preventDefault(); stepForward(); break;
        case "ArrowLeft":  e.preventDefault(); stepBackward(); break;
        case "r": case "R": reset(); break;
        case "+": case "=": setSpeed(Math.min(speed + 1, 10)); break;
        case "-":           setSpeed(Math.max(speed - 1, 1));  break;
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []); // empty deps — uses ref for fresh values

  return {
    playbackState, currentStep, totalSteps, speed, progress,
    isPlaying, isFinished, canStepForward, canStepBackward,
    play, pause, reset, stepForward, stepBackward, togglePlay, setSpeed,
  };
}

export function useRandomArray() {
  const setInputArray = useVisualizerStore((s) => s.setInputArray);
  const reset = useVisualizerStore((s) => s.reset);
  return useCallback((size?: number) => {
    setInputArray(generateRandomArray(size ?? 12));
    reset();
  }, [setInputArray, reset]);
}

export function useCurrentArrayStep() {
  return useVisualizerStore((s) =>
    s.activeStepType === "array" ? s.arraySteps[s.currentStep] ?? null : null
  );
}
export function useCurrentGraphStep() {
  return useVisualizerStore((s) =>
    s.activeStepType === "graph" ? s.graphSteps[s.currentStep] ?? null : null
  );
}
export function useCurrentDPStep() {
  return useVisualizerStore((s) =>
    s.activeStepType === "dp" ? s.dpSteps[s.currentStep] ?? null : null
  );
}
export function useCurrentTreeStep() {
  return useVisualizerStore((s) =>
    s.activeStepType === "tree" ? s.treeSteps[s.currentStep] ?? null : null
  );
}

export function useStepDescription(): string {
  const arrayStep = useCurrentArrayStep();
  const graphStep = useCurrentGraphStep();
  const dpStep    = useCurrentDPStep();
  const treeStep  = useCurrentTreeStep();
  const playbackState = useVisualizerStore((s) => s.playbackState);
  if (playbackState === "idle")     return "Configure inputs and press Visualize to begin.";
  if (playbackState === "finished") return "✓ Algorithm complete!";
  return arrayStep?.description ?? graphStep?.description ?? dpStep?.description ?? treeStep?.description ?? "Running...";
}
