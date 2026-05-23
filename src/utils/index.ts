import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";
import {
  MIN_ARRAY_VALUE,
  MAX_ARRAY_VALUE,
  DEFAULT_ARRAY_SIZE,
} from "@/constants";

// ─── Tailwind Class Merge ─────────────────────────────────────────────────────

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// ─── Array Utilities ──────────────────────────────────────────────────────────

export function generateRandomArray(
  size: number = DEFAULT_ARRAY_SIZE,
  min: number = MIN_ARRAY_VALUE,
  max: number = MAX_ARRAY_VALUE
): number[] {
  return Array.from(
    { length: size },
    () => Math.floor(Math.random() * (max - min + 1)) + min
  );
}

export function generateNearlySortedArray(size: number = DEFAULT_ARRAY_SIZE): number[] {
  const arr = Array.from({ length: size }, (_, i) => (i + 1) * 7);
  // Swap a few elements
  const swaps = Math.max(1, Math.floor(size * 0.1));
  for (let i = 0; i < swaps; i++) {
    const a = Math.floor(Math.random() * size);
    const b = Math.floor(Math.random() * size);
    [arr[a], arr[b]] = [arr[b], arr[a]];
  }
  return arr;
}

export function generateReversedArray(size: number = DEFAULT_ARRAY_SIZE): number[] {
  return Array.from({ length: size }, (_, i) => (size - i) * 7);
}

export function parseArrayInput(input: string): number[] {
  return input
    .split(/[\s,]+/)
    .map((s) => parseInt(s.trim(), 10))
    .filter((n) => !isNaN(n) && n >= 0 && n <= 999);
}

// ─── Format Utilities ─────────────────────────────────────────────────────────

export function formatNumber(n: number): string {
  return n.toLocaleString();
}

export function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

export function formatStepDescription(step: number, total: number): string {
  return `Step ${step} of ${total}`;
}

// ─── String Utilities ─────────────────────────────────────────────────────────

export function slugify(str: string): string {
  return str
    .toLowerCase()
    .replace(/\s+/g, "-")
    .replace(/[^a-z0-9-]/g, "");
}

export function capitalize(str: string): string {
  return str.charAt(0).toUpperCase() + str.slice(1);
}

export function truncate(str: string, length: number): string {
  return str.length > length ? str.slice(0, length) + "…" : str;
}

// ─── Array Operations ─────────────────────────────────────────────────────────

export function swap<T>(arr: T[], i: number, j: number): T[] {
  const result = [...arr];
  [result[i], result[j]] = [result[j], result[i]];
  return result;
}

export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

// ─── Color Utilities ──────────────────────────────────────────────────────────

export function getBarColor(
  index: number,
  meta: {
    comparing?: number[];
    swapping?: number[];
    pivot?: number;
    sorted?: number[];
    found?: number;
  }
): string {
  const { comparing = [], swapping = [], pivot, sorted = [], found } = meta;

  if (found === index) return "hsl(142 71% 52%)";
  if (sorted.includes(index)) return "hsl(142 71% 45% / 0.8)";
  if (pivot === index) return "hsl(262 83% 70%)";
  if (swapping.includes(index)) return "hsl(346 87% 65%)";
  if (comparing.includes(index)) return "hsl(38 92% 58%)";
  return "hsl(220 14% 40%)";
}

// ─── Graph Utilities ──────────────────────────────────────────────────────────

export function edgeKey(u: string, v: string, directed = false): string {
  if (directed) return `${u}->${v}`;
  return [u, v].sort().join("--");
}

// ─── Delay Utility (for step-based animations) ────────────────────────────────

export function createStepQueue<T>(
  steps: T[],
  onStep: (step: T, index: number) => void,
  delay: number
): { start: () => void; stop: () => void } {
  let timeoutId: ReturnType<typeof setTimeout> | null = null;
  let stopped = false;

  function run(index: number) {
    if (stopped || index >= steps.length) return;
    onStep(steps[index], index);
    timeoutId = setTimeout(() => run(index + 1), delay);
  }

  return {
    start: () => {
      stopped = false;
      run(0);
    },
    stop: () => {
      stopped = true;
      if (timeoutId) clearTimeout(timeoutId);
    },
  };
}
