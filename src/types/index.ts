// ─── Algorithm Domain Types ──────────────────────────────────────────────────

export type AlgorithmCategory =
  | "sorting"
  | "searching"
  | "graph"
  | "trees"
  | "dp"
  | "recursion"
  | "backtracking";

export type ComplexityClass =
  | "O(1)"
  | "O(log n)"
  | "O(√n)"
  | "O(n)"
  | "O(n log n)"
  | "O(n²)"
  | "O(n³)"
  | "O(2ⁿ)"
  | "O(n!)"
  | "O(V+E)"
  | "O(E log V)"
  | "O(nm)"
  | "O(nW)";

export interface Complexity {
  time: {
    best: ComplexityClass;
    average: ComplexityClass;
    worst: ComplexityClass;
  };
  space: ComplexityClass;
}

export interface AlgorithmMeta {
  id: string;
  name: string;
  category: AlgorithmCategory;
  complexity: Complexity;
  description: string;
  keyInsight: string;
  useCases: string[];
  stable?: boolean;
  inPlace?: boolean;
  tags: string[];
}

// ─── Visualization Step Types ─────────────────────────────────────────────────

export type StepType =
  | "compare"
  | "swap"
  | "pivot"
  | "sorted"
  | "highlight"
  | "insert"
  | "merge"
  | "partition"
  | "visit"
  | "enqueue"
  | "dequeue"
  | "relax"
  | "found"
  | "not-found"
  | "dp-fill"
  | "dp-match"
  | "overwrite"
  | "info";

export interface ArrayStep {
  type: StepType;
  array: number[];
  indices: number[];
  description: string;
  auxiliaryData?: Record<string, unknown>;
}

export interface GraphStep {
  type: StepType;
  visitedNodes: string[];
  activeNodes: string[];
  activeEdges: string[][];
  highlightedEdges: string[][];
  distances?: Record<string, number>;
  description: string;
  auxiliaryData?: Record<string, unknown>;
}

export interface DPStep {
  type: StepType;
  table: number[][];
  activeCell: [number, number];
  highlightedCells: [number, number][];
  description: string;
  auxiliaryData?: Record<string, unknown>;
}

export interface TreeStep {
  type: StepType;
  highlightedNodes: string[];
  activeNode: string | null;
  description: string;
  auxiliaryData?: Record<string, unknown>;
}

export type VisualizationStep = ArrayStep | GraphStep | DPStep | TreeStep;

// ─── Control State ────────────────────────────────────────────────────────────

export type PlaybackState = "idle" | "playing" | "paused" | "finished";

export interface VisualizerControls {
  playbackState: PlaybackState;
  currentStep: number;
  totalSteps: number;
  speed: number; // 1-10
  isLooping: boolean;
}

// ─── Graph Data Types ─────────────────────────────────────────────────────────

export interface GraphNode {
  id: string;
  label: string;
  x: number;
  y: number;
  data?: Record<string, unknown>;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  weight?: number;
  directed?: boolean;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  directed: boolean;
  weighted: boolean;
}

// ─── Tree Data Types ──────────────────────────────────────────────────────────

export interface TreeNode {
  id: string;
  value: number;
  left: TreeNode | null;
  right: TreeNode | null;
  height?: number; // for AVL
  parent?: string | null;
}

// ─── Code Language Support ────────────────────────────────────────────────────

export type CodeLanguage = "typescript" | "python" | "java" | "cpp";

export interface CodeImplementation {
  language: CodeLanguage;
  code: string;
}

export interface AlgorithmCode {
  implementations: CodeImplementation[];
}

// ─── Comparison Mode ──────────────────────────────────────────────────────────

export interface ComparisonSession {
  leftAlgorithm: string;
  rightAlgorithm: string;
  inputArray: number[];
  leftSteps: ArrayStep[];
  rightSteps: ArrayStep[];
  leftOperations: number;
  rightOperations: number;
}

// ─── UI Types ─────────────────────────────────────────────────────────────────

export interface NavigationItem {
  id: string;
  label: string;
  href: string;
  icon?: string;
  badge?: string;
  children?: NavigationItem[];
}

export interface SearchResult {
  algorithm: AlgorithmMeta;
  score: number;
}
