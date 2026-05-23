import { create } from "zustand";
import { subscribeWithSelector } from "zustand/middleware";
import type {
  PlaybackState,
  ArrayStep,
  GraphStep,
  DPStep,
  TreeStep,
  CodeLanguage,
} from "@/types";
import { ANIMATION_SPEEDS, DEFAULT_SPEED } from "@/constants";

// ─── Visualizer Store ─────────────────────────────────────────────────────────

interface VisualizerState {
  // Current algorithm
  currentAlgorithmId: string | null;

  // Playback
  playbackState: PlaybackState;
  currentStep: number;
  speed: number;

  // Steps (one type active at a time)
  arraySteps: ArrayStep[];
  graphSteps: GraphStep[];
  dpSteps: DPStep[];
  treeSteps: TreeStep[];
  activeStepType: "array" | "graph" | "dp" | "tree" | null;

  // Input
  inputArray: number[];
  inputString1: string;
  inputString2: string;
  searchTarget: number;
  graphStartNode: string;
  sortOrder: "asc" | "desc";
  nValue: number;

  // UI
  selectedLanguage: CodeLanguage;
  showCode: boolean;
  showComplexity: boolean;
  showDescription: boolean;

  // Comparison
  comparisonMode: boolean;
  comparisonAlgoId: string | null;

  // Internal ticker
  _tickerId: ReturnType<typeof setInterval> | null;
}

interface VisualizerActions {
  // Algorithm selection
  setAlgorithm: (id: string) => void;

  // Step management
  setArraySteps: (steps: ArrayStep[]) => void;
  setGraphSteps: (steps: GraphStep[]) => void;
  setDPSteps: (steps: DPStep[]) => void;
  setTreeSteps: (steps: TreeStep[]) => void;

  // Playback controls
  play: () => void;
  pause: () => void;
  reset: () => void;
  stepForward: () => void;
  stepBackward: () => void;
  jumpToStep: (step: number) => void;
  setSpeed: (speed: number) => void;

  // Input controls
  setInputArray: (arr: number[]) => void;
  setInputString1: (s: string) => void;
  setInputString2: (s: string) => void;
  setSearchTarget: (t: number) => void;
  setGraphStartNode: (n: string) => void;
  setSortOrder: (order: "asc" | "desc") => void;
  setNValue: (n: number) => void;

  // UI
  setSelectedLanguage: (lang: CodeLanguage) => void;
  toggleCode: () => void;
  toggleComplexity: () => void;
  toggleDescription: () => void;

  // Internal
  _tick: () => void;
  _stopTicker: () => void;
  _startTicker: () => void;
}

type VisualizerStore = VisualizerState & VisualizerActions;

export const useVisualizerStore = create<VisualizerStore>()(
  subscribeWithSelector((set, get) => ({
    // ── Initial State ──────────────────────────────────────────────────────
    currentAlgorithmId: null,
    playbackState: "idle",
    currentStep: 0,
    speed: DEFAULT_SPEED,
    arraySteps: [],
    graphSteps: [],
    dpSteps: [],
    treeSteps: [],
    activeStepType: null,
    inputArray: [64, 34, 25, 12, 22, 11, 90, 45],
    inputString1: "ABCBDAB",
    inputString2: "BDCABA",
    searchTarget: 25,
    graphStartNode: "A",
    sortOrder: "asc",
    nValue: 10,
    selectedLanguage: "typescript",
    showCode: true,
    showComplexity: true,
    showDescription: true,
    comparisonMode: false,
    comparisonAlgoId: null,
    _tickerId: null,

    // ── Algorithm Selection ────────────────────────────────────────────────
    setAlgorithm: (id) => {
      get()._stopTicker();
      set({
        currentAlgorithmId: id,
        playbackState: "idle",
        currentStep: 0,
        arraySteps: [],
        graphSteps: [],
        dpSteps: [],
        treeSteps: [],
        activeStepType: null,
      });
    },

    // ── Step Management ────────────────────────────────────────────────────
    setArraySteps: (steps) =>
      set({
        arraySteps: steps,
        activeStepType: "array",
        currentStep: 0,
        playbackState: "idle",
      }),

    setGraphSteps: (steps) =>
      set({
        graphSteps: steps,
        activeStepType: "graph",
        currentStep: 0,
        playbackState: "idle",
      }),

    setDPSteps: (steps) =>
      set({
        dpSteps: steps,
        activeStepType: "dp",
        currentStep: 0,
        playbackState: "idle",
      }),

    setTreeSteps: (steps) =>
      set({
        treeSteps: steps,
        activeStepType: "tree",
        currentStep: 0,
        playbackState: "idle",
      }),

    // ── Playback Controls ──────────────────────────────────────────────────
    play: () => {
      const { playbackState, currentStep, activeStepType } = get();
      const totalSteps = getTotalSteps(get());
      if (totalSteps === 0) return;
      if (currentStep >= totalSteps - 1) {
        set({ currentStep: 0 });
      }
      if (playbackState !== "playing") {
        set({ playbackState: "playing" });
        get()._startTicker();
      }
    },

    pause: () => {
      get()._stopTicker();
      set({ playbackState: "paused" });
    },

    reset: () => {
      get()._stopTicker();
      set({ currentStep: 0, playbackState: "idle" });
    },

    stepForward: () => {
      const { currentStep } = get();
      const totalSteps = getTotalSteps(get());
      if (currentStep < totalSteps - 1) {
        set({ currentStep: currentStep + 1, playbackState: "paused" });
      }
    },

    stepBackward: () => {
      const { currentStep } = get();
      if (currentStep > 0) {
        set({ currentStep: currentStep - 1, playbackState: "paused" });
      }
    },

    jumpToStep: (step) => {
      const totalSteps = getTotalSteps(get());
      const clamped = Math.max(0, Math.min(step, totalSteps - 1));
      set({ currentStep: clamped, playbackState: "paused" });
    },

    setSpeed: (speed) => {
      set({ speed });
      const { playbackState } = get();
      if (playbackState === "playing") {
        get()._stopTicker();
        get()._startTicker();
      }
    },

    // ── Input Controls ─────────────────────────────────────────────────────
    setInputArray: (arr) => set({ inputArray: arr }),
    setInputString1: (s) => set({ inputString1: s }),
    setInputString2: (s) => set({ inputString2: s }),
    setSearchTarget: (t) => set({ searchTarget: t }),
    setGraphStartNode: (n) => set({ graphStartNode: n }),
    setSortOrder: (order) => set({ sortOrder: order }),
    setNValue: (n) => set({ nValue: n }),

    // ── UI ─────────────────────────────────────────────────────────────────
    setSelectedLanguage: (lang) => set({ selectedLanguage: lang }),
    toggleCode: () => set((s) => ({ showCode: !s.showCode })),
    toggleComplexity: () => set((s) => ({ showComplexity: !s.showComplexity })),
    toggleDescription: () =>
      set((s) => ({ showDescription: !s.showDescription })),

    // ── Internal Ticker ────────────────────────────────────────────────────
    _tick: () => {
      const { currentStep } = get();
      const totalSteps = getTotalSteps(get());
      if (currentStep >= totalSteps - 1) {
        get()._stopTicker();
        set({ playbackState: "finished" });
        return;
      }
      set({ currentStep: currentStep + 1 });
    },

    _startTicker: () => {
      const { speed } = get();
      const delay = ANIMATION_SPEEDS[speed as keyof typeof ANIMATION_SPEEDS] ?? 700;
      const id = setInterval(() => get()._tick(), delay);
      set({ _tickerId: id });
    },

    _stopTicker: () => {
      const { _tickerId } = get();
      if (_tickerId !== null) {
        clearInterval(_tickerId);
        set({ _tickerId: null });
      }
    },
  }))
);

// ─── Helpers ──────────────────────────────────────────────────────────────────

function getTotalSteps(state: VisualizerState): number {
  switch (state.activeStepType) {
    case "array": return state.arraySteps.length;
    case "graph": return state.graphSteps.length;
    case "dp": return state.dpSteps.length;
    case "tree": return state.treeSteps.length;
    default: return 0;
  }
}

// ─── Derived Selectors ────────────────────────────────────────────────────────

export const selectCurrentArrayStep = (state: VisualizerStore): ArrayStep | null =>
  state.activeStepType === "array" ? state.arraySteps[state.currentStep] ?? null : null;

export const selectCurrentGraphStep = (state: VisualizerStore): GraphStep | null =>
  state.activeStepType === "graph" ? state.graphSteps[state.currentStep] ?? null : null;

export const selectCurrentDPStep = (state: VisualizerStore): DPStep | null =>
  state.activeStepType === "dp" ? state.dpSteps[state.currentStep] ?? null : null;

export const selectCurrentTreeStep = (state: VisualizerStore): TreeStep | null =>
  state.activeStepType === "tree" ? state.treeSteps[state.currentStep] ?? null : null;

export const selectTotalSteps = (state: VisualizerStore): number => {
  switch (state.activeStepType) {
    case "array": return state.arraySteps.length;
    case "graph": return state.graphSteps.length;
    case "dp": return state.dpSteps.length;
    case "tree": return state.treeSteps.length;
    default: return 0;
  }
};

export const selectProgress = (state: VisualizerStore): number => {
  const total = selectTotalSteps(state);
  return total === 0 ? 0 : (state.currentStep / (total - 1)) * 100;
};
