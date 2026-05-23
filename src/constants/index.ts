import type { AlgorithmMeta, AlgorithmCategory } from "@/types";

// ─── App Config ───────────────────────────────────────────────────────────────

export const APP_CONFIG = {
  name: "AlgoVista",
  tagline: "Visualize. Understand. Master.",
  description:
    "An industry-grade DSA visualization platform for serious learners.",
  version: "2.0.0",
  github: "https://github.com/yourusername/algovista",
} as const;

// ─── Animation Config ─────────────────────────────────────────────────────────

export const ANIMATION_SPEEDS = {
  1: 2000,
  2: 1500,
  3: 1200,
  4: 900,
  5: 700,
  6: 500,
  7: 350,
  8: 200,
  9: 100,
  10: 50,
} as const;

export const DEFAULT_SPEED = 5;
export const DEFAULT_ARRAY_SIZE = 12;
export const MAX_ARRAY_SIZE = 24;
export const MIN_ARRAY_VALUE = 5;
export const MAX_ARRAY_VALUE = 100;

// ─── Color Palette ────────────────────────────────────────────────────────────

export const VIZ_COLORS = {
  default: "hsl(220, 14%, 30%)",
  compare: "hsl(38, 92%, 58%)",
  swap: "hsl(346, 87%, 65%)",
  pivot: "hsl(262, 83%, 70%)",
  sorted: "hsl(142, 71%, 52%)",
  found: "hsl(142, 71%, 52%)",
  notFound: "hsl(346, 87%, 65%)",
  highlight: "hsl(199, 89%, 60%)",
  active: "hsl(262, 83%, 70%)",
  visited: "hsl(220, 14%, 50%)",
  path: "hsl(142, 71%, 52%)",
} as const;

// ─── Category Config ──────────────────────────────────────────────────────────

export const CATEGORY_CONFIG: Record<
  AlgorithmCategory,
  { label: string; color: string; icon: string; description: string }
> = {
  sorting: {
    label: "Sorting",
    color: "hsl(262, 83%, 70%)",
    icon: "ArrowUpDown",
    description: "Ordering elements efficiently",
  },
  searching: {
    label: "Searching",
    color: "hsl(199, 89%, 60%)",
    icon: "Search",
    description: "Finding elements in collections",
  },
  graph: {
    label: "Graph",
    color: "hsl(142, 71%, 52%)",
    icon: "Network",
    description: "Traversal and shortest paths",
  },
  trees: {
    label: "Trees",
    color: "hsl(38, 92%, 58%)",
    icon: "GitBranch",
    description: "Hierarchical data structures",
  },
  dp: {
    label: "Dynamic Programming",
    color: "hsl(346, 87%, 65%)",
    icon: "TableProperties",
    description: "Optimal substructure problems",
  },
  recursion: {
    label: "Recursion",
    color: "hsl(293, 70%, 65%)",
    icon: "RefreshCw",
    description: "Self-referential problem solving",
  },
  backtracking: {
    label: "Backtracking",
    color: "hsl(25, 95%, 65%)",
    icon: "Undo2",
    description: "Exhaustive constraint search",
  },
};

// ─── Algorithm Registry ───────────────────────────────────────────────────────

export const ALGORITHM_REGISTRY: AlgorithmMeta[] = [
  // Sorting
  {
    id: "bubble-sort",
    name: "Bubble Sort",
    category: "sorting",
    complexity: {
      time: { best: "O(n)", average: "O(n²)", worst: "O(n²)" },
      space: "O(1)",
    },
    description:
      "Repeatedly compares adjacent elements and swaps them if out of order. The largest element 'bubbles up' to the correct position each pass.",
    keyInsight:
      "After each pass, the largest unsorted element is guaranteed to be in its final position.",
    useCases: ["Educational purposes", "Small nearly-sorted datasets"],
    stable: true,
    inPlace: true,
    tags: ["comparison", "simple", "stable"],
  },
  {
    id: "selection-sort",
    name: "Selection Sort",
    category: "sorting",
    complexity: {
      time: { best: "O(n²)", average: "O(n²)", worst: "O(n²)" },
      space: "O(1)",
    },
    description:
      "Finds the minimum element from the unsorted portion and places it at the beginning. Makes exactly n-1 swaps.",
    keyInsight:
      "Minimizes the number of swaps, making it useful when write operations are expensive.",
    useCases: ["Memory-constrained environments", "When swaps are costly"],
    stable: false,
    inPlace: true,
    tags: ["comparison", "simple", "in-place"],
  },
  {
    id: "insertion-sort",
    name: "Insertion Sort",
    category: "sorting",
    complexity: {
      time: { best: "O(n)", average: "O(n²)", worst: "O(n²)" },
      space: "O(1)",
    },
    description:
      "Builds a sorted array one element at a time by inserting each element into its correct position in the already-sorted portion.",
    keyInsight:
      "Adaptive: performs well on nearly-sorted data. Online algorithm that works on streaming input.",
    useCases: [
      "Nearly sorted data",
      "Online sorting",
      "Small arrays",
      "Hybrid algorithm base",
    ],
    stable: true,
    inPlace: true,
    tags: ["comparison", "adaptive", "stable", "online"],
  },
  {
    id: "merge-sort",
    name: "Merge Sort",
    category: "sorting",
    complexity: {
      time: { best: "O(n log n)", average: "O(n log n)", worst: "O(n log n)" },
      space: "O(n)",
    },
    description:
      "Divide-and-conquer: recursively splits the array in half, sorts each half, then merges. Guarantees O(n log n) in all cases.",
    keyInsight:
      "Consistent performance regardless of input. Preferred for linked lists and external sorting.",
    useCases: [
      "Linked lists",
      "External sorting",
      "Stable sort requirement",
      "Parallel sorting",
    ],
    stable: true,
    inPlace: false,
    tags: ["divide-conquer", "stable", "guaranteed"],
  },
  {
    id: "quick-sort",
    name: "Quick Sort",
    category: "sorting",
    complexity: {
      time: { best: "O(n log n)", average: "O(n log n)", worst: "O(n²)" },
      space: "O(log n)",
    },
    description:
      "Selects a pivot, partitions the array around it, then recursively sorts both partitions. Cache-friendly and fast in practice.",
    keyInsight:
      "Despite O(n²) worst case, excellent average performance due to cache efficiency. Most real-world implementations use randomized pivot.",
    useCases: [
      "General-purpose sorting",
      "Arrays",
      "When average performance matters",
    ],
    stable: false,
    inPlace: true,
    tags: ["divide-conquer", "cache-friendly", "practical"],
  },
  {
    id: "heap-sort",
    name: "Heap Sort",
    category: "sorting",
    complexity: {
      time: { best: "O(n log n)", average: "O(n log n)", worst: "O(n log n)" },
      space: "O(1)",
    },
    description:
      "Builds a max-heap, then repeatedly extracts the maximum element. Combines the best properties of selection sort and merge sort.",
    keyInsight:
      "O(1) space and O(n log n) time in all cases, but poor cache performance makes it slower than Quick Sort in practice.",
    useCases: [
      "Memory-constrained environments",
      "Guaranteed worst-case performance",
    ],
    stable: false,
    inPlace: true,
    tags: ["heap", "guaranteed", "in-place"],
  },
  {
    id: "radix-sort",
    name: "Radix Sort",
    category: "sorting",
    complexity: {
      time: { best: "O(nk)", average: "O(nk)", worst: "O(nk)" },
      space: "O(n+k)",
    },
    description:
      "Non-comparison sort that processes digits from least to most significant. Groups numbers into buckets for each digit position.",
    keyInsight:
      "Breaks the comparison-based O(n log n) lower bound by exploiting structure in the data.",
    useCases: [
      "Integer sorting",
      "String sorting",
      "Fixed-length keys",
      "When k is small",
    ],
    stable: true,
    inPlace: false,
    tags: ["non-comparison", "linear", "stable"],
  },
  {
    id: "counting-sort",
    name: "Counting Sort",
    category: "sorting",
    complexity: {
      time: { best: "O(n+k)", average: "O(n+k)", worst: "O(n+k)" },
      space: "O(k)",
    },
    description:
      "Counts occurrences of each element, builds cumulative counts, then reconstructs the sorted array. Linear time for bounded integers.",
    keyInsight:
      "Achieves linear time by exploiting the bounded range of input values, not comparisons.",
    useCases: [
      "Small integer ranges",
      "Counting frequencies",
      "Base for Radix Sort",
    ],
    stable: true,
    inPlace: false,
    tags: ["non-comparison", "linear", "stable"],
  },
  // Searching
  {
    id: "linear-search",
    name: "Linear Search",
    category: "searching",
    complexity: {
      time: { best: "O(1)", average: "O(n)", worst: "O(n)" },
      space: "O(1)",
    },
    description:
      "Sequentially checks each element until the target is found or the end is reached. No preprocessing required.",
    keyInsight: "Simple and works on any collection, sorted or unsorted.",
    useCases: ["Small unsorted arrays", "Linked lists", "Simple lookups"],
    tags: ["sequential", "simple"],
  },
  {
    id: "binary-search",
    name: "Binary Search",
    category: "searching",
    complexity: {
      time: { best: "O(1)", average: "O(log n)", worst: "O(log n)" },
      space: "O(1)",
    },
    description:
      "Efficiently searches a sorted array by repeatedly halving the search interval. Eliminates half the remaining elements each step.",
    keyInsight:
      "Requires sorted input but achieves logarithmic time — searching 1 billion elements takes at most 30 comparisons.",
    useCases: [
      "Sorted arrays",
      "Database indices",
      "Range queries",
      "Monotonic functions",
    ],
    tags: ["divide-conquer", "sorted", "efficient"],
  },
  {
    id: "jump-search",
    name: "Jump Search",
    category: "searching",
    complexity: {
      time: { best: "O(1)", average: "O(√n)", worst: "O(√n)" },
      space: "O(1)",
    },
    description:
      "Jumps ahead by √n steps to find the block containing the target, then performs linear search within that block.",
    keyInsight:
      "Optimal jump size √n balances the number of jumps with linear search length.",
    useCases: ["Sorted arrays where binary search is expensive to implement"],
    tags: ["sorted", "block-based"],
  },
  // Graph
  {
    id: "bfs",
    name: "Breadth-First Search",
    category: "graph",
    complexity: {
      time: { best: "O(V+E)", average: "O(V+E)", worst: "O(V+E)" },
      space: "O(V)",
    },
    description:
      "Explores all neighbors at current depth before moving deeper. Uses a queue to process nodes level by level.",
    keyInsight:
      "Guarantees shortest path in unweighted graphs. Explores in concentric rings from source.",
    useCases: [
      "Shortest path (unweighted)",
      "Level-order traversal",
      "Social network analysis",
      "Web crawling",
    ],
    tags: ["traversal", "shortest-path", "queue"],
  },
  {
    id: "dfs",
    name: "Depth-First Search",
    category: "graph",
    complexity: {
      time: { best: "O(V+E)", average: "O(V+E)", worst: "O(V+E)" },
      space: "O(V)",
    },
    description:
      "Explores as far as possible along each branch before backtracking. Uses a stack (or recursion) to track the path.",
    keyInsight:
      "Naturally finds connected components, detects cycles, and enables topological sorting.",
    useCases: [
      "Cycle detection",
      "Topological sort",
      "Maze solving",
      "Connected components",
    ],
    tags: ["traversal", "backtracking", "stack"],
  },
  {
    id: "dijkstra",
    name: "Dijkstra's Algorithm",
    category: "graph",
    complexity: {
      time: { best: "O(V+E)", average: "O(E log V)", worst: "O(E log V)" },
      space: "O(V)",
    },
    description:
      "Finds shortest paths from a source node to all other nodes in a weighted graph with non-negative edge weights.",
    keyInsight:
      "Greedy: always relaxes the unvisited node with the smallest known distance. Correct because edge weights are non-negative.",
    useCases: ["GPS navigation", "Network routing", "Maps", "Flight paths"],
    tags: ["shortest-path", "greedy", "weighted"],
  },
  {
    id: "kruskal",
    name: "Kruskal's MST",
    category: "graph",
    complexity: {
      time: { best: "O(E log E)", average: "O(E log E)", worst: "O(E log E)" },
      space: "O(V)",
    },
    description:
      "Builds a Minimum Spanning Tree by sorting all edges and greedily adding them, skipping those that form cycles. Uses Union-Find.",
    keyInsight:
      "Cut property: the minimum-weight edge crossing any cut of the graph belongs to some MST.",
    useCases: [
      "Network design",
      "Cluster analysis",
      "Approximation algorithms",
    ],
    tags: ["mst", "greedy", "union-find"],
  },
  {
    id: "topological-sort",
    name: "Topological Sort",
    category: "graph",
    complexity: {
      time: { best: "O(V+E)", average: "O(V+E)", worst: "O(V+E)" },
      space: "O(V)",
    },
    description:
      "Linear ordering of vertices in a DAG such that for every directed edge u→v, u comes before v.",
    keyInsight:
      "Only possible for Directed Acyclic Graphs. DFS-based: push to result after all descendants visited.",
    useCases: [
      "Task scheduling",
      "Build systems",
      "Course prerequisites",
      "Package managers",
    ],
    tags: ["dag", "ordering", "dfs"],
  },
  // DP
  {
    id: "fibonacci",
    name: "Fibonacci (DP)",
    category: "dp",
    complexity: {
      time: { best: "O(n)", average: "O(n)", worst: "O(n)" },
      space: "O(n)",
    },
    description:
      "Computes Fibonacci numbers bottom-up, building a table from base cases. Eliminates exponential recursion.",
    keyInsight:
      "Transforms O(2ⁿ) naive recursion to O(n) by storing subproblem solutions — the essence of DP.",
    useCases: ["Classic DP introduction", "Understanding memoization"],
    tags: ["dp", "memoization", "classic"],
  },
  {
    id: "lcs",
    name: "Longest Common Subsequence",
    category: "dp",
    complexity: {
      time: { best: "O(nm)", average: "O(nm)", worst: "O(nm)" },
      space: "O(nm)",
    },
    description:
      "Finds the longest subsequence common to two sequences using a 2D DP table.",
    keyInsight:
      "Optimal substructure: if last chars match, extend LCS of prefixes; otherwise take max of two sub-problems.",
    useCases: ["Diff tools", "DNA sequence analysis", "Version control", "NLP"],
    tags: ["dp", "strings", "subsequence"],
  },
  {
    id: "knapsack",
    name: "0/1 Knapsack",
    category: "dp",
    complexity: {
      time: { best: "O(nW)", average: "O(nW)", worst: "O(nW)" },
      space: "O(nW)",
    },
    description:
      "Maximize value of items in a knapsack without exceeding weight capacity. Each item is taken or left.",
    keyInsight:
      "Pseudo-polynomial: exponential items but tractable when capacity W is bounded.",
    useCases: [
      "Resource allocation",
      "Portfolio optimization",
      "Budget planning",
    ],
    tags: ["dp", "optimization", "combinatorial"],
  },
  {
    id: "edit-distance",
    name: "Edit Distance",
    category: "dp",
    complexity: {
      time: { best: "O(nm)", average: "O(nm)", worst: "O(nm)" },
      space: "O(nm)",
    },
    description:
      "Minimum edit operations (insert, delete, substitute) to transform one string into another.",
    keyInsight:
      "Every alignment of the two strings corresponds to a sequence of edits; DP finds the minimum.",
    useCases: ["Spell checking", "DNA alignment", "Plagiarism detection"],
    tags: ["dp", "strings", "edit"],
  },
  // Trees
  {
    id: "bst",
    name: "Binary Search Tree",
    category: "trees",
    complexity: {
      time: { best: "O(log n)", average: "O(log n)", worst: "O(n)" },
      space: "O(n)",
    },
    description:
      "A binary tree where every left child < parent < every right child. Supports efficient search, insert, and delete.",
    keyInsight:
      "Degenerates to O(n) with sorted input — the motivation for balanced BSTs like AVL and Red-Black trees.",
    useCases: ["Ordered sets", "Maps", "Databases", "Priority queues"],
    tags: ["tree", "search", "ordered"],
  },
  {
    id: "tree-traversal",
    name: "Tree Traversals",
    category: "trees",
    complexity: {
      time: { best: "O(n)", average: "O(n)", worst: "O(n)" },
      space: "O(h)",
    },
    description:
      "Three fundamental traversals: Inorder (sorted output), Preorder (copy/serialize), Postorder (delete/evaluate).",
    keyInsight:
      "Inorder of a BST produces sorted output — a key property connecting trees and sorting.",
    useCases: [
      "Expression trees",
      "File systems",
      "Serialization",
      "Syntax trees",
    ],
    tags: ["tree", "traversal", "fundamental"],
  },
];

export const ALGORITHM_MAP = new Map(
  ALGORITHM_REGISTRY.map((algo) => [algo.id, algo])
);

export const ALGORITHMS_BY_CATEGORY = ALGORITHM_REGISTRY.reduce(
  (acc, algo) => {
    if (!acc[algo.category]) acc[algo.category] = [];
    acc[algo.category].push(algo);
    return acc;
  },
  {} as Record<AlgorithmCategory, AlgorithmMeta[]>
);
