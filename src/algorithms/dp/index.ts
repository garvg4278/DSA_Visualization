import type { DPStep } from "@/types";

// ─── Fibonacci DP ─────────────────────────────────────────────────────────────

export function fibonacciSteps(n: number): DPStep[] {
  const steps: DPStep[] = [];
  const maxN = Math.min(n, 15);
  const fib = new Array(maxN + 1).fill(0);
  fib[0] = 0;
  if (maxN > 0) fib[1] = 1;

  // Single-row table representation
  const table = [fib.map(() => 0)];

  steps.push({
    type: "info",
    table: [fib.map(() => -1)],
    activeCell: [0, 0],
    highlightedCells: [],
    description: `Computing F(${maxN}) using bottom-up DP. F(0)=0, F(1)=1`,
  });

  for (let i = 0; i <= maxN; i++) {
    if (i <= 1) {
      table[0][i] = fib[i];
      steps.push({
        type: "dp-fill",
        table: [table[0].map((v, j) => (j <= i ? v : -1))],
        activeCell: [0, i],
        highlightedCells: [],
        description: `Base case: F(${i}) = ${fib[i]}`,
      });
    } else {
      fib[i] = fib[i - 1] + fib[i - 2];
      table[0][i] = fib[i];

      steps.push({
        type: "compare",
        table: [table[0].map((v, j) => (j < i ? v : -1))],
        activeCell: [0, i],
        highlightedCells: [[0, i - 1], [0, i - 2]],
        description: `F(${i}) = F(${i - 1}) + F(${i - 2}) = ${fib[i - 1]} + ${fib[i - 2]} = ${fib[i]}`,
      });

      steps.push({
        type: "dp-fill",
        table: [table[0].map((v, j) => (j <= i ? v : -1))],
        activeCell: [0, i],
        highlightedCells: [[0, i - 1], [0, i - 2]],
        description: `Stored F(${i}) = ${fib[i]}`,
      });
    }
  }

  steps.push({
    type: "dp-match",
    table: [table[0]],
    activeCell: [0, maxN],
    highlightedCells: [[0, maxN]],
    description: `Done! F(${maxN}) = ${fib[maxN]}`,
  });

  return steps;
}

// ─── LCS ──────────────────────────────────────────────────────────────────────

export function lcsSteps(s1: string, s2: string): DPStep[] {
  const steps: DPStep[] = [];
  const m = Math.min(s1.length, 8);
  const nLen = Math.min(s2.length, 8);
  const a = s1.slice(0, m);
  const b = s2.slice(0, nLen);

  const dp: number[][] = Array.from({ length: m + 1 }, () =>
    new Array(nLen + 1).fill(0)
  );

  steps.push({
    type: "info",
    table: dp.map((r) => [...r]),
    activeCell: [0, 0],
    highlightedCells: [],
    description: `LCS of "${a}" and "${b}". Building (${m + 1})×(${nLen + 1}) DP table.`,
    auxiliaryData: { s1: a, s2: b },
  });

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= nLen; j++) {
      const match = a[i - 1] === b[j - 1];

      steps.push({
        type: "compare",
        table: dp.map((r) => [...r]),
        activeCell: [i, j],
        highlightedCells: match
          ? [[i - 1, j - 1]]
          : [[i - 1, j], [i, j - 1]],
        description: match
          ? `a[${i - 1}]='${a[i - 1]}' == b[${j - 1}]='${b[j - 1]}' → dp[${i}][${j}] = dp[${i - 1}][${j - 1}] + 1`
          : `a[${i - 1}]='${a[i - 1]}' ≠ b[${j - 1}]='${b[j - 1]}' → dp[${i}][${j}] = max(dp[${i - 1}][${j}], dp[${i}][${j - 1}])`,
        auxiliaryData: { s1: a, s2: b, i, j },
      });

      if (match) {
        dp[i][j] = dp[i - 1][j - 1] + 1;
      } else {
        dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
      }

      steps.push({
        type: match ? "dp-match" : "dp-fill",
        table: dp.map((r) => [...r]),
        activeCell: [i, j],
        highlightedCells: [[i, j]],
        description: `dp[${i}][${j}] = ${dp[i][j]}`,
        auxiliaryData: { s1: a, s2: b },
      });
    }
  }

  steps.push({
    type: "dp-match",
    table: dp.map((r) => [...r]),
    activeCell: [m, nLen],
    highlightedCells: [[m, nLen]],
    description: `LCS length = ${dp[m][nLen]}. Trace back for the actual subsequence.`,
    auxiliaryData: { s1: a, s2: b, lcsLength: dp[m][nLen] },
  });

  return steps;
}

// ─── 0/1 Knapsack ─────────────────────────────────────────────────────────────

export interface KnapsackItem {
  weight: number;
  value: number;
  name: string;
}

export const DEFAULT_KNAPSACK_ITEMS: KnapsackItem[] = [
  { weight: 2, value: 6, name: "Gem" },
  { weight: 3, value: 4, name: "Book" },
  { weight: 4, value: 5, name: "Tool" },
  { weight: 5, value: 3, name: "Vase" },
];

export function knapsackSteps(
  items: KnapsackItem[],
  capacity: number
): DPStep[] {
  const steps: DPStep[] = [];
  const n = items.length;
  const dp: number[][] = Array.from({ length: n + 1 }, () =>
    new Array(capacity + 1).fill(0)
  );

  steps.push({
    type: "info",
    table: dp.map((r) => [...r]),
    activeCell: [0, 0],
    highlightedCells: [],
    description: `Knapsack: ${n} items, capacity=${capacity}. Items: ${items.map((i) => `${i.name}(w=${i.weight},v=${i.value})`).join(", ")}`,
    auxiliaryData: { items, capacity },
  });

  for (let i = 1; i <= n; i++) {
    const { weight, value, name } = items[i - 1];

    for (let w = 0; w <= capacity; w++) {
      if (weight > w) {
        dp[i][w] = dp[i - 1][w];
        steps.push({
          type: "dp-fill",
          table: dp.map((r) => [...r]),
          activeCell: [i, w],
          highlightedCells: [[i - 1, w]],
          description: `Item "${name}" (w=${weight}) > capacity ${w} → can't include. dp[${i}][${w}] = ${dp[i][w]}`,
          auxiliaryData: { items, capacity },
        });
      } else {
        const include = dp[i - 1][w - weight] + value;
        const exclude = dp[i - 1][w];

        steps.push({
          type: "compare",
          table: dp.map((r) => [...r]),
          activeCell: [i, w],
          highlightedCells: [[i - 1, w], [i - 1, w - weight]],
          description: `Include "${name}": ${exclude} vs ${include}. ${include > exclude ? "Include!" : "Exclude."}`,
          auxiliaryData: { items, capacity },
        });

        dp[i][w] = Math.max(include, exclude);
        steps.push({
          type: include > exclude ? "dp-match" : "dp-fill",
          table: dp.map((r) => [...r]),
          activeCell: [i, w],
          highlightedCells: [[i, w]],
          description: `dp[${i}][${w}] = ${dp[i][w]} (${include > exclude ? "included" : "excluded"} "${name}")`,
          auxiliaryData: { items, capacity },
        });
      }
    }
  }

  steps.push({
    type: "dp-match",
    table: dp.map((r) => [...r]),
    activeCell: [n, capacity],
    highlightedCells: [[n, capacity]],
    description: `Maximum value = ${dp[n][capacity]}. Trace back to find chosen items.`,
    auxiliaryData: { items, capacity, maxValue: dp[n][capacity] },
  });

  return steps;
}

// ─── Edit Distance ────────────────────────────────────────────────────────────

export function editDistanceSteps(s1: string, s2: string): DPStep[] {
  const steps: DPStep[] = [];
  const m = Math.min(s1.length, 7);
  const nLen = Math.min(s2.length, 7);
  const a = s1.slice(0, m);
  const b = s2.slice(0, nLen);

  const dp: number[][] = Array.from({ length: m + 1 }, (_, i) =>
    Array.from({ length: nLen + 1 }, (_, j) => (i === 0 ? j : j === 0 ? i : 0))
  );

  steps.push({
    type: "info",
    table: dp.map((r) => [...r]),
    activeCell: [0, 0],
    highlightedCells: [],
    description: `Edit Distance: "${a}" → "${b}". Base cases: first row/col filled with 0..n.`,
    auxiliaryData: { s1: a, s2: b },
  });

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= nLen; j++) {
      const match = a[i - 1] === b[j - 1];

      steps.push({
        type: "compare",
        table: dp.map((r) => [...r]),
        activeCell: [i, j],
        highlightedCells: [[i - 1, j - 1], [i - 1, j], [i, j - 1]],
        description: match
          ? `'${a[i - 1]}' == '${b[j - 1]}': no operation needed. Copy dp[${i - 1}][${j - 1}]`
          : `'${a[i - 1]}' ≠ '${b[j - 1]}': 1 + min(delete, insert, substitute)`,
        auxiliaryData: { s1: a, s2: b },
      });

      if (match) {
        dp[i][j] = dp[i - 1][j - 1];
      } else {
        dp[i][j] = 1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
      }

      steps.push({
        type: match ? "dp-match" : "dp-fill",
        table: dp.map((r) => [...r]),
        activeCell: [i, j],
        highlightedCells: [[i, j]],
        description: `dp[${i}][${j}] = ${dp[i][j]}${!match ? ` (${["delete", "insert", "substitute"][[dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]].indexOf(dp[i][j] - 1)]})` : ""}`,
        auxiliaryData: { s1: a, s2: b },
      });
    }
  }

  steps.push({
    type: "dp-match",
    table: dp.map((r) => [...r]),
    activeCell: [m, nLen],
    highlightedCells: [[m, nLen]],
    description: `Edit distance = ${dp[m][nLen]}. "${a}" → "${b}" requires ${dp[m][nLen]} operation(s).`,
    auxiliaryData: { s1: a, s2: b },
  });

  return steps;
}

// ─── Code Implementations ─────────────────────────────────────────────────────

export const dpImplementations = {
  fibonacci: {
    typescript: `function fibonacci(n: number): number {
  if (n <= 1) return n;
  const dp = [0, 1];
  for (let i = 2; i <= n; i++) {
    dp[i] = dp[i - 1] + dp[i - 2];
  }
  return dp[n];
}

// Space-optimized O(1):
function fibOptimized(n: number): number {
  if (n <= 1) return n;
  let [a, b] = [0, 1];
  for (let i = 2; i <= n; i++) [a, b] = [b, a + b];
  return b;
}`,
    python: `def fibonacci(n: int) -> int:
    if n <= 1: return n
    dp = [0, 1]
    for i in range(2, n + 1):
        dp.append(dp[-1] + dp[-2])
    return dp[n]`,
    java: `public static int fibonacci(int n) {
    if (n <= 1) return n;
    int[] dp = new int[n + 1];
    dp[0] = 0; dp[1] = 1;
    for (int i = 2; i <= n; i++) dp[i] = dp[i-1] + dp[i-2];
    return dp[n];
}`,
    cpp: `int fibonacci(int n) {
    if (n <= 1) return n;
    vector<int> dp(n + 1);
    dp[0] = 0; dp[1] = 1;
    for (int i = 2; i <= n; i++) dp[i] = dp[i-1] + dp[i-2];
    return dp[n];
}`,
  },
  lcs: {
    typescript: `function lcs(s1: string, s2: string): number {
  const m = s1.length, n = s2.length;
  const dp = Array.from({length: m+1}, () => new Array(n+1).fill(0));
  
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (s1[i-1] === s2[j-1]) dp[i][j] = dp[i-1][j-1] + 1;
      else dp[i][j] = Math.max(dp[i-1][j], dp[i][j-1]);
    }
  }
  return dp[m][n];
}`,
    python: `def lcs(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]`,
    java: `public static int lcs(String s1, String s2) {
    int m = s1.length(), n = s2.length();
    int[][] dp = new int[m + 1][n + 1];
    for (int i = 1; i <= m; i++)
        for (int j = 1; j <= n; j++)
            dp[i][j] = s1.charAt(i-1) == s2.charAt(j-1)
                ? dp[i-1][j-1] + 1
                : Math.max(dp[i-1][j], dp[i][j-1]);
    return dp[m][n];
}`,
    cpp: `int lcs(string s1, string s2) {
    int m = s1.size(), n = s2.size();
    vector<vector<int>> dp(m+1, vector<int>(n+1, 0));
    for (int i = 1; i <= m; i++)
        for (int j = 1; j <= n; j++)
            dp[i][j] = s1[i-1] == s2[j-1]
                ? dp[i-1][j-1] + 1
                : max(dp[i-1][j], dp[i][j-1]);
    return dp[m][n];
}`,
  },
};
