import type { ArrayStep } from "@/types";

export function selectionSortSteps(
  inputArray: number[],
  order: "asc" | "desc" = "asc"
): ArrayStep[] {
  const steps: ArrayStep[] = [];
  const arr = [...inputArray];
  const n = arr.length;
  const sortedIndices: number[] = [];

  steps.push({
    type: "info",
    array: [...arr],
    indices: [],
    description: `Starting Selection Sort on [${arr.join(", ")}]`,
  });

  for (let i = 0; i < n - 1; i++) {
    let extremeIdx = i;

    steps.push({
      type: "highlight",
      array: [...arr],
      indices: [i],
      description: `Pass ${i + 1}: Finding the ${order === "asc" ? "minimum" : "maximum"} in positions ${i}–${n - 1}`,
      auxiliaryData: { sortedIndices: [...sortedIndices] },
    });

    for (let j = i + 1; j < n; j++) {
      steps.push({
        type: "compare",
        array: [...arr],
        indices: [extremeIdx, j],
        description: `Comparing current ${order === "asc" ? "min" : "max"} ${arr[extremeIdx]} with ${arr[j]}`,
        auxiliaryData: { currentExtreme: extremeIdx, sortedIndices: [...sortedIndices] },
      });

      const newExtreme =
        order === "asc"
          ? arr[j] < arr[extremeIdx]
          : arr[j] > arr[extremeIdx];

      if (newExtreme) {
        extremeIdx = j;
        steps.push({
          type: "highlight",
          array: [...arr],
          indices: [extremeIdx],
          description: `New ${order === "asc" ? "minimum" : "maximum"} found: ${arr[extremeIdx]} at index ${extremeIdx}`,
          auxiliaryData: { currentExtreme: extremeIdx, sortedIndices: [...sortedIndices] },
        });
      }
    }

    if (extremeIdx !== i) {
      [arr[i], arr[extremeIdx]] = [arr[extremeIdx], arr[i]];
      steps.push({
        type: "swap",
        array: [...arr],
        indices: [i, extremeIdx],
        description: `Swapping ${arr[extremeIdx]} (was at ${extremeIdx}) with ${arr[i]} (at ${i})`,
        auxiliaryData: { sortedIndices: [...sortedIndices] },
      });
    } else {
      steps.push({
        type: "info",
        array: [...arr],
        indices: [i],
        description: `${arr[i]} is already in position ${i}, no swap needed`,
        auxiliaryData: { sortedIndices: [...sortedIndices] },
      });
    }

    sortedIndices.push(i);
    steps.push({
      type: "sorted",
      array: [...arr],
      indices: [...sortedIndices],
      description: `${arr[i]} placed at its final position ${i}`,
    });
  }

  sortedIndices.push(n - 1);
  steps.push({
    type: "sorted",
    array: [...arr],
    indices: Array.from({ length: n }, (_, k) => k),
    description: `Selection Sort complete! Sorted: [${arr.join(", ")}]`,
  });

  return steps;
}

export const selectionSortMeta = {
  id: "selection-sort",
  implementations: {
    typescript: `function selectionSort(arr: number[]): number[] {
  const n = arr.length;
  
  for (let i = 0; i < n - 1; i++) {
    let minIdx = i;
    
    for (let j = i + 1; j < n; j++) {
      if (arr[j] < arr[minIdx]) {
        minIdx = j;
      }
    }
    
    if (minIdx !== i) {
      [arr[i], arr[minIdx]] = [arr[minIdx], arr[i]];
    }
  }
  
  return arr;
}`,
    python: `def selection_sort(arr: list[int]) -> list[int]:
    n = len(arr)
    
    for i in range(n - 1):
        min_idx = i
        
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    
    return arr`,
    java: `public static int[] selectionSort(int[] arr) {
    int n = arr.length;
    
    for (int i = 0; i < n - 1; i++) {
        int minIdx = i;
        
        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[minIdx]) {
                minIdx = j;
            }
        }
        
        int temp = arr[minIdx];
        arr[minIdx] = arr[i];
        arr[i] = temp;
    }
    
    return arr;
}`,
    cpp: `vector<int> selectionSort(vector<int> arr) {
    int n = arr.size();
    
    for (int i = 0; i < n - 1; i++) {
        int minIdx = i;
        
        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[minIdx]) {
                minIdx = j;
            }
        }
        
        swap(arr[i], arr[minIdx]);
    }
    
    return arr;
}`,
  },
};
