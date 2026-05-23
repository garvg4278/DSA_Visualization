import type { ArrayStep } from "@/types";

export function bubbleSortSteps(
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
    description: `Starting Bubble Sort on [${arr.join(", ")}]`,
  });

  for (let i = 0; i < n - 1; i++) {
    let swapped = false;

    for (let j = 0; j < n - i - 1; j++) {
      // Compare step
      steps.push({
        type: "compare",
        array: [...arr],
        indices: [j, j + 1],
        description: `Pass ${i + 1}: Comparing ${arr[j]} and ${arr[j + 1]}`,
        auxiliaryData: { pass: i + 1, sortedFrom: n - i },
      });

      const shouldSwap =
        order === "asc" ? arr[j] > arr[j + 1] : arr[j] < arr[j + 1];

      if (shouldSwap) {
        [arr[j], arr[j + 1]] = [arr[j + 1], arr[j]];
        swapped = true;

        steps.push({
          type: "swap",
          array: [...arr],
          indices: [j, j + 1],
          description: `Swapping ${arr[j + 1]} and ${arr[j]} → [${arr.join(", ")}]`,
          auxiliaryData: { pass: i + 1, sortedFrom: n - i },
        });
      }
    }

    sortedIndices.push(n - 1 - i);
    steps.push({
      type: "sorted",
      array: [...arr],
      indices: [...sortedIndices],
      description: `Pass ${i + 1} complete. ${arr[n - 1 - i]} is in its final position.`,
      auxiliaryData: { pass: i + 1, sortedFrom: n - i - 1 },
    });

    if (!swapped) {
      steps.push({
        type: "info",
        array: [...arr],
        indices: Array.from({ length: n }, (_, k) => k),
        description: "No swaps in this pass — array is already sorted! Early termination.",
      });
      break;
    }
  }

  // Final sorted state
  const allIndices = Array.from({ length: n }, (_, k) => k);
  steps.push({
    type: "sorted",
    array: [...arr],
    indices: allIndices,
    description: `Bubble Sort complete! Sorted array: [${arr.join(", ")}]`,
  });

  return steps;
}

export const bubbleSortMeta = {
  id: "bubble-sort",
  implementations: {
    typescript: `function bubbleSort(arr: number[]): number[] {
  const n = arr.length;
  
  for (let i = 0; i < n - 1; i++) {
    let swapped = false;
    
    for (let j = 0; j < n - i - 1; j++) {
      if (arr[j] > arr[j + 1]) {
        [arr[j], arr[j + 1]] = [arr[j + 1], arr[j]];
        swapped = true;
      }
    }
    
    // Optimization: early exit if no swaps
    if (!swapped) break;
  }
  
  return arr;
}`,
    python: `def bubble_sort(arr: list[int]) -> list[int]:
    n = len(arr)
    
    for i in range(n - 1):
        swapped = False
        
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        
        if not swapped:
            break
    
    return arr`,
    java: `public static int[] bubbleSort(int[] arr) {
    int n = arr.length;
    
    for (int i = 0; i < n - 1; i++) {
        boolean swapped = false;
        
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
                swapped = true;
            }
        }
        
        if (!swapped) break;
    }
    
    return arr;
}`,
    cpp: `vector<int> bubbleSort(vector<int> arr) {
    int n = arr.size();
    
    for (int i = 0; i < n - 1; i++) {
        bool swapped = false;
        
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
                swapped = true;
            }
        }
        
        if (!swapped) break;
    }
    
    return arr;
}`,
  },
};
