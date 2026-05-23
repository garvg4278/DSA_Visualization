import type { ArrayStep } from "@/types";

// ─── Insertion Sort ───────────────────────────────────────────────────────────

export function insertionSortSteps(
  inputArray: number[],
  order: "asc" | "desc" = "asc"
): ArrayStep[] {
  const steps: ArrayStep[] = [];
  const arr = [...inputArray];
  const n = arr.length;

  steps.push({
    type: "sorted",
    array: [...arr],
    indices: [0],
    description: `Starting Insertion Sort. Element ${arr[0]} is trivially sorted.`,
  });

  for (let i = 1; i < n; i++) {
    const key = arr[i];
    let j = i - 1;

    steps.push({
      type: "highlight",
      array: [...arr],
      indices: [i],
      description: `Inserting ${key} into the sorted portion [${arr.slice(0, i).join(", ")}]`,
    });

    while (j >= 0 && (order === "asc" ? arr[j] > key : arr[j] < key)) {
      steps.push({
        type: "compare",
        array: [...arr],
        indices: [j, j + 1],
        description: `${arr[j]} > ${key}, shifting ${arr[j]} right`,
      });

      arr[j + 1] = arr[j];
      steps.push({
        type: "overwrite",
        array: [...arr],
        indices: [j + 1],
        description: `Shifted ${arr[j + 1]} to position ${j + 1}`,
      });
      j--;
    }

    arr[j + 1] = key;
    steps.push({
      type: "sorted",
      array: [...arr],
      indices: Array.from({ length: i + 1 }, (_, k) => k),
      description: `Inserted ${key} at position ${j + 1}. Sorted so far: [${arr.slice(0, i + 1).join(", ")}]`,
    });
  }

  steps.push({
    type: "sorted",
    array: [...arr],
    indices: Array.from({ length: n }, (_, k) => k),
    description: `Insertion Sort complete! [${arr.join(", ")}]`,
  });

  return steps;
}

// ─── Merge Sort ───────────────────────────────────────────────────────────────

export function mergeSortSteps(
  inputArray: number[],
  order: "asc" | "desc" = "asc"
): ArrayStep[] {
  const steps: ArrayStep[] = [];
  const arr = [...inputArray];

  steps.push({
    type: "info",
    array: [...arr],
    indices: [],
    description: `Starting Merge Sort — divide and conquer on [${arr.join(", ")}]`,
  });

  function mergeSort(a: number[], lo: number, hi: number): void {
    if (lo >= hi) return;
    const mid = Math.floor((lo + hi) / 2);

    steps.push({
      type: "partition",
      array: [...arr],
      indices: Array.from({ length: hi - lo + 1 }, (_, k) => lo + k),
      description: `Dividing [${lo}..${hi}] into [${lo}..${mid}] and [${mid + 1}..${hi}]`,
    });

    mergeSort(a, lo, mid);
    mergeSort(a, mid + 1, hi);

    // Merge
    const left = a.slice(lo, mid + 1);
    const right = a.slice(mid + 1, hi + 1);
    let i = 0, j = 0, k = lo;

    while (i < left.length && j < right.length) {
      steps.push({
        type: "compare",
        array: [...arr],
        indices: [lo + i, mid + 1 + j],
        description: `Merging: comparing ${left[i]} and ${right[j]}`,
      });

      if (order === "asc" ? left[i] <= right[j] : left[i] >= right[j]) {
        arr[k++] = left[i++];
      } else {
        arr[k++] = right[j++];
      }

      steps.push({
        type: "merge",
        array: [...arr],
        indices: [k - 1],
        description: `Placed ${arr[k - 1]} at position ${k - 1}`,
      });
    }

    while (i < left.length) { arr[k++] = left[i++]; }
    while (j < right.length) { arr[k++] = right[j++]; }

    steps.push({
      type: "sorted",
      array: [...arr],
      indices: Array.from({ length: hi - lo + 1 }, (_, k) => lo + k),
      description: `Merged [${lo}..${hi}]: [${arr.slice(lo, hi + 1).join(", ")}]`,
    });
  }

  mergeSort(arr, 0, arr.length - 1);

  steps.push({
    type: "sorted",
    array: [...arr],
    indices: Array.from({ length: arr.length }, (_, k) => k),
    description: `Merge Sort complete! [${arr.join(", ")}]`,
  });

  return steps;
}

// ─── Quick Sort ───────────────────────────────────────────────────────────────

export function quickSortSteps(
  inputArray: number[],
  order: "asc" | "desc" = "asc"
): ArrayStep[] {
  const steps: ArrayStep[] = [];
  const arr = [...inputArray];
  const sortedIndices = new Set<number>();

  steps.push({
    type: "info",
    array: [...arr],
    indices: [],
    description: `Starting Quick Sort on [${arr.join(", ")}]`,
  });

  function quickSort(lo: number, hi: number): void {
    if (lo >= hi) {
      if (lo === hi) sortedIndices.add(lo);
      return;
    }

    // Lomuto partition
    const pivotVal = arr[hi];
    steps.push({
      type: "pivot",
      array: [...arr],
      indices: [hi],
      description: `Pivot selected: ${pivotVal} at index ${hi}`,
      auxiliaryData: { lo, hi, pivot: hi },
    });

    let s = lo - 1;
    for (let j = lo; j < hi; j++) {
      steps.push({
        type: "compare",
        array: [...arr],
        indices: [j, hi],
        description: `Comparing ${arr[j]} with pivot ${pivotVal}`,
        auxiliaryData: { lo, hi, pivot: hi, storeIdx: s + 1 },
      });

      const cond = order === "asc" ? arr[j] <= pivotVal : arr[j] >= pivotVal;
      if (cond) {
        s++;
        [arr[s], arr[j]] = [arr[j], arr[s]];
        if (s !== j) {
          steps.push({
            type: "swap",
            array: [...arr],
            indices: [s, j],
            description: `Swapped ${arr[j]} and ${arr[s]}`,
            auxiliaryData: { lo, hi, pivot: hi },
          });
        }
      }
    }

    [arr[s + 1], arr[hi]] = [arr[hi], arr[s + 1]];
    sortedIndices.add(s + 1);

    steps.push({
      type: "sorted",
      array: [...arr],
      indices: [s + 1],
      description: `Pivot ${pivotVal} placed at final position ${s + 1}`,
      auxiliaryData: { lo, hi, pivot: s + 1 },
    });

    quickSort(lo, s);
    quickSort(s + 2, hi);
  }

  quickSort(0, arr.length - 1);

  steps.push({
    type: "sorted",
    array: [...arr],
    indices: Array.from({ length: arr.length }, (_, k) => k),
    description: `Quick Sort complete! [${arr.join(", ")}]`,
  });

  return steps;
}

// ─── Heap Sort ────────────────────────────────────────────────────────────────

export function heapSortSteps(
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
    description: `Starting Heap Sort — building ${order === "asc" ? "max" : "min"}-heap`,
  });

  function heapify(size: number, root: number): void {
    let extreme = root;
    const left = 2 * root + 1;
    const right = 2 * root + 2;

    steps.push({
      type: "compare",
      array: [...arr],
      indices: [root, left, right].filter((i) => i < size),
      description: `Heapifying at root=${root}, left=${left}, right=${right}`,
      auxiliaryData: { sortedIndices: [...sortedIndices] },
    });

    if (left < size) {
      const cond = order === "asc" ? arr[left] > arr[extreme] : arr[left] < arr[extreme];
      if (cond) extreme = left;
    }
    if (right < size) {
      const cond = order === "asc" ? arr[right] > arr[extreme] : arr[right] < arr[extreme];
      if (cond) extreme = right;
    }

    if (extreme !== root) {
      [arr[root], arr[extreme]] = [arr[extreme], arr[root]];
      steps.push({
        type: "swap",
        array: [...arr],
        indices: [root, extreme],
        description: `Swapped ${arr[extreme]} and ${arr[root]} to maintain heap property`,
        auxiliaryData: { sortedIndices: [...sortedIndices] },
      });
      heapify(size, extreme);
    }
  }

  // Build heap
  for (let i = Math.floor(n / 2) - 1; i >= 0; i--) {
    heapify(n, i);
  }

  steps.push({
    type: "info",
    array: [...arr],
    indices: [0],
    description: `Heap built! Root ${arr[0]} is the ${order === "asc" ? "maximum" : "minimum"}. Now extracting...`,
  });

  // Extract elements
  for (let i = n - 1; i > 0; i--) {
    [arr[0], arr[i]] = [arr[i], arr[0]];
    sortedIndices.push(i);

    steps.push({
      type: "sorted",
      array: [...arr],
      indices: [...sortedIndices],
      description: `Extracted ${arr[i]}, placed at index ${i}`,
    });

    heapify(i, 0);
  }

  sortedIndices.push(0);
  steps.push({
    type: "sorted",
    array: [...arr],
    indices: Array.from({ length: n }, (_, k) => k),
    description: `Heap Sort complete! [${arr.join(", ")}]`,
  });

  return steps;
}

// ─── Radix Sort ───────────────────────────────────────────────────────────────

export function radixSortSteps(inputArray: number[]): ArrayStep[] {
  const steps: ArrayStep[] = [];
  let arr = [...inputArray];
  const n = arr.length;

  function getDigit(num: number, place: number): number {
    return Math.floor(Math.abs(num) / Math.pow(10, place)) % 10;
  }

  function digitCount(num: number): number {
    if (num === 0) return 1;
    return Math.floor(Math.log10(Math.abs(num))) + 1;
  }

  const maxDigits = Math.max(...arr.map(digitCount));

  steps.push({
    type: "info",
    array: [...arr],
    indices: [],
    description: `Radix Sort: max digits = ${maxDigits}. Processing ${maxDigits} passes.`,
  });

  for (let k = 0; k < maxDigits; k++) {
    const buckets: number[][] = Array.from({ length: 10 }, () => []);

    steps.push({
      type: "partition",
      array: [...arr],
      indices: Array.from({ length: n }, (_, i) => i),
      description: `Pass ${k + 1}: Sorting by ${k === 0 ? "ones" : k === 1 ? "tens" : k === 2 ? "hundreds" : `10^${k}`} digit`,
    });

    for (let i = 0; i < n; i++) {
      const digit = getDigit(arr[i], k);
      buckets[digit].push(arr[i]);
      steps.push({
        type: "highlight",
        array: [...arr],
        indices: [i],
        description: `${arr[i]} → digit at position ${k} is ${digit} → bucket[${digit}]`,
        auxiliaryData: { digit, buckets: buckets.map((b) => [...b]) },
      });
    }

    arr = ([] as number[]).concat(...buckets);

    steps.push({
      type: "merge",
      array: [...arr],
      indices: Array.from({ length: n }, (_, i) => i),
      description: `After pass ${k + 1}: [${arr.join(", ")}]`,
      auxiliaryData: { buckets: buckets.map((b) => [...b]) },
    });
  }

  steps.push({
    type: "sorted",
    array: [...arr],
    indices: Array.from({ length: n }, (_, k) => k),
    description: `Radix Sort complete! [${arr.join(", ")}]`,
  });

  return steps;
}

// ─── Counting Sort ────────────────────────────────────────────────────────────

export function countingSortSteps(inputArray: number[]): ArrayStep[] {
  const steps: ArrayStep[] = [];
  const arr = [...inputArray];
  const n = arr.length;
  const maxVal = Math.max(...arr);

  steps.push({
    type: "info",
    array: [...arr],
    indices: [],
    description: `Counting Sort: range [0, ${maxVal}]`,
  });

  const count = new Array(maxVal + 1).fill(0);

  // Count occurrences
  for (let i = 0; i < n; i++) {
    count[arr[i]]++;
    steps.push({
      type: "highlight",
      array: [...arr],
      indices: [i],
      description: `Counting ${arr[i]}: count[${arr[i]}] = ${count[arr[i]]}`,
      auxiliaryData: { count: [...count] },
    });
  }

  // Cumulative
  for (let i = 1; i <= maxVal; i++) {
    count[i] += count[i - 1];
  }
  steps.push({
    type: "info",
    array: [...arr],
    indices: [],
    description: `Built cumulative count array. Now reconstructing sorted array.`,
    auxiliaryData: { count: [...count] },
  });

  const output = new Array(n);
  for (let i = n - 1; i >= 0; i--) {
    output[count[arr[i]] - 1] = arr[i];
    count[arr[i]]--;
    steps.push({
      type: "overwrite",
      array: [...arr],
      indices: [i],
      description: `Placing ${arr[i]} at position ${count[arr[i]]}`,
      auxiliaryData: { output: [...output] },
    });
  }

  steps.push({
    type: "sorted",
    array: output,
    indices: Array.from({ length: n }, (_, k) => k),
    description: `Counting Sort complete! [${output.join(", ")}]`,
  });

  return steps;
}

// ─── Code Implementations ─────────────────────────────────────────────────────

export const sortingImplementations = {
  "insertion-sort": {
    typescript: `function insertionSort(arr: number[]): number[] {
  for (let i = 1; i < arr.length; i++) {
    const key = arr[i];
    let j = i - 1;
    
    while (j >= 0 && arr[j] > key) {
      arr[j + 1] = arr[j];
      j--;
    }
    arr[j + 1] = key;
  }
  return arr;
}`,
    python: `def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr`,
    java: `public static int[] insertionSort(int[] arr) {
    for (int i = 1; i < arr.length; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
    return arr;
}`,
    cpp: `vector<int> insertionSort(vector<int> arr) {
    for (int i = 1; i < arr.size(); i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
    return arr;
}`,
  },
  "merge-sort": {
    typescript: `function mergeSort(arr: number[]): number[] {
  if (arr.length <= 1) return arr;
  const mid = Math.floor(arr.length / 2);
  const left = mergeSort(arr.slice(0, mid));
  const right = mergeSort(arr.slice(mid));
  return merge(left, right);
}

function merge(left: number[], right: number[]): number[] {
  const result: number[] = [];
  let i = 0, j = 0;
  while (i < left.length && j < right.length) {
    if (left[i] <= right[j]) result.push(left[i++]);
    else result.push(right[j++]);
  }
  return result.concat(left.slice(i), right.slice(j));
}`,
    python: `def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    return result + left[i:] + right[j:]`,
    java: `public static int[] mergeSort(int[] arr) {
    if (arr.length <= 1) return arr;
    int mid = arr.length / 2;
    int[] left = mergeSort(Arrays.copyOfRange(arr, 0, mid));
    int[] right = mergeSort(Arrays.copyOfRange(arr, mid, arr.length));
    return merge(left, right);
}`,
    cpp: `vector<int> mergeSort(vector<int> arr) {
    if (arr.size() <= 1) return arr;
    int mid = arr.size() / 2;
    auto left = mergeSort({arr.begin(), arr.begin() + mid});
    auto right = mergeSort({arr.begin() + mid, arr.end()});
    return merge(left, right);
}`,
  },
  "quick-sort": {
    typescript: `function quickSort(arr: number[], lo = 0, hi = arr.length - 1): number[] {
  if (lo < hi) {
    const pivot = partition(arr, lo, hi);
    quickSort(arr, lo, pivot - 1);
    quickSort(arr, pivot + 1, hi);
  }
  return arr;
}

function partition(arr: number[], lo: number, hi: number): number {
  const pivot = arr[hi];
  let i = lo - 1;
  for (let j = lo; j < hi; j++) {
    if (arr[j] <= pivot) {
      [arr[++i], arr[j]] = [arr[j], arr[i]];
    }
  }
  [arr[i + 1], arr[hi]] = [arr[hi], arr[i + 1]];
  return i + 1;
}`,
    python: `def quick_sort(arr, lo=0, hi=None):
    if hi is None: hi = len(arr) - 1
    if lo < hi:
        pivot = partition(arr, lo, hi)
        quick_sort(arr, lo, pivot - 1)
        quick_sort(arr, pivot + 1, hi)
    return arr

def partition(arr, lo, hi):
    pivot = arr[hi]
    i = lo - 1
    for j in range(lo, hi):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i+1], arr[hi] = arr[hi], arr[i+1]
    return i + 1`,
    java: `public static void quickSort(int[] arr, int lo, int hi) {
    if (lo < hi) {
        int pivot = partition(arr, lo, hi);
        quickSort(arr, lo, pivot - 1);
        quickSort(arr, pivot + 1, hi);
    }
}`,
    cpp: `void quickSort(vector<int>& arr, int lo, int hi) {
    if (lo < hi) {
        int pivot = partition(arr, lo, hi);
        quickSort(arr, lo, pivot - 1);
        quickSort(arr, pivot + 1, hi);
    }
}`,
  },
};
