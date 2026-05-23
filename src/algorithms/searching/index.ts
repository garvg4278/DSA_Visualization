import type { ArrayStep } from "@/types";

// ─── Linear Search ────────────────────────────────────────────────────────────

export function linearSearchSteps(
  inputArray: number[],
  target: number
): ArrayStep[] {
  const steps: ArrayStep[] = [];
  const arr = [...inputArray];

  steps.push({
    type: "info",
    array: [...arr],
    indices: [],
    description: `Linear Search for target = ${target} in [${arr.join(", ")}]`,
  });

  for (let i = 0; i < arr.length; i++) {
    steps.push({
      type: "compare",
      array: [...arr],
      indices: [i],
      description: `Checking index ${i}: arr[${i}] = ${arr[i]} ${arr[i] === target ? "===" : "!=="} ${target}`,
      auxiliaryData: { checked: i },
    });

    if (arr[i] === target) {
      steps.push({
        type: "found",
        array: [...arr],
        indices: [i],
        description: `✓ Found ${target} at index ${i}!`,
        auxiliaryData: { foundAt: i },
      });
      return steps;
    }
  }

  steps.push({
    type: "not-found",
    array: [...arr],
    indices: [],
    description: `✗ ${target} not found after checking all ${arr.length} elements.`,
  });

  return steps;
}

// ─── Binary Search ────────────────────────────────────────────────────────────

export function binarySearchSteps(
  inputArray: number[],
  target: number
): ArrayStep[] {
  const steps: ArrayStep[] = [];
  const arr = [...inputArray].sort((a, b) => a - b);

  steps.push({
    type: "info",
    array: [...arr],
    indices: [],
    description: `Binary Search for ${target} in sorted array [${arr.join(", ")}]`,
  });

  let lo = 0;
  let hi = arr.length - 1;

  while (lo <= hi) {
    const mid = Math.floor((lo + hi) / 2);

    steps.push({
      type: "compare",
      array: [...arr],
      indices: [lo, mid, hi],
      description: `lo=${lo}, hi=${hi} → mid=${mid}, arr[mid]=${arr[mid]}`,
      auxiliaryData: { lo, mid, hi },
    });

    if (arr[mid] === target) {
      steps.push({
        type: "found",
        array: [...arr],
        indices: [mid],
        description: `✓ Found ${target} at index ${mid}!`,
        auxiliaryData: { lo, mid, hi, foundAt: mid },
      });
      return steps;
    } else if (arr[mid] < target) {
      steps.push({
        type: "highlight",
        array: [...arr],
        indices: Array.from({ length: mid - lo + 1 }, (_, k) => lo + k),
        description: `${arr[mid]} < ${target} → eliminate left half, search [${mid + 1}..${hi}]`,
        auxiliaryData: { lo, mid, hi, eliminated: "left" },
      });
      lo = mid + 1;
    } else {
      steps.push({
        type: "highlight",
        array: [...arr],
        indices: Array.from({ length: hi - mid }, (_, k) => mid + 1 + k),
        description: `${arr[mid]} > ${target} → eliminate right half, search [${lo}..${mid - 1}]`,
        auxiliaryData: { lo, mid, hi, eliminated: "right" },
      });
      hi = mid - 1;
    }
  }

  steps.push({
    type: "not-found",
    array: [...arr],
    indices: [],
    description: `✗ ${target} not found. Search space exhausted.`,
  });

  return steps;
}

// ─── Jump Search ──────────────────────────────────────────────────────────────

export function jumpSearchSteps(
  inputArray: number[],
  target: number
): ArrayStep[] {
  const steps: ArrayStep[] = [];
  const arr = [...inputArray].sort((a, b) => a - b);
  const n = arr.length;
  const jumpSize = Math.floor(Math.sqrt(n));

  steps.push({
    type: "info",
    array: [...arr],
    indices: [],
    description: `Jump Search for ${target}. Jump size = √${n} ≈ ${jumpSize}`,
  });

  let prev = 0;
  let curr = jumpSize;

  // Jump phase
  while (curr < n && arr[curr] < target) {
    steps.push({
      type: "highlight",
      array: [...arr],
      indices: [curr],
      description: `Jumping to index ${curr}: arr[${curr}]=${arr[curr]} < ${target}, jump again`,
      auxiliaryData: { phase: "jump", jumpSize, prev, curr },
    });
    prev = curr;
    curr += jumpSize;
  }

  steps.push({
    type: "highlight",
    array: [...arr],
    indices: [Math.min(curr, n - 1)],
    description: `Block found! Target might be in [${prev}..${Math.min(curr, n - 1)}]. Linear scan...`,
    auxiliaryData: { phase: "linear", prev, curr: Math.min(curr, n - 1) },
  });

  // Linear scan in block
  for (let i = prev; i <= Math.min(curr, n - 1); i++) {
    steps.push({
      type: "compare",
      array: [...arr],
      indices: [i],
      description: `Linear scan: arr[${i}]=${arr[i]} ${arr[i] === target ? "===" : "!=="} ${target}`,
      auxiliaryData: { phase: "linear" },
    });

    if (arr[i] === target) {
      steps.push({
        type: "found",
        array: [...arr],
        indices: [i],
        description: `✓ Found ${target} at index ${i}!`,
      });
      return steps;
    }
  }

  steps.push({
    type: "not-found",
    array: [...arr],
    indices: [],
    description: `✗ ${target} not found in the identified block.`,
  });

  return steps;
}

// ─── Code Implementations ─────────────────────────────────────────────────────

export const searchingImplementations = {
  "linear-search": {
    typescript: `function linearSearch(arr: number[], target: number): number {
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] === target) return i;
  }
  return -1; // not found
}`,
    python: `def linear_search(arr: list[int], target: int) -> int:
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1`,
    java: `public static int linearSearch(int[] arr, int target) {
    for (int i = 0; i < arr.length; i++) {
        if (arr[i] == target) return i;
    }
    return -1;
}`,
    cpp: `int linearSearch(vector<int>& arr, int target) {
    for (int i = 0; i < arr.size(); i++) {
        if (arr[i] == target) return i;
    }
    return -1;
}`,
  },
  "binary-search": {
    typescript: `function binarySearch(arr: number[], target: number): number {
  let lo = 0, hi = arr.length - 1;
  
  while (lo <= hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (arr[mid] === target) return mid;
    else if (arr[mid] < target) lo = mid + 1;
    else hi = mid - 1;
  }
  
  return -1;
}`,
    python: `def binary_search(arr: list[int], target: int) -> int:
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1`,
    java: `public static int binarySearch(int[] arr, int target) {
    int lo = 0, hi = arr.length - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (arr[mid] == target) return mid;
        else if (arr[mid] < target) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}`,
    cpp: `int binarySearch(vector<int>& arr, int target) {
    int lo = 0, hi = arr.size() - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (arr[mid] == target) return mid;
        else if (arr[mid] < target) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}`,
  },
  "jump-search": {
    typescript: `function jumpSearch(arr: number[], target: number): number {
  const n = arr.length;
  const step = Math.floor(Math.sqrt(n));
  let prev = 0, curr = step;
  
  while (curr < n && arr[curr] < target) {
    prev = curr;
    curr += step;
  }
  
  for (let i = prev; i <= Math.min(curr, n - 1); i++) {
    if (arr[i] === target) return i;
  }
  
  return -1;
}`,
    python: `import math

def jump_search(arr, target):
    n = len(arr)
    step = int(math.sqrt(n))
    prev = 0
    curr = step
    
    while curr < n and arr[curr] < target:
        prev = curr
        curr += step
    
    for i in range(prev, min(curr + 1, n)):
        if arr[i] == target:
            return i
    return -1`,
    java: `public static int jumpSearch(int[] arr, int target) {
    int n = arr.length;
    int step = (int) Math.sqrt(n);
    int prev = 0, curr = step;
    
    while (curr < n && arr[curr] < target) {
        prev = curr;
        curr += step;
    }
    
    for (int i = prev; i <= Math.min(curr, n - 1); i++) {
        if (arr[i] == target) return i;
    }
    return -1;
}`,
    cpp: `int jumpSearch(vector<int>& arr, int target) {
    int n = arr.size();
    int step = sqrt(n);
    int prev = 0, curr = step;
    
    while (curr < n && arr[curr] < target) {
        prev = curr;
        curr += step;
    }
    
    for (int i = prev; i <= min(curr, n - 1); i++) {
        if (arr[i] == target) return i;
    }
    return -1;
}`,
  },
};
