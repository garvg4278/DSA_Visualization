// Pseudocode + multi-approach implementations for every algorithm

export interface Approach {
  label: string;        // "Brute Force" | "Better" | "Optimal"
  badge: string;        // "O(n²)" etc.
  badgeColor: string;
  description: string;
  pseudocode: string;
  code: {
    cpp: string;
    java: string;
    python: string;
  };
}

export interface AlgorithmApproaches {
  approaches: Approach[];
}

export const PSEUDOCODE_DATA: Record<string, AlgorithmApproaches> = {

  // ───────────────────────── BUBBLE SORT ──────────────────────────────────
  "bubble-sort": {
    approaches: [
      {
        label: "Brute Force", badge: "O(n²)", badgeColor: "hsl(346 87% 65%)",
        description: "Compare every adjacent pair in each pass. No optimization — always does n×(n-1)/2 comparisons.",
        pseudocode: `BUBBLE-SORT(A):
  n = length(A)
  FOR i = 0 TO n-2:
    FOR j = 0 TO n-i-2:
      IF A[j] > A[j+1]:
        SWAP(A[j], A[j+1])
  RETURN A`,
        code: {
          cpp: `void bubbleSort(vector<int>& A) {
    int n = A.size();
    for (int i = 0; i < n-1; i++)
        for (int j = 0; j < n-i-1; j++)
            if (A[j] > A[j+1])
                swap(A[j], A[j+1]);
}`,
          java: `void bubbleSort(int[] A) {
    int n = A.length;
    for (int i = 0; i < n-1; i++)
        for (int j = 0; j < n-i-1; j++)
            if (A[j] > A[j+1]) {
                int t = A[j]; A[j] = A[j+1]; A[j+1] = t;
            }
}`,
          python: `def bubble_sort(A):
    n = len(A)
    for i in range(n - 1):
        for j in range(n - i - 1):
            if A[j] > A[j + 1]:
                A[j], A[j + 1] = A[j + 1], A[j]
    return A`,
        },
      },
      {
        label: "Optimal", badge: "O(n) best", badgeColor: "hsl(142 71% 52%)",
        description: "Add a `swapped` flag. If no swap occurs in a pass the array is already sorted — exit early. Best case becomes O(n).",
        pseudocode: `BUBBLE-SORT-OPTIMIZED(A):
  n = length(A)
  FOR i = 0 TO n-2:
    swapped = FALSE
    FOR j = 0 TO n-i-2:
      IF A[j] > A[j+1]:
        SWAP(A[j], A[j+1])
        swapped = TRUE
    IF NOT swapped:
      BREAK          // already sorted!
  RETURN A`,
        code: {
          cpp: `void bubbleSortOpt(vector<int>& A) {
    int n = A.size();
    for (int i = 0; i < n-1; i++) {
        bool swapped = false;
        for (int j = 0; j < n-i-1; j++)
            if (A[j] > A[j+1]) {
                swap(A[j], A[j+1]);
                swapped = true;
            }
        if (!swapped) break; // O(n) if nearly sorted
    }
}`,
          java: `void bubbleSortOpt(int[] A) {
    int n = A.length;
    for (int i = 0; i < n-1; i++) {
        boolean swapped = false;
        for (int j = 0; j < n-i-1; j++)
            if (A[j] > A[j+1]) {
                int t = A[j]; A[j] = A[j+1]; A[j+1] = t;
                swapped = true;
            }
        if (!swapped) break;
    }
}`,
          python: `def bubble_sort_opt(A):
    n = len(A)
    for i in range(n - 1):
        swapped = False
        for j in range(n - i - 1):
            if A[j] > A[j + 1]:
                A[j], A[j + 1] = A[j + 1], A[j]
                swapped = True
        if not swapped:
            break   # Already sorted — O(n) best case
    return A`,
        },
      },
    ],
  },

  // ───────────────────────── SELECTION SORT ───────────────────────────────
  "selection-sort": {
    approaches: [
      {
        label: "Standard", badge: "O(n²)", badgeColor: "hsl(346 87% 65%)",
        description: "Find min in unsorted portion, place at front. Makes exactly n-1 swaps regardless of input.",
        pseudocode: `SELECTION-SORT(A):
  n = length(A)
  FOR i = 0 TO n-2:
    minIdx = i
    FOR j = i+1 TO n-1:
      IF A[j] < A[minIdx]:
        minIdx = j
    SWAP(A[i], A[minIdx])
  RETURN A`,
        code: {
          cpp: `void selectionSort(vector<int>& A) {
    int n = A.size();
    for (int i = 0; i < n-1; i++) {
        int mi = i;
        for (int j = i+1; j < n; j++)
            if (A[j] < A[mi]) mi = j;
        swap(A[i], A[mi]);
    }
}`,
          java: `void selectionSort(int[] A) {
    int n = A.length;
    for (int i = 0; i < n-1; i++) {
        int mi = i;
        for (int j = i+1; j < n; j++)
            if (A[j] < A[mi]) mi = j;
        int t = A[i]; A[i] = A[mi]; A[mi] = t;
    }
}`,
          python: `def selection_sort(A):
    n = len(A)
    for i in range(n - 1):
        mi = i
        for j in range(i + 1, n):
            if A[j] < A[mi]:
                mi = j
        A[i], A[mi] = A[mi], A[i]
    return A`,
        },
      },
    ],
  },

  // ───────────────────────── INSERTION SORT ───────────────────────────────
  "insertion-sort": {
    approaches: [
      {
        label: "Standard", badge: "O(n²)", badgeColor: "hsl(346 87% 65%)",
        description: "Shift elements right to make room. Best case O(n) on sorted input.",
        pseudocode: `INSERTION-SORT(A):
  FOR i = 1 TO n-1:
    key = A[i]
    j = i - 1
    WHILE j >= 0 AND A[j] > key:
      A[j+1] = A[j]   // shift right
      j = j - 1
    A[j+1] = key      // place key
  RETURN A`,
        code: {
          cpp: `void insertionSort(vector<int>& A) {
    for (int i = 1; i < A.size(); i++) {
        int key = A[i], j = i-1;
        while (j >= 0 && A[j] > key)
            A[j+1] = A[j--];
        A[j+1] = key;
    }
}`,
          java: `void insertionSort(int[] A) {
    for (int i = 1; i < A.length; i++) {
        int key = A[i], j = i-1;
        while (j >= 0 && A[j] > key)
            A[j+1] = A[j--];
        A[j+1] = key;
    }
}`,
          python: `def insertion_sort(A):
    for i in range(1, len(A)):
        key = A[i]
        j = i - 1
        while j >= 0 and A[j] > key:
            A[j + 1] = A[j]
            j -= 1
        A[j + 1] = key
    return A`,
        },
      },
      {
        label: "Binary Insertion", badge: "O(n log n) cmps", badgeColor: "hsl(38 92% 58%)",
        description: "Use binary search to find insertion position. Reduces comparisons to O(n log n) but shifts are still O(n²).",
        pseudocode: `BINARY-INSERTION-SORT(A):
  FOR i = 1 TO n-1:
    key = A[i]
    // Binary search in A[0..i-1]
    lo = 0, hi = i
    WHILE lo < hi:
      mid = (lo + hi) / 2
      IF A[mid] <= key: lo = mid + 1
      ELSE: hi = mid
    // Shift A[lo..i-1] right
    FOR j = i-1 DOWNTO lo:
      A[j+1] = A[j]
    A[lo] = key`,
        code: {
          cpp: `void binaryInsertionSort(vector<int>& A) {
    for (int i = 1; i < (int)A.size(); i++) {
        int key = A[i];
        int lo = 0, hi = i;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (A[mid] <= key) lo = mid + 1;
            else hi = mid;
        }
        for (int j = i; j > lo; j--) A[j] = A[j-1];
        A[lo] = key;
    }
}`,
          java: `void binaryInsertionSort(int[] A) {
    for (int i = 1; i < A.length; i++) {
        int key = A[i], lo = 0, hi = i;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (A[mid] <= key) lo = mid + 1;
            else hi = mid;
        }
        System.arraycopy(A, lo, A, lo+1, i-lo);
        A[lo] = key;
    }
}`,
          python: `import bisect
def binary_insertion_sort(A):
    for i in range(1, len(A)):
        key = A[i]
        lo = bisect.bisect_left(A, key, 0, i)
        A[lo+1:i+1] = A[lo:i]
        A[lo] = key
    return A`,
        },
      },
    ],
  },

  // ───────────────────────── MERGE SORT ───────────────────────────────────
  "merge-sort": {
    approaches: [
      {
        label: "Top-down (Recursive)", badge: "O(n log n)", badgeColor: "hsl(142 71% 52%)",
        description: "Recursively split, sort halves, then merge. Classic divide-and-conquer.",
        pseudocode: `MERGE-SORT(A, lo, hi):
  IF lo >= hi: RETURN
  mid = (lo + hi) / 2
  MERGE-SORT(A, lo, mid)
  MERGE-SORT(A, mid+1, hi)
  MERGE(A, lo, mid, hi)

MERGE(A, lo, mid, hi):
  L = A[lo..mid],  R = A[mid+1..hi]
  i = j = 0,  k = lo
  WHILE i < |L| AND j < |R|:
    IF L[i] <= R[j]: A[k++] = L[i++]
    ELSE:            A[k++] = R[j++]
  COPY remaining L or R into A`,
        code: {
          cpp: `void merge(vector<int>& A, int lo, int mid, int hi) {
    vector<int> L(A.begin()+lo, A.begin()+mid+1);
    vector<int> R(A.begin()+mid+1, A.begin()+hi+1);
    int i=0, j=0, k=lo;
    while (i<L.size() && j<R.size())
        A[k++] = (L[i]<=R[j]) ? L[i++] : R[j++];
    while (i<L.size()) A[k++]=L[i++];
    while (j<R.size()) A[k++]=R[j++];
}
void mergeSort(vector<int>& A, int lo, int hi) {
    if (lo >= hi) return;
    int mid = (lo+hi)/2;
    mergeSort(A, lo, mid);
    mergeSort(A, mid+1, hi);
    merge(A, lo, mid, hi);
}`,
          java: `void mergeSort(int[] A, int lo, int hi) {
    if (lo >= hi) return;
    int mid = (lo + hi) / 2;
    mergeSort(A, lo, mid);
    mergeSort(A, mid+1, hi);
    merge(A, lo, mid, hi);
}
void merge(int[] A, int lo, int mid, int hi) {
    int[] tmp = Arrays.copyOfRange(A, lo, hi+1);
    int i=0, j=mid-lo+1, k=lo;
    while (i<=mid-lo && j<=hi-lo)
        A[k++] = tmp[i]<=tmp[j] ? tmp[i++] : tmp[j++];
    while (i<=mid-lo) A[k++]=tmp[i++];
    while (j<=hi-lo) A[k++]=tmp[j++];
}`,
          python: `def merge_sort(A):
    if len(A) <= 1: return A
    mid = len(A) // 2
    L = merge_sort(A[:mid])
    R = merge_sort(A[mid:])
    i = j = 0
    result = []
    while i < len(L) and j < len(R):
        if L[i] <= R[j]: result.append(L[i]); i += 1
        else:             result.append(R[j]); j += 1
    return result + L[i:] + R[j:]`,
        },
      },
      {
        label: "Bottom-up (Iterative)", badge: "O(n log n)", badgeColor: "hsl(199 89% 60%)",
        description: "No recursion — merge sorted runs of size 1, then 2, then 4, etc. Same complexity, avoids call stack overhead.",
        pseudocode: `MERGE-SORT-ITERATIVE(A):
  n = length(A)
  size = 1
  WHILE size < n:
    FOR lo = 0 TO n-1 STEP 2*size:
      mid = min(lo + size - 1, n-1)
      hi  = min(lo + 2*size - 1, n-1)
      IF mid < hi:
        MERGE(A, lo, mid, hi)
    size = size * 2`,
        code: {
          cpp: `void mergeSortIter(vector<int>& A) {
    int n = A.size();
    for (int sz = 1; sz < n; sz *= 2)
        for (int lo = 0; lo < n-sz; lo += 2*sz) {
            int mid = lo+sz-1, hi = min(lo+2*sz-1,n-1);
            merge(A, lo, mid, hi); // same merge as above
        }
}`,
          java: `void mergeSortIter(int[] A) {
    int n = A.length;
    for (int sz = 1; sz < n; sz *= 2)
        for (int lo = 0; lo < n-sz; lo += 2*sz) {
            int mid = lo+sz-1, hi = Math.min(lo+2*sz-1,n-1);
            merge(A, lo, mid, hi);
        }
}`,
          python: `def merge_sort_iter(A):
    n = len(A)
    sz = 1
    while sz < n:
        for lo in range(0, n - sz, 2 * sz):
            mid = lo + sz - 1
            hi  = min(lo + 2 * sz - 1, n - 1)
            # inline merge
            L, R = A[lo:mid+1], A[mid+1:hi+1]
            i = j = 0; k = lo
            while i<len(L) and j<len(R):
                if L[i]<=R[j]: A[k]=L[i]; i+=1
                else: A[k]=R[j]; j+=1
                k+=1
            while i<len(L): A[k]=L[i]; i+=1; k+=1
            while j<len(R): A[k]=R[j]; j+=1; k+=1
        sz *= 2
    return A`,
        },
      },
    ],
  },

  // ───────────────────────── QUICK SORT ───────────────────────────────────
  "quick-sort": {
    approaches: [
      {
        label: "Lomuto Partition", badge: "O(n log n) avg", badgeColor: "hsl(38 92% 58%)",
        description: "Pivot = last element. Partition with one pointer. Simple but O(n²) on sorted input.",
        pseudocode: `QUICKSORT(A, lo, hi):
  IF lo < hi:
    p = PARTITION(A, lo, hi)
    QUICKSORT(A, lo, p-1)
    QUICKSORT(A, p+1, hi)

PARTITION(A, lo, hi):   // Lomuto
  pivot = A[hi]
  i = lo - 1
  FOR j = lo TO hi-1:
    IF A[j] <= pivot:
      i++
      SWAP(A[i], A[j])
  SWAP(A[i+1], A[hi])
  RETURN i+1`,
        code: {
          cpp: `int partition(vector<int>& A, int lo, int hi) {
    int pivot = A[hi], i = lo-1;
    for (int j = lo; j < hi; j++)
        if (A[j] <= pivot) swap(A[++i], A[j]);
    swap(A[i+1], A[hi]);
    return i+1;
}
void quickSort(vector<int>& A, int lo, int hi) {
    if (lo < hi) {
        int p = partition(A, lo, hi);
        quickSort(A, lo, p-1);
        quickSort(A, p+1, hi);
    }
}`,
          java: `int partition(int[] A, int lo, int hi) {
    int pivot = A[hi], i = lo-1;
    for (int j = lo; j < hi; j++)
        if (A[j] <= pivot) { int t=A[++i]; A[i]=A[j]; A[j]=t; }
    int t=A[i+1]; A[i+1]=A[hi]; A[hi]=t;
    return i+1;
}
void quickSort(int[] A, int lo, int hi) {
    if (lo < hi) {
        int p = partition(A, lo, hi);
        quickSort(A, lo, p-1);
        quickSort(A, p+1, hi);
    }
}`,
          python: `def quick_sort(A, lo=0, hi=None):
    if hi is None: hi = len(A)-1
    if lo < hi:
        p = partition(A, lo, hi)
        quick_sort(A, lo, p-1)
        quick_sort(A, p+1, hi)

def partition(A, lo, hi):
    pivot, i = A[hi], lo-1
    for j in range(lo, hi):
        if A[j] <= pivot:
            i += 1
            A[i], A[j] = A[j], A[i]
    A[i+1], A[hi] = A[hi], A[i+1]
    return i+1`,
        },
      },
      {
        label: "Hoare Partition", badge: "3x fewer swaps", badgeColor: "hsl(142 71% 52%)",
        description: "Two pointers from both ends. Uses ~3x fewer swaps than Lomuto. Pivot = first element.",
        pseudocode: `PARTITION-HOARE(A, lo, hi):
  pivot = A[lo]
  i = lo - 1,  j = hi + 1
  LOOP:
    REPEAT i++ UNTIL A[i] >= pivot
    REPEAT j-- UNTIL A[j] <= pivot
    IF i >= j: RETURN j
    SWAP(A[i], A[j])`,
        code: {
          cpp: `int hoarePartition(vector<int>& A, int lo, int hi) {
    int pivot = A[lo], i = lo-1, j = hi+1;
    while (true) {
        do i++; while (A[i] < pivot);
        do j--; while (A[j] > pivot);
        if (i >= j) return j;
        swap(A[i], A[j]);
    }
}`,
          java: `int hoarePartition(int[] A, int lo, int hi) {
    int pivot = A[lo], i = lo-1, j = hi+1;
    while (true) {
        do i++; while (A[i] < pivot);
        do j--; while (A[j] > pivot);
        if (i >= j) return j;
        int t=A[i]; A[i]=A[j]; A[j]=t;
    }
}`,
          python: `def hoare_partition(A, lo, hi):
    pivot = A[lo]
    i, j = lo - 1, hi + 1
    while True:
        i += 1
        while A[i] < pivot: i += 1
        j -= 1
        while A[j] > pivot: j -= 1
        if i >= j: return j
        A[i], A[j] = A[j], A[i]`,
        },
      },
      {
        label: "3-Way (Dutch Flag)", badge: "O(n) for dupes", badgeColor: "hsl(258 90% 70%)",
        description: "Partition into <pivot, =pivot, >pivot. Linear on arrays with many duplicate elements.",
        pseudocode: `3-WAY-QUICKSORT(A, lo, hi):
  IF lo >= hi: RETURN
  lt = lo, gt = hi, i = lo
  pivot = A[lo]
  WHILE i <= gt:
    IF A[i] < pivot: SWAP(A[lt++], A[i++])
    ELSE IF A[i] > pivot: SWAP(A[i], A[gt--])
    ELSE: i++
  // A[lo..lt-1] < pivot
  // A[lt..gt]   = pivot
  // A[gt+1..hi] > pivot
  3-WAY-QUICKSORT(A, lo, lt-1)
  3-WAY-QUICKSORT(A, gt+1, hi)`,
        code: {
          cpp: `void quickSort3Way(vector<int>& A, int lo, int hi) {
    if (lo >= hi) return;
    int lt=lo, gt=hi, i=lo;
    int pivot = A[lo];
    while (i <= gt) {
        if (A[i] < pivot)  swap(A[lt++], A[i++]);
        else if (A[i] > pivot) swap(A[i], A[gt--]);
        else i++;
    }
    quickSort3Way(A, lo, lt-1);
    quickSort3Way(A, gt+1, hi);
}`,
          java: `void quickSort3Way(int[] A, int lo, int hi) {
    if (lo >= hi) return;
    int lt=lo, gt=hi, i=lo, pivot=A[lo];
    while (i <= gt) {
        if (A[i] < pivot) { int t=A[lt];A[lt++]=A[i];A[i++]=t; }
        else if (A[i]>pivot){ int t=A[i];A[i]=A[gt];A[gt--]=t; }
        else i++;
    }
    quickSort3Way(A, lo, lt-1);
    quickSort3Way(A, gt+1, hi);
}`,
          python: `def quick_sort_3way(A, lo, hi):
    if lo >= hi: return
    lt, gt, i = lo, hi, lo
    pivot = A[lo]
    while i <= gt:
        if A[i] < pivot:
            A[lt], A[i] = A[i], A[lt]; lt += 1; i += 1
        elif A[i] > pivot:
            A[i], A[gt] = A[gt], A[i]; gt -= 1
        else:
            i += 1
    quick_sort_3way(A, lo, lt-1)
    quick_sort_3way(A, gt+1, hi)`,
        },
      },
    ],
  },

  // ───────────────────────── BINARY SEARCH ────────────────────────────────
  "binary-search": {
    approaches: [
      {
        label: "Iterative", badge: "O(log n)", badgeColor: "hsl(142 71% 52%)",
        description: "Eliminate half the search space each iteration. No recursion overhead.",
        pseudocode: `BINARY-SEARCH(A, target):
  lo = 0,  hi = length(A) - 1
  WHILE lo <= hi:
    mid = lo + (hi - lo) / 2   // avoids overflow
    IF A[mid] == target: RETURN mid
    IF A[mid] < target:  lo = mid + 1
    ELSE:                hi = mid - 1
  RETURN -1   // not found`,
        code: {
          cpp: `int binarySearch(vector<int>& A, int target) {
    int lo = 0, hi = A.size()-1;
    while (lo <= hi) {
        int mid = lo + (hi-lo)/2;
        if (A[mid] == target) return mid;
        if (A[mid] < target)  lo = mid+1;
        else                  hi = mid-1;
    }
    return -1;
}`,
          java: `int binarySearch(int[] A, int target) {
    int lo = 0, hi = A.length-1;
    while (lo <= hi) {
        int mid = lo + (hi-lo)/2;
        if (A[mid] == target) return mid;
        if (A[mid] < target)  lo = mid+1;
        else                  hi = mid-1;
    }
    return -1;
}`,
          python: `def binary_search(A, target):
    lo, hi = 0, len(A) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if A[mid] == target: return mid
        if A[mid] < target:  lo = mid + 1
        else:                hi = mid - 1
    return -1`,
        },
      },
      {
        label: "Recursive", badge: "O(log n)", badgeColor: "hsl(199 89% 60%)",
        description: "Same logic expressed recursively. Uses O(log n) stack space.",
        pseudocode: `BINARY-SEARCH-REC(A, lo, hi, target):
  IF lo > hi: RETURN -1
  mid = lo + (hi - lo) / 2
  IF A[mid] == target: RETURN mid
  IF A[mid] < target:
    RETURN BINARY-SEARCH-REC(A, mid+1, hi, target)
  ELSE:
    RETURN BINARY-SEARCH-REC(A, lo, mid-1, target)`,
        code: {
          cpp: `int bsRec(vector<int>& A, int lo, int hi, int t) {
    if (lo > hi) return -1;
    int mid = lo + (hi-lo)/2;
    if (A[mid] == t) return mid;
    return A[mid] < t ? bsRec(A, mid+1, hi, t)
                      : bsRec(A, lo, mid-1, t);
}`,
          java: `int bsRec(int[] A, int lo, int hi, int t) {
    if (lo > hi) return -1;
    int mid = lo + (hi-lo)/2;
    if (A[mid] == t) return mid;
    return A[mid] < t ? bsRec(A, mid+1, hi, t)
                      : bsRec(A, lo, mid-1, t);
}`,
          python: `def bs_rec(A, lo, hi, target):
    if lo > hi: return -1
    mid = (lo + hi) // 2
    if A[mid] == target: return mid
    if A[mid] < target: return bs_rec(A, mid+1, hi, target)
    return bs_rec(A, lo, mid-1, target)`,
        },
      },
    ],
  },

  // ───────────────────────── LINEAR SEARCH ────────────────────────────────
  "linear-search": {
    approaches: [
      {
        label: "Standard", badge: "O(n)", badgeColor: "hsl(38 92% 58%)",
        description: "Sequentially scan every element. Works on unsorted data.",
        pseudocode: `LINEAR-SEARCH(A, target):
  FOR i = 0 TO length(A)-1:
    IF A[i] == target:
      RETURN i    // found at index i
  RETURN -1       // not found`,
        code: {
          cpp: `int linearSearch(vector<int>& A, int t) {
    for (int i = 0; i < A.size(); i++)
        if (A[i] == t) return i;
    return -1;
}`,
          java: `int linearSearch(int[] A, int t) {
    for (int i = 0; i < A.length; i++)
        if (A[i] == t) return i;
    return -1;
}`,
          python: `def linear_search(A, target):
    for i, val in enumerate(A):
        if val == target:
            return i
    return -1`,
        },
      },
      {
        label: "Sentinel", badge: "Fewer comparisons", badgeColor: "hsl(142 71% 52%)",
        description: "Place target at end as sentinel. Eliminates the bounds check from inner loop — ~50% fewer comparisons per iteration.",
        pseudocode: `SENTINEL-SEARCH(A, target):
  n = length(A)
  last = A[n-1]
  A[n-1] = target       // sentinel
  i = 0
  WHILE A[i] != target:
    i++
  A[n-1] = last         // restore
  IF i < n-1 OR last == target:
    RETURN i
  RETURN -1`,
        code: {
          cpp: `int sentinelSearch(vector<int> A, int t) {
    int n = A.size(), last = A[n-1];
    A[n-1] = t;
    int i = 0;
    while (A[i] != t) i++;
    A[n-1] = last;
    if (i < n-1 || last == t) return i;
    return -1;
}`,
          java: `int sentinelSearch(int[] A, int t) {
    int n = A.length, last = A[n-1];
    A[n-1] = t;
    int i = 0;
    while (A[i] != t) i++;
    A[n-1] = last;
    return (i < n-1 || last == t) ? i : -1;
}`,
          python: `def sentinel_search(A, target):
    A = A[:]  # copy
    n = len(A)
    last = A[n-1]
    A[n-1] = target
    i = 0
    while A[i] != target: i += 1
    A[n-1] = last
    if i < n-1 or last == target: return i
    return -1`,
        },
      },
    ],
  },

  // ───────────────────────── BFS ──────────────────────────────────────────
  "bfs": {
    approaches: [
      {
        label: "Queue-based BFS", badge: "O(V+E)", badgeColor: "hsl(142 71% 52%)",
        description: "Level-order traversal using a queue. Guarantees shortest path in unweighted graphs.",
        pseudocode: `BFS(graph, start):
  visited = {start}
  queue = [start]
  order = []

  WHILE queue not empty:
    node = DEQUEUE(queue)
    order.append(node)

    FOR each neighbor OF node:
      IF neighbor NOT in visited:
        visited.add(neighbor)
        ENQUEUE(queue, neighbor)

  RETURN order`,
        code: {
          cpp: `vector<int> bfs(vector<vector<int>>& g, int src) {
    vector<bool> vis(g.size(), false);
    vector<int> order;
    queue<int> q;
    vis[src] = true; q.push(src);
    while (!q.empty()) {
        int u = q.front(); q.pop();
        order.push_back(u);
        for (int v : g[u])
            if (!vis[v]) { vis[v]=true; q.push(v); }
    }
    return order;
}`,
          java: `List<Integer> bfs(List<List<Integer>> g, int src) {
    boolean[] vis = new boolean[g.size()];
    List<Integer> order = new ArrayList<>();
    Queue<Integer> q = new LinkedList<>();
    vis[src] = true; q.offer(src);
    while (!q.isEmpty()) {
        int u = q.poll(); order.add(u);
        for (int v : g.get(u))
            if (!vis[v]) { vis[v]=true; q.offer(v); }
    }
    return order;
}`,
          python: `from collections import deque
def bfs(graph, src):
    visited = {src}
    queue = deque([src])
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for nb in graph[node]:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return order`,
        },
      },
    ],
  },

  // ───────────────────────── DFS ──────────────────────────────────────────
  "dfs": {
    approaches: [
      {
        label: "Recursive DFS", badge: "O(V+E)", badgeColor: "hsl(199 89% 60%)",
        description: "Naturally expresses the recursive call stack. Simpler code.",
        pseudocode: `DFS(graph, node, visited, order):
  visited.add(node)
  order.append(node)
  FOR each neighbor OF node:
    IF neighbor NOT in visited:
      DFS(graph, neighbor, visited, order)`,
        code: {
          cpp: `void dfs(vector<vector<int>>& g, int u, vector<bool>& vis, vector<int>& order) {
    vis[u] = true; order.push_back(u);
    for (int v : g[u])
        if (!vis[v]) dfs(g, v, vis, order);
}`,
          java: `void dfs(List<List<Integer>> g, int u, boolean[] vis, List<Integer> order) {
    vis[u] = true; order.add(u);
    for (int v : g.get(u))
        if (!vis[v]) dfs(g, v, vis, order);
}`,
          python: `def dfs(graph, node, visited=None, order=None):
    if visited is None: visited, order = set(), []
    visited.add(node); order.append(node)
    for nb in graph[node]:
        if nb not in visited:
            dfs(graph, nb, visited, order)
    return order`,
        },
      },
      {
        label: "Iterative DFS (Stack)", badge: "No stack overflow", badgeColor: "hsl(142 71% 52%)",
        description: "Use an explicit stack instead of recursion. Avoids stack overflow on deep graphs.",
        pseudocode: `DFS-ITERATIVE(graph, start):
  visited = {}
  stack = [start]
  order = []

  WHILE stack not empty:
    node = POP(stack)
    IF node NOT in visited:
      visited.add(node)
      order.append(node)
      FOR each neighbor OF node (reversed):
        IF neighbor NOT in visited:
          PUSH(stack, neighbor)`,
        code: {
          cpp: `vector<int> dfsIter(vector<vector<int>>& g, int src) {
    vector<bool> vis(g.size(), false);
    vector<int> order;
    stack<int> st; st.push(src);
    while (!st.empty()) {
        int u = st.top(); st.pop();
        if (vis[u]) continue;
        vis[u] = true; order.push_back(u);
        for (int i=g[u].size()-1; i>=0; i--)
            if (!vis[g[u][i]]) st.push(g[u][i]);
    }
    return order;
}`,
          java: `List<Integer> dfsIter(List<List<Integer>> g, int src) {
    boolean[] vis = new boolean[g.size()];
    List<Integer> order = new ArrayList<>();
    Deque<Integer> stack = new ArrayDeque<>();
    stack.push(src);
    while (!stack.isEmpty()) {
        int u = stack.pop();
        if (vis[u]) continue;
        vis[u]=true; order.add(u);
        List<Integer> nb = g.get(u);
        for (int i=nb.size()-1;i>=0;i--)
            if (!vis[nb.get(i)]) stack.push(nb.get(i));
    }
    return order;
}`,
          python: `def dfs_iter(graph, src):
    visited, order = set(), []
    stack = [src]
    while stack:
        node = stack.pop()
        if node in visited: continue
        visited.add(node); order.append(node)
        for nb in reversed(graph[node]):
            if nb not in visited:
                stack.append(nb)
    return order`,
        },
      },
    ],
  },

  // ───────────────────────── FIBONACCI ────────────────────────────────────
  "fibonacci": {
    approaches: [
      {
        label: "Brute Force (Recursion)", badge: "O(2ⁿ)", badgeColor: "hsl(346 87% 65%)",
        description: "Naive recursion recomputes sub-problems exponentially. F(50) makes ~10¹⁰ calls.",
        pseudocode: `FIB-NAIVE(n):
  IF n <= 1: RETURN n
  RETURN FIB-NAIVE(n-1) + FIB-NAIVE(n-2)`,
        code: {
          cpp: `int fib(int n) {
    if (n <= 1) return n;
    return fib(n-1) + fib(n-2); // O(2^n) calls!
}`,
          java: `int fib(int n) {
    if (n <= 1) return n;
    return fib(n-1) + fib(n-2);
}`,
          python: `def fib(n):
    if n <= 1: return n
    return fib(n-1) + fib(n-2)  # O(2^n)`,
        },
      },
      {
        label: "Memoization (Top-down)", badge: "O(n)", badgeColor: "hsl(38 92% 58%)",
        description: "Cache results of sub-problems. Each unique F(i) computed exactly once.",
        pseudocode: `memo = {}
FIB-MEMO(n):
  IF n <= 1: RETURN n
  IF n IN memo: RETURN memo[n]
  memo[n] = FIB-MEMO(n-1) + FIB-MEMO(n-2)
  RETURN memo[n]`,
        code: {
          cpp: `unordered_map<int,long long> memo;
long long fib(int n) {
    if (n <= 1) return n;
    if (memo.count(n)) return memo[n];
    return memo[n] = fib(n-1) + fib(n-2);
}`,
          java: `Map<Integer,Long> memo = new HashMap<>();
long fib(int n) {
    if (n <= 1) return n;
    if (memo.containsKey(n)) return memo.get(n);
    long result = fib(n-1) + fib(n-2);
    memo.put(n, result); return result;
}`,
          python: `from functools import lru_cache
@lru_cache(maxsize=None)
def fib(n):
    if n <= 1: return n
    return fib(n-1) + fib(n-2)`,
        },
      },
      {
        label: "Bottom-up DP (Optimal)", badge: "O(n) / O(1)", badgeColor: "hsl(142 71% 52%)",
        description: "Fill table iteratively. Space-optimized version uses only two variables — O(1) space.",
        pseudocode: `FIB-DP(n):
  IF n <= 1: RETURN n
  dp[0] = 0,  dp[1] = 1
  FOR i = 2 TO n:
    dp[i] = dp[i-1] + dp[i-2]
  RETURN dp[n]

// Space optimized:
FIB-O1(n):
  a = 0, b = 1
  FOR i = 2 TO n:
    a, b = b, a+b
  RETURN b`,
        code: {
          cpp: `long long fibDP(int n) {
    if (n<=1) return n;
    long long a=0, b=1;
    for (int i=2; i<=n; i++) { long long c=a+b; a=b; b=c; }
    return b;
}`,
          java: `long fibDP(int n) {
    if (n <= 1) return n;
    long a = 0, b = 1;
    for (int i = 2; i <= n; i++) { long c=a+b; a=b; b=c; }
    return b;
}`,
          python: `def fib_dp(n):
    if n <= 1: return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b`,
        },
      },
    ],
  },

  // ───────────────────────── DIJKSTRA ──────────────────────────────────────
  "dijkstra": {
    approaches: [
      {
        label: "Simple Array", badge: "O(V²)", badgeColor: "hsl(346 87% 65%)",
        description: "Scan all vertices to find minimum distance. Simple but O(V²) — good for dense graphs.",
        pseudocode: `DIJKSTRA-SIMPLE(graph, src):
  dist[src] = 0; all others = ∞
  visited = {}

  REPEAT V times:
    u = unvisited vertex with min dist
    visited.add(u)
    FOR each (u,v,w) edge:
      IF dist[u] + w < dist[v]:
        dist[v] = dist[u] + w   // relax

  RETURN dist`,
        code: {
          cpp: `vector<int> dijkstra(vector<vector<pair<int,int>>>& g, int src) {
    int n = g.size();
    vector<int> dist(n, INT_MAX);
    vector<bool> vis(n, false);
    dist[src] = 0;
    for (int i = 0; i < n; i++) {
        int u = -1;
        for (int j=0;j<n;j++) if (!vis[j] && (u==-1||dist[j]<dist[u])) u=j;
        if (dist[u]==INT_MAX) break;
        vis[u] = true;
        for (auto [v,w] : g[u])
            if (dist[u]+w < dist[v]) dist[v]=dist[u]+w;
    }
    return dist;
}`,
          java: `int[] dijkstra(List<int[]>[] g, int src) {
    int n = g.length;
    int[] dist = new int[n]; Arrays.fill(dist, Integer.MAX_VALUE);
    boolean[] vis = new boolean[n]; dist[src] = 0;
    for (int i=0; i<n; i++) {
        int u=-1;
        for (int j=0;j<n;j++) if (!vis[j]&&(u==-1||dist[j]<dist[u])) u=j;
        if (dist[u]==Integer.MAX_VALUE) break; vis[u]=true;
        for (int[] e : g[u]) if (dist[u]+e[1]<dist[e[0]]) dist[e[0]]=dist[u]+e[1];
    }
    return dist;
}`,
          python: `def dijkstra_simple(graph, src, n):
    dist = [float('inf')] * n
    visited = [False] * n
    dist[src] = 0
    for _ in range(n):
        u = min((i for i in range(n) if not visited[i]), key=lambda i: dist[i])
        if dist[u] == float('inf'): break
        visited[u] = True
        for v, w in graph[u]:
            if dist[u] + w < dist[v]: dist[v] = dist[u] + w
    return dist`,
        },
      },
      {
        label: "Min-Heap (Optimal)", badge: "O((V+E) log V)", badgeColor: "hsl(142 71% 52%)",
        description: "Use a priority queue to always extract the minimum in O(log V). Best for sparse graphs.",
        pseudocode: `DIJKSTRA-HEAP(graph, src):
  dist[src] = 0; all others = ∞
  pq = MinHeap{(0, src)}

  WHILE pq not empty:
    (d, u) = EXTRACT-MIN(pq)
    IF d > dist[u]: SKIP  // stale entry
    FOR each (u,v,w) edge:
      IF dist[u] + w < dist[v]:
        dist[v] = dist[u] + w
        INSERT(pq, (dist[v], v))

  RETURN dist`,
        code: {
          cpp: `vector<int> dijkstraHeap(vector<vector<pair<int,int>>>& g, int src) {
    int n = g.size();
    vector<int> dist(n, INT_MAX);
    priority_queue<pair<int,int>, vector<pair<int,int>>, greater<>> pq;
    dist[src] = 0; pq.push({0, src});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        for (auto [v, w] : g[u])
            if (dist[u]+w < dist[v]) { dist[v]=dist[u]+w; pq.push({dist[v],v}); }
    }
    return dist;
}`,
          java: `int[] dijkstraHeap(List<int[]>[] g, int src) {
    int n=g.length; int[] dist=new int[n]; Arrays.fill(dist,Integer.MAX_VALUE);
    PriorityQueue<int[]> pq=new PriorityQueue<>(Comparator.comparingInt(a->a[0]));
    dist[src]=0; pq.offer(new int[]{0,src});
    while (!pq.isEmpty()) {
        int[] cur=pq.poll(); int d=cur[0],u=cur[1];
        if (d>dist[u]) continue;
        for (int[] e:g[u]) if (dist[u]+e[1]<dist[e[0]]){dist[e[0]]=dist[u]+e[1];pq.offer(new int[]{dist[e[0]],e[0]});}
    }
    return dist;
}`,
          python: `import heapq
def dijkstra_heap(graph, src):
    dist = {node: float('inf') for node in graph}
    dist[src] = 0
    heap = [(0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]: continue
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(heap, (dist[v], v))
    return dist`,
        },
      },
    ],
  },

  // ───────────────────────── LCS ───────────────────────────────────────────
  "lcs": {
    approaches: [
      {
        label: "Brute Force", badge: "O(2ⁿ)", badgeColor: "hsl(346 87% 65%)",
        description: "Generate all subsequences of s1, check each against s2. Exponential — only feasible for very short strings.",
        pseudocode: `LCS-BRUTE(s1, s2):
  best = ""
  FOR each subsequence sub OF s1:
    IF sub is subsequence of s2:
      IF len(sub) > len(best):
        best = sub
  RETURN len(best)`,
        code: { cpp: "// O(2^n) — generate all subsets, check each", java: "// O(2^n) — generate all subsets, check each", python: "# O(2^n) — generate all subsets, check each" },
      },
      {
        label: "DP Table (Optimal)", badge: "O(mn)", badgeColor: "hsl(142 71% 52%)",
        description: "Build 2D table bottom-up. dp[i][j] = LCS of s1[0..i-1] and s2[0..j-1].",
        pseudocode: `LCS-DP(s1, s2):
  m = len(s1), n = len(s2)
  dp[0..m][0..n] = 0   // base case: empty string

  FOR i = 1 TO m:
    FOR j = 1 TO n:
      IF s1[i-1] == s2[j-1]:
        dp[i][j] = dp[i-1][j-1] + 1   // chars match!
      ELSE:
        dp[i][j] = max(dp[i-1][j], dp[i][j-1])

  RETURN dp[m][n]`,
        code: {
          cpp: `int lcs(string& s1, string& s2) {
    int m=s1.size(), n=s2.size();
    vector<vector<int>> dp(m+1, vector<int>(n+1, 0));
    for (int i=1;i<=m;i++)
        for (int j=1;j<=n;j++)
            dp[i][j] = s1[i-1]==s2[j-1] ? dp[i-1][j-1]+1
                                          : max(dp[i-1][j],dp[i][j-1]);
    return dp[m][n];
}`,
          java: `int lcs(String s1, String s2) {
    int m=s1.length(), n=s2.length();
    int[][] dp = new int[m+1][n+1];
    for (int i=1;i<=m;i++)
        for (int j=1;j<=n;j++)
            dp[i][j] = s1.charAt(i-1)==s2.charAt(j-1)
                ? dp[i-1][j-1]+1 : Math.max(dp[i-1][j],dp[i][j-1]);
    return dp[m][n];
}`,
          python: `def lcs(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]`,
        },
      },
    ],
  },

  // ───────────────────────── BST ───────────────────────────────────────────
  "bst": {
    approaches: [
      {
        label: "Recursive BST", badge: "O(log n) avg", badgeColor: "hsl(38 92% 58%)",
        description: "Standard BST with recursive insert, search, and delete.",
        pseudocode: `BST-INSERT(root, val):
  IF root == NULL: RETURN new Node(val)
  IF val < root.val: root.left  = BST-INSERT(root.left, val)
  IF val > root.val: root.right = BST-INSERT(root.right, val)
  RETURN root

BST-SEARCH(root, val):
  IF root == NULL OR root.val == val: RETURN root
  IF val < root.val: RETURN BST-SEARCH(root.left, val)
  RETURN BST-SEARCH(root.right, val)`,
        code: {
          cpp: `struct Node { int val; Node *left=nullptr, *right=nullptr; Node(int v):val(v){} };

Node* insert(Node* root, int val) {
    if (!root) return new Node(val);
    if (val < root->val) root->left  = insert(root->left, val);
    else if (val > root->val) root->right = insert(root->right, val);
    return root;
}
Node* search(Node* root, int val) {
    if (!root || root->val == val) return root;
    return val < root->val ? search(root->left, val) : search(root->right, val);
}`,
          java: `class BST {
    int val; BST left, right;
    BST(int v) { val = v; }
    
    BST insert(BST root, int v) {
        if (root == null) return new BST(v);
        if (v < root.val) root.left  = insert(root.left, v);
        else if (v > root.val) root.right = insert(root.right, v);
        return root;
    }
    BST search(BST root, int v) {
        if (root == null || root.val == v) return root;
        return v < root.val ? search(root.left, v) : search(root.right, v);
    }
}`,
          python: `class Node:
    def __init__(self, val): self.val=val; self.left=self.right=None

def insert(root, val):
    if not root: return Node(val)
    if val < root.val: root.left  = insert(root.left, val)
    elif val > root.val: root.right = insert(root.right, val)
    return root

def search(root, val):
    if not root or root.val == val: return root
    return search(root.left, val) if val < root.val else search(root.right, val)`,
        },
      },
    ],
  },


  // ─────────────────────────── HEAP SORT ──────────────────────────────────
  "heap-sort": {
    approaches: [
      {
        label: "Max-Heap Sort", badge: "O(n log n)", badgeColor: "hsl(142 71% 52%)",
        description: "Build a max-heap in O(n), then repeatedly extract the maximum and place it at the end. Guaranteed O(n log n) with O(1) space.",
        pseudocode: `HEAP-SORT(A):
  n = length(A)

  // Phase 1: Build max-heap (bottom-up)
  FOR i = floor(n/2)-1 DOWNTO 0:
    HEAPIFY(A, n, i)

  // Phase 2: Extract elements one by one
  FOR i = n-1 DOWNTO 1:
    SWAP(A[0], A[i])     // move max to end
    HEAPIFY(A, i, 0)     // restore heap on reduced array

HEAPIFY(A, size, root):
  largest = root
  left  = 2*root + 1
  right = 2*root + 2

  IF left < size AND A[left] > A[largest]:
    largest = left
  IF right < size AND A[right] > A[largest]:
    largest = right

  IF largest != root:
    SWAP(A[root], A[largest])
    HEAPIFY(A, size, largest)  // fix subtree`,
        code: {
          cpp: `void heapify(vector<int>& A, int size, int root) {
    int largest = root;
    int left  = 2 * root + 1;
    int right = 2 * root + 2;

    if (left < size && A[left] > A[largest])
        largest = left;
    if (right < size && A[right] > A[largest])
        largest = right;

    if (largest != root) {
        swap(A[root], A[largest]);
        heapify(A, size, largest);
    }
}

void heapSort(vector<int>& A) {
    int n = A.size();

    // Build max-heap
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(A, n, i);

    // Extract elements
    for (int i = n - 1; i > 0; i--) {
        swap(A[0], A[i]);
        heapify(A, i, 0);
    }
}`,
          java: `void heapify(int[] A, int size, int root) {
    int largest = root;
    int left  = 2 * root + 1;
    int right = 2 * root + 2;

    if (left < size && A[left] > A[largest])
        largest = left;
    if (right < size && A[right] > A[largest])
        largest = right;

    if (largest != root) {
        int temp = A[root];
        A[root] = A[largest];
        A[largest] = temp;
        heapify(A, size, largest);
    }
}

void heapSort(int[] A) {
    int n = A.length;
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(A, n, i);
    for (int i = n - 1; i > 0; i--) {
        int temp = A[0]; A[0] = A[i]; A[i] = temp;
        heapify(A, i, 0);
    }
}`,
          python: `def heapify(A, size, root):
    largest = root
    left  = 2 * root + 1
    right = 2 * root + 2

    if left < size and A[left] > A[largest]:
        largest = left
    if right < size and A[right] > A[largest]:
        largest = right

    if largest != root:
        A[root], A[largest] = A[largest], A[root]
        heapify(A, size, largest)

def heap_sort(A):
    n = len(A)
    # Build max-heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(A, n, i)
    # Extract elements
    for i in range(n - 1, 0, -1):
        A[0], A[i] = A[i], A[0]
        heapify(A, i, 0)
    return A`,
        },
      },
    ],
  },

  // ─────────────────────────── RADIX SORT ─────────────────────────────────
  "radix-sort": {
    approaches: [
      {
        label: "LSD Radix Sort", badge: "O(nk)", badgeColor: "hsl(199 89% 60%)",
        description: "Sort digit-by-digit from least significant to most significant. Uses Counting Sort as a stable subroutine for each digit pass.",
        pseudocode: `RADIX-SORT(A):
  maxVal = max(A)
  k = number of digits in maxVal

  FOR place = 1, 10, 100, ... (each digit position):
    COUNTING-SORT-BY-DIGIT(A, place)

COUNTING-SORT-BY-DIGIT(A, place):
  n = length(A)
  output = array of size n
  count  = array of size 10, all zeros

  // Count occurrences of each digit
  FOR i = 0 TO n-1:
    digit = (A[i] / place) % 10
    count[digit]++

  // Cumulative count
  FOR i = 1 TO 9:
    count[i] += count[i-1]

  // Build output (traverse right-to-left for stability)
  FOR i = n-1 DOWNTO 0:
    digit = (A[i] / place) % 10
    output[count[digit]-1] = A[i]
    count[digit]--

  COPY output INTO A`,
        code: {
          cpp: `void countByDigit(vector<int>& A, int place) {
    int n = A.size();
    vector<int> output(n), count(10, 0);

    for (int i = 0; i < n; i++)
        count[(A[i] / place) % 10]++;

    for (int i = 1; i < 10; i++)
        count[i] += count[i - 1];

    for (int i = n - 1; i >= 0; i--) {
        int digit = (A[i] / place) % 10;
        output[--count[digit]] = A[i];
    }
    A = output;
}

void radixSort(vector<int>& A) {
    int maxVal = *max_element(A.begin(), A.end());
    for (int place = 1; maxVal / place > 0; place *= 10)
        countByDigit(A, place);
}`,
          java: `void countByDigit(int[] A, int place) {
    int n = A.length;
    int[] output = new int[n];
    int[] count  = new int[10];

    for (int x : A) count[(x / place) % 10]++;
    for (int i = 1; i < 10; i++) count[i] += count[i - 1];

    for (int i = n - 1; i >= 0; i--) {
        int d = (A[i] / place) % 10;
        output[--count[d]] = A[i];
    }
    System.arraycopy(output, 0, A, 0, n);
}

void radixSort(int[] A) {
    int max = Arrays.stream(A).max().getAsInt();
    for (int p = 1; max / p > 0; p *= 10)
        countByDigit(A, p);
}`,
          python: `def count_by_digit(A, place):
    n = len(A)
    output = [0] * n
    count  = [0] * 10

    for x in A:
        count[(x // place) % 10] += 1
    for i in range(1, 10):
        count[i] += count[i - 1]

    for i in range(n - 1, -1, -1):
        d = (A[i] // place) % 10
        count[d] -= 1
        output[count[d]] = A[i]

    for i in range(n):
        A[i] = output[i]

def radix_sort(A):
    max_val = max(A)
    place = 1
    while max_val // place > 0:
        count_by_digit(A, place)
        place *= 10
    return A`,
        },
      },
    ],
  },

  // ─────────────────────────── COUNTING SORT ──────────────────────────────
  "counting-sort": {
    approaches: [
      {
        label: "Standard Counting Sort", badge: "O(n+k)", badgeColor: "hsl(142 71% 52%)",
        description: "Count each element's frequency, build cumulative counts, then place elements in sorted order. Linear time when k (range) is small.",
        pseudocode: `COUNTING-SORT(A, k):
  n = length(A)
  count  = array of size k+1, all zeros
  output = array of size n

  // Step 1: Count each element
  FOR i = 0 TO n-1:
    count[A[i]]++

  // Step 2: Cumulative count
  // count[i] now = number of elements <= i
  FOR i = 1 TO k:
    count[i] += count[i-1]

  // Step 3: Place elements (right-to-left for stability)
  FOR i = n-1 DOWNTO 0:
    output[count[A[i]]-1] = A[i]
    count[A[i]]--

  RETURN output`,
        code: {
          cpp: `vector<int> countingSort(vector<int>& A) {
    int k = *max_element(A.begin(), A.end());
    int n = A.size();

    vector<int> count(k + 1, 0);
    vector<int> output(n);

    // Count frequencies
    for (int x : A) count[x]++;

    // Cumulative counts
    for (int i = 1; i <= k; i++)
        count[i] += count[i - 1];

    // Build output (right-to-left for stability)
    for (int i = n - 1; i >= 0; i--) {
        output[--count[A[i]]] = A[i];
    }
    return output;
}`,
          java: `int[] countingSort(int[] A) {
    int k = Arrays.stream(A).max().getAsInt();
    int n = A.length;

    int[] count  = new int[k + 1];
    int[] output = new int[n];

    for (int x : A) count[x]++;
    for (int i = 1; i <= k; i++) count[i] += count[i - 1];

    for (int i = n - 1; i >= 0; i--)
        output[--count[A[i]]] = A[i];

    return output;
}`,
          python: `def counting_sort(A):
    k = max(A)
    n = len(A)

    count  = [0] * (k + 1)
    output = [0] * n

    for x in A:
        count[x] += 1

    for i in range(1, k + 1):
        count[i] += count[i - 1]

    for i in range(n - 1, -1, -1):
        count[A[i]] -= 1
        output[count[A[i]]] = A[i]

    return output`,
        },
      },
    ],
  },

  // ─────────────────────────── JUMP SEARCH ────────────────────────────────
  "jump-search": {
    approaches: [
      {
        label: "Jump Search", badge: "O(√n)", badgeColor: "hsl(199 89% 60%)",
        description: "Jump forward in blocks of size √n to find the block containing the target, then do a linear scan within that block. Requires sorted input.",
        pseudocode: `JUMP-SEARCH(A, target):
  n    = length(A)
  step = floor(sqrt(n))
  prev = 0

  // Phase 1: Jump forward until block may contain target
  WHILE A[min(step, n)-1] < target:
    prev = step
    step = step + floor(sqrt(n))
    IF prev >= n: RETURN -1   // not found

  // Phase 2: Linear scan in identified block
  FOR i = prev TO min(step, n)-1:
    IF A[i] == target: RETURN i

  RETURN -1   // not found`,
        code: {
          cpp: `int jumpSearch(vector<int>& A, int target) {
    int n    = A.size();
    int step = sqrt(n);
    int prev = 0;

    // Jump phase
    while (A[min(step, n) - 1] < target) {
        prev = step;
        step += sqrt(n);
        if (prev >= n) return -1;
    }

    // Linear scan phase
    for (int i = prev; i < min(step, n); i++) {
        if (A[i] == target) return i;
    }
    return -1;
}`,
          java: `int jumpSearch(int[] A, int target) {
    int n    = A.length;
    int step = (int) Math.sqrt(n);
    int prev = 0;

    while (A[Math.min(step, n) - 1] < target) {
        prev = step;
        step += (int) Math.sqrt(n);
        if (prev >= n) return -1;
    }

    for (int i = prev; i < Math.min(step, n); i++) {
        if (A[i] == target) return i;
    }
    return -1;
}`,
          python: `import math

def jump_search(A, target):
    n    = len(A)
    step = int(math.sqrt(n))
    prev = 0

    # Jump phase
    while A[min(step, n) - 1] < target:
        prev = step
        step += int(math.sqrt(n))
        if prev >= n:
            return -1

    # Linear scan phase
    for i in range(prev, min(step, n)):
        if A[i] == target:
            return i
    return -1`,
        },
      },
    ],
  },

  // ─────────────────────────── KRUSKAL'S MST ──────────────────────────────
  "kruskal": {
    approaches: [
      {
        label: "Kruskal's with Union-Find", badge: "O(E log E)", badgeColor: "hsl(142 71% 52%)",
        description: "Sort all edges by weight, then greedily add each edge to the MST if it doesn't create a cycle. Cycle detection uses Union-Find (Disjoint Set Union) with path compression and union by rank.",
        pseudocode: `KRUSKAL(graph):
  Sort all edges by weight (ascending)
  Initialize Union-Find for each vertex

  mst = []
  totalWeight = 0

  FOR each edge (u, v, weight) in sorted order:
    IF FIND(u) != FIND(v):    // different components → no cycle
      UNION(u, v)
      mst.append((u, v, weight))
      totalWeight += weight

  RETURN mst, totalWeight

FIND(x):                     // with path compression
  IF parent[x] != x:
    parent[x] = FIND(parent[x])
  RETURN parent[x]

UNION(x, y):                 // union by rank
  rx, ry = FIND(x), FIND(y)
  IF rx == ry: RETURN
  IF rank[rx] < rank[ry]: SWAP(rx, ry)
  parent[ry] = rx
  IF rank[rx] == rank[ry]: rank[rx]++`,
        code: {
          cpp: `struct Edge { int u, v, w; };

// Union-Find
int parent[105], rnk[105];

int find(int x) {
    if (parent[x] != x)
        parent[x] = find(parent[x]); // path compression
    return parent[x];
}

bool unite(int x, int y) {
    int rx = find(x), ry = find(y);
    if (rx == ry) return false;       // same component
    if (rnk[rx] < rnk[ry]) swap(rx, ry);
    parent[ry] = rx;
    if (rnk[rx] == rnk[ry]) rnk[rx]++;
    return true;
}

int kruskal(int n, vector<Edge>& edges) {
    // Initialize Union-Find
    for (int i = 0; i < n; i++) { parent[i] = i; rnk[i] = 0; }

    // Sort edges by weight
    sort(edges.begin(), edges.end(),
         [](const Edge& a, const Edge& b) { return a.w < b.w; });

    int totalWeight = 0;
    vector<Edge> mst;

    for (auto& e : edges) {
        if (unite(e.u, e.v)) {
            mst.push_back(e);
            totalWeight += e.w;
            if (mst.size() == n - 1) break; // MST complete
        }
    }
    return totalWeight;
}`,
          java: `int[] parent, rank;

int find(int x) {
    if (parent[x] != x)
        parent[x] = find(parent[x]);
    return parent[x];
}

boolean unite(int x, int y) {
    int rx = find(x), ry = find(y);
    if (rx == ry) return false;
    if (rank[rx] < rank[ry]) { int t = rx; rx = ry; ry = t; }
    parent[ry] = rx;
    if (rank[rx] == rank[ry]) rank[rx]++;
    return true;
}

int kruskal(int n, int[][] edges) {
    parent = new int[n]; rank = new int[n];
    for (int i = 0; i < n; i++) parent[i] = i;

    // Sort by weight
    Arrays.sort(edges, (a, b) -> a[2] - b[2]);

    int totalWeight = 0, edgeCount = 0;
    for (int[] e : edges) {
        if (unite(e[0], e[1])) {
            totalWeight += e[2];
            if (++edgeCount == n - 1) break;
        }
    }
    return totalWeight;
}`,
          python: `class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank   = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def unite(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True

def kruskal(n, edges):
    # edges: list of (weight, u, v)
    edges.sort()
    uf = UnionFind(n)
    mst_weight = 0
    mst_edges  = []

    for w, u, v in edges:
        if uf.unite(u, v):
            mst_weight += w
            mst_edges.append((u, v, w))
            if len(mst_edges) == n - 1:
                break

    return mst_weight, mst_edges`,
        },
      },
    ],
  },

  // ─────────────────────────── TOPOLOGICAL SORT ───────────────────────────
  "topological-sort": {
    approaches: [
      {
        label: "DFS-based (Depth First)", badge: "O(V+E)", badgeColor: "hsl(199 89% 60%)",
        description: "Run DFS on the DAG. After all descendants of a node are fully explored, push the node to the front of the result. The final order is a valid topological ordering.",
        pseudocode: `TOPOLOGICAL-SORT-DFS(graph):
  visited = empty set
  result  = empty list

  FOR each vertex v in graph:
    IF v not in visited:
      DFS(v, visited, result)

  RETURN result

DFS(v, visited, result):
  visited.add(v)

  FOR each neighbor u of v:
    IF u not in visited:
      DFS(u, visited, result)

  result.prepend(v)   // add v AFTER all descendants`,
        code: {
          cpp: `void dfs(int v, vector<vector<int>>& graph,
        vector<bool>& visited, vector<int>& result) {
    visited[v] = true;
    for (int u : graph[v]) {
        if (!visited[u])
            dfs(u, graph, visited, result);
    }
    result.push_back(v);  // add after exploring all descendants
}

vector<int> topoSort(int n, vector<vector<int>>& graph) {
    vector<bool>  visited(n, false);
    vector<int>   result;

    for (int v = 0; v < n; v++) {
        if (!visited[v])
            dfs(v, graph, visited, result);
    }

    reverse(result.begin(), result.end());
    return result;
}`,
          java: `List<Integer> result  = new ArrayList<>();
boolean[]    visited = new boolean[n];

void dfs(int v, List<List<Integer>> graph) {
    visited[v] = true;
    for (int u : graph.get(v)) {
        if (!visited[u]) dfs(u, graph);
    }
    result.add(v);
}

List<Integer> topoSort(int n, List<List<Integer>> graph) {
    for (int v = 0; v < n; v++) {
        if (!visited[v]) dfs(v, graph);
    }
    Collections.reverse(result);
    return result;
}`,
          python: `def topo_sort_dfs(graph):
    visited = set()
    result  = []

    def dfs(v):
        visited.add(v)
        for u in graph.get(v, []):
            if u not in visited:
                dfs(u)
        result.append(v)   # add AFTER all descendants

    for v in graph:
        if v not in visited:
            dfs(v)

    return list(reversed(result))`,
        },
      },
      {
        label: "Kahn's Algorithm (BFS)", badge: "O(V+E)", badgeColor: "hsl(142 71% 52%)",
        description: "Find all nodes with in-degree 0 (no prerequisites). Process them, reduce neighbors' in-degrees, and enqueue newly zero-degree nodes. Naturally detects cycles.",
        pseudocode: `KAHNS-TOPO-SORT(graph):
  Compute in-degree for all vertices
  queue = all vertices with in-degree 0
  result = []

  WHILE queue not empty:
    v = DEQUEUE(queue)
    result.append(v)

    FOR each neighbor u of v:
      in-degree[u]--
      IF in-degree[u] == 0:
        ENQUEUE(queue, u)

  IF len(result) != V: // cycle detected
    RETURN error
  RETURN result`,
        code: {
          cpp: `vector<int> kahnTopoSort(int n, vector<vector<int>>& graph) {
    vector<int> inDegree(n, 0);
    for (int v = 0; v < n; v++)
        for (int u : graph[v])
            inDegree[u]++;

    queue<int> q;
    for (int v = 0; v < n; v++)
        if (inDegree[v] == 0) q.push(v);

    vector<int> result;
    while (!q.empty()) {
        int v = q.front(); q.pop();
        result.push_back(v);
        for (int u : graph[v]) {
            if (--inDegree[u] == 0)
                q.push(u);
        }
    }

    if ((int)result.size() != n)
        return {}; // cycle detected
    return result;
}`,
          java: `List<Integer> kahnTopoSort(int n, List<List<Integer>> graph) {
    int[] inDegree = new int[n];
    for (int v = 0; v < n; v++)
        for (int u : graph.get(v))
            inDegree[u]++;

    Queue<Integer> queue = new LinkedList<>();
    for (int v = 0; v < n; v++)
        if (inDegree[v] == 0) queue.offer(v);

    List<Integer> result = new ArrayList<>();
    while (!queue.isEmpty()) {
        int v = queue.poll();
        result.add(v);
        for (int u : graph.get(v))
            if (--inDegree[u] == 0) queue.offer(u);
    }

    return result.size() == n ? result : Collections.emptyList();
}`,
          python: `from collections import deque

def kahn_topo_sort(graph, n):
    in_degree = [0] * n
    for v in range(n):
        for u in graph[v]:
            in_degree[u] += 1

    queue = deque(v for v in range(n) if in_degree[v] == 0)
    result = []

    while queue:
        v = queue.popleft()
        result.append(v)
        for u in graph[v]:
            in_degree[u] -= 1
            if in_degree[u] == 0:
                queue.append(u)

    return result if len(result) == n else []  # empty = cycle`,
        },
      },
    ],
  },

  // ─────────────────────────── TREE TRAVERSAL ─────────────────────────────
  "tree-traversal": {
    approaches: [
      {
        label: "Inorder (Left → Root → Right)", badge: "O(n)", badgeColor: "hsl(142 71% 52%)",
        description: "Visit left subtree, then root, then right subtree. For a BST this produces elements in sorted ascending order — a key property.",
        pseudocode: `INORDER(node):
  IF node == NULL: RETURN
  INORDER(node.left)    // 1. visit left subtree
  PROCESS(node.value)   // 2. process root
  INORDER(node.right)   // 3. visit right subtree

// Result for BST: sorted ascending order
// Example tree:    4
//                 / \\
//                2   6
//               / \\ / \\
//              1  3 5  7
// Inorder: 1, 2, 3, 4, 5, 6, 7`,
        code: {
          cpp: `struct Node { int val; Node *left, *right; };

// Recursive
void inorder(Node* root, vector<int>& result) {
    if (!root) return;
    inorder(root->left, result);
    result.push_back(root->val);
    inorder(root->right, result);
}

// Iterative (using stack)
vector<int> inorderIterative(Node* root) {
    vector<int> result;
    stack<Node*> st;
    Node* curr = root;

    while (curr || !st.empty()) {
        while (curr) {         // go left as far as possible
            st.push(curr);
            curr = curr->left;
        }
        curr = st.top(); st.pop();
        result.push_back(curr->val);
        curr = curr->right;    // move to right subtree
    }
    return result;
}`,
          java: `// Recursive
void inorder(TreeNode root, List<Integer> result) {
    if (root == null) return;
    inorder(root.left, result);
    result.add(root.val);
    inorder(root.right, result);
}

// Iterative
List<Integer> inorderIterative(TreeNode root) {
    List<Integer> result = new ArrayList<>();
    Deque<TreeNode> stack = new ArrayDeque<>();
    TreeNode curr = root;

    while (curr != null || !stack.isEmpty()) {
        while (curr != null) {
            stack.push(curr);
            curr = curr.left;
        }
        curr = stack.pop();
        result.add(curr.val);
        curr = curr.right;
    }
    return result;
}`,
          python: `def inorder(root, result=None):
    if result is None: result = []
    if not root: return result
    inorder(root.left, result)
    result.append(root.val)
    inorder(root.right, result)
    return result

# Iterative version
def inorder_iterative(root):
    result, stack = [], []
    curr = root
    while curr or stack:
        while curr:
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()
        result.append(curr.val)
        curr = curr.right
    return result`,
        },
      },
      {
        label: "Preorder + Postorder", badge: "O(n)", badgeColor: "hsl(199 89% 60%)",
        description: "Preorder (Root→Left→Right) is used for copying or serializing a tree. Postorder (Left→Right→Root) is used for deleting a tree or evaluating expression trees.",
        pseudocode: `PREORDER(node):           // Root first
  IF node == NULL: RETURN
  PROCESS(node.value)   // 1. process root
  PREORDER(node.left)   // 2. visit left
  PREORDER(node.right)  // 3. visit right
// Use: copy/serialize tree, prefix expression

POSTORDER(node):          // Root last
  IF node == NULL: RETURN
  POSTORDER(node.left)  // 1. visit left
  POSTORDER(node.right) // 2. visit right
  PROCESS(node.value)   // 3. process root
// Use: delete tree, evaluate postfix expression`,
        code: {
          cpp: `void preorder(Node* root, vector<int>& res) {
    if (!root) return;
    res.push_back(root->val);   // root first
    preorder(root->left,  res);
    preorder(root->right, res);
}

void postorder(Node* root, vector<int>& res) {
    if (!root) return;
    postorder(root->left,  res);
    postorder(root->right, res);
    res.push_back(root->val);   // root last
}`,
          java: `void preorder(TreeNode root, List<Integer> res) {
    if (root == null) return;
    res.add(root.val);
    preorder(root.left,  res);
    preorder(root.right, res);
}

void postorder(TreeNode root, List<Integer> res) {
    if (root == null) return;
    postorder(root.left,  res);
    postorder(root.right, res);
    res.add(root.val);
}`,
          python: `def preorder(root, result=None):
    if result is None: result = []
    if not root: return result
    result.append(root.val)          # root first
    preorder(root.left,  result)
    preorder(root.right, result)
    return result

def postorder(root, result=None):
    if result is None: result = []
    if not root: return result
    postorder(root.left,  result)
    postorder(root.right, result)
    result.append(root.val)          # root last
    return result`,
        },
      },
    ],
  },

  // ─────────────────────────── 0/1 KNAPSACK ───────────────────────────────
  "knapsack": {
    approaches: [
      {
        label: "Brute Force (Recursive)", badge: "O(2ⁿ)", badgeColor: "hsl(346 87% 65%)",
        description: "Try all 2ⁿ subsets of items. For each item either include it (if weight allows) or exclude it. Take the maximum value.",
        pseudocode: `KNAPSACK-BRUTE(items, capacity, n):
  IF n == 0 OR capacity == 0:
    RETURN 0

  // If item is too heavy, skip it
  IF items[n-1].weight > capacity:
    RETURN KNAPSACK-BRUTE(items, capacity, n-1)

  // Max of: including item n OR excluding item n
  include = items[n-1].value +
            KNAPSACK-BRUTE(items, capacity - items[n-1].weight, n-1)
  exclude = KNAPSACK-BRUTE(items, capacity, n-1)

  RETURN max(include, exclude)`,
        code: {
          cpp: `int knapsackBrute(vector<int>& w, vector<int>& v, int cap, int n) {
    if (n == 0 || cap == 0) return 0;

    if (w[n-1] > cap)  // item too heavy
        return knapsackBrute(w, v, cap, n-1);

    int include = v[n-1] + knapsackBrute(w, v, cap - w[n-1], n-1);
    int exclude = knapsackBrute(w, v, cap, n-1);

    return max(include, exclude);
}`,
          java: `int knapsackBrute(int[] w, int[] v, int cap, int n) {
    if (n == 0 || cap == 0) return 0;

    if (w[n-1] > cap)
        return knapsackBrute(w, v, cap, n-1);

    int include = v[n-1] + knapsackBrute(w, v, cap - w[n-1], n-1);
    int exclude = knapsackBrute(w, v, cap, n-1);

    return Math.max(include, exclude);
}`,
          python: `def knapsack_brute(weights, values, cap, n):
    if n == 0 or cap == 0:
        return 0

    if weights[n-1] > cap:  # item too heavy
        return knapsack_brute(weights, values, cap, n-1)

    include = values[n-1] + knapsack_brute(weights, values, cap - weights[n-1], n-1)
    exclude = knapsack_brute(weights, values, cap, n-1)

    return max(include, exclude)`,
        },
      },
      {
        label: "DP Table (Optimal)", badge: "O(nW)", badgeColor: "hsl(142 71% 52%)",
        description: "Build a 2D table dp[i][w] = max value using first i items with capacity w. Fill bottom-up, eliminating redundant recursion.",
        pseudocode: `KNAPSACK-DP(items, W):
  n = number of items
  dp[0..n][0..W] = 0    // base cases

  FOR i = 1 TO n:
    FOR w = 0 TO W:
      // Option 1: exclude item i
      dp[i][w] = dp[i-1][w]

      // Option 2: include item i (if it fits)
      IF items[i].weight <= w:
        include = items[i].value + dp[i-1][w - items[i].weight]
        dp[i][w] = max(dp[i][w], include)

  RETURN dp[n][W]

// To find WHICH items were selected:
// Trace back from dp[n][W] — if dp[i][W] != dp[i-1][W],
// item i was included.`,
        code: {
          cpp: `int knapsackDP(vector<int>& w, vector<int>& v, int W) {
    int n = w.size();
    // dp[i][cap] = max value using first i items with capacity cap
    vector<vector<int>> dp(n + 1, vector<int>(W + 1, 0));

    for (int i = 1; i <= n; i++) {
        for (int cap = 0; cap <= W; cap++) {
            dp[i][cap] = dp[i-1][cap];   // exclude item i
            if (w[i-1] <= cap) {
                int include = v[i-1] + dp[i-1][cap - w[i-1]];
                dp[i][cap] = max(dp[i][cap], include);
            }
        }
    }
    return dp[n][W];
}`,
          java: `int knapsackDP(int[] w, int[] v, int W) {
    int n = w.length;
    int[][] dp = new int[n + 1][W + 1];

    for (int i = 1; i <= n; i++) {
        for (int cap = 0; cap <= W; cap++) {
            dp[i][cap] = dp[i-1][cap];  // exclude
            if (w[i-1] <= cap) {
                int include = v[i-1] + dp[i-1][cap - w[i-1]];
                dp[i][cap] = Math.max(dp[i][cap], include);
            }
        }
    }
    return dp[n][W];
}`,
          python: `def knapsack_dp(weights, values, W):
    n = len(weights)
    # dp[i][cap] = max value using first i items with capacity cap
    dp = [[0] * (W + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for cap in range(W + 1):
            dp[i][cap] = dp[i-1][cap]  # exclude item i
            if weights[i-1] <= cap:
                include = values[i-1] + dp[i-1][cap - weights[i-1]]
                dp[i][cap] = max(dp[i][cap], include)

    return dp[n][W]`,
        },
      },
    ],
  },

  // ─────────────────────────── EDIT DISTANCE ──────────────────────────────
  "edit-distance": {
    approaches: [
      {
        label: "Brute Force (Recursive)", badge: "O(3ⁿ)", badgeColor: "hsl(346 87% 65%)",
        description: "At each position try all three operations (insert, delete, substitute) recursively and take the minimum. Exponential without memoization.",
        pseudocode: `EDIT-DIST-BRUTE(s1, s2, i, j):
  // Base cases
  IF i == 0: RETURN j    // insert j chars
  IF j == 0: RETURN i    // delete i chars

  // Characters match — no operation needed
  IF s1[i-1] == s2[j-1]:
    RETURN EDIT-DIST-BRUTE(s1, s2, i-1, j-1)

  // Try all 3 operations, take minimum
  insert  = EDIT-DIST-BRUTE(s1, s2, i,   j-1)
  delete  = EDIT-DIST-BRUTE(s1, s2, i-1, j  )
  replace = EDIT-DIST-BRUTE(s1, s2, i-1, j-1)

  RETURN 1 + min(insert, delete, replace)`,
        code: {
          cpp: `int editBrute(string& s1, string& s2, int i, int j) {
    if (i == 0) return j;
    if (j == 0) return i;

    if (s1[i-1] == s2[j-1])
        return editBrute(s1, s2, i-1, j-1);

    int ins = editBrute(s1, s2, i,   j-1);
    int del = editBrute(s1, s2, i-1, j  );
    int rep = editBrute(s1, s2, i-1, j-1);

    return 1 + min({ins, del, rep});
}`,
          java: `int editBrute(String s1, String s2, int i, int j) {
    if (i == 0) return j;
    if (j == 0) return i;

    if (s1.charAt(i-1) == s2.charAt(j-1))
        return editBrute(s1, s2, i-1, j-1);

    int ins = editBrute(s1, s2, i,   j-1);
    int del = editBrute(s1, s2, i-1, j  );
    int rep = editBrute(s1, s2, i-1, j-1);

    return 1 + Math.min(ins, Math.min(del, rep));
}`,
          python: `def edit_brute(s1, s2, i, j):
    if i == 0: return j
    if j == 0: return i

    if s1[i-1] == s2[j-1]:
        return edit_brute(s1, s2, i-1, j-1)

    ins = edit_brute(s1, s2, i,   j-1)
    delete = edit_brute(s1, s2, i-1, j  )
    rep = edit_brute(s1, s2, i-1, j-1)

    return 1 + min(ins, delete, rep)`,
        },
      },
      {
        label: "DP Table (Optimal)", badge: "O(mn)", badgeColor: "hsl(142 71% 52%)",
        description: "dp[i][j] = minimum edit operations to transform s1[0..i-1] into s2[0..j-1]. Fill bottom-up from base cases.",
        pseudocode: `EDIT-DIST-DP(s1, s2):
  m = len(s1),  n = len(s2)

  // Base cases: transforming to/from empty string
  dp[i][0] = i  for all i   // delete i chars
  dp[0][j] = j  for all j   // insert j chars

  FOR i = 1 TO m:
    FOR j = 1 TO n:
      IF s1[i-1] == s2[j-1]:
        dp[i][j] = dp[i-1][j-1]      // free — chars match
      ELSE:
        dp[i][j] = 1 + min(
          dp[i][j-1],     // insert
          dp[i-1][j],     // delete
          dp[i-1][j-1]    // substitute
        )

  RETURN dp[m][n]`,
        code: {
          cpp: `int editDistDP(string& s1, string& s2) {
    int m = s1.size(), n = s2.size();
    // dp[i][j] = edit distance between s1[0..i-1] and s2[0..j-1]
    vector<vector<int>> dp(m + 1, vector<int>(n + 1));

    // Base cases
    for (int i = 0; i <= m; i++) dp[i][0] = i;
    for (int j = 0; j <= n; j++) dp[0][j] = j;

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (s1[i-1] == s2[j-1]) {
                dp[i][j] = dp[i-1][j-1];    // no cost
            } else {
                dp[i][j] = 1 + min({
                    dp[i][j-1],    // insert
                    dp[i-1][j],    // delete
                    dp[i-1][j-1]   // substitute
                });
            }
        }
    }
    return dp[m][n];
}`,
          java: `int editDistDP(String s1, String s2) {
    int m = s1.length(), n = s2.length();
    int[][] dp = new int[m + 1][n + 1];

    for (int i = 0; i <= m; i++) dp[i][0] = i;
    for (int j = 0; j <= n; j++) dp[0][j] = j;

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (s1.charAt(i-1) == s2.charAt(j-1)) {
                dp[i][j] = dp[i-1][j-1];
            } else {
                dp[i][j] = 1 + Math.min(dp[i][j-1],
                               Math.min(dp[i-1][j], dp[i-1][j-1]));
            }
        }
    }
    return dp[m][n];
}`,
          python: `def edit_dist_dp(s1, s2):
    m, n = len(s1), len(s2)
    # dp[i][j] = edit distance for s1[:i] -> s2[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1): dp[i][0] = i   # delete all
    for j in range(n + 1): dp[0][j] = j   # insert all

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]          # match, no cost
            else:
                dp[i][j] = 1 + min(
                    dp[i][j-1],    # insert
                    dp[i-1][j],    # delete
                    dp[i-1][j-1]   # substitute
                )

    return dp[m][n]`,
        },
      },
    ],
  },
};
