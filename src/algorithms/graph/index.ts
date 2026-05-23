import type { GraphStep, GraphData } from "@/types";

// ─── Default Graph Data ───────────────────────────────────────────────────────

export const DEFAULT_WEIGHTED_GRAPH: GraphData = {
  nodes: [
    { id: "A", label: "A", x: 100, y: 180 },
    { id: "B", label: "B", x: 260, y: 80 },
    { id: "C", label: "C", x: 260, y: 290 },
    { id: "D", label: "D", x: 430, y: 80 },
    { id: "E", label: "E", x: 560, y: 200 },
    { id: "F", label: "F", x: 430, y: 310 },
  ],
  edges: [
    { id: "AB", source: "A", target: "B", weight: 4 },
    { id: "AC", source: "A", target: "C", weight: 2 },
    { id: "BC", source: "B", target: "C", weight: 1 },
    { id: "BD", source: "B", target: "D", weight: 5 },
    { id: "CE", source: "C", target: "E", weight: 10 },
    { id: "CF", source: "C", target: "F", weight: 8 },
    { id: "DE", source: "D", target: "E", weight: 3 },
    { id: "DF", source: "D", target: "F", weight: 2 },
    { id: "EF", source: "E", target: "F", weight: 4 },
  ],
  directed: false,
  weighted: true,
};

export const DEFAULT_DAG: GraphData = {
  nodes: [
    { id: "A", label: "A", x: 80, y: 50 },
    { id: "B", label: "B", x: 240, y: 50 },
    { id: "C", label: "C", x: 160, y: 150 },
    { id: "D", label: "D", x: 320, y: 150 },
    { id: "E", label: "E", x: 240, y: 260 },
    { id: "F", label: "F", x: 80, y: 260 },
  ],
  edges: [
    { id: "AB", source: "A", target: "B", directed: true },
    { id: "AC", source: "A", target: "C", directed: true },
    { id: "BC", source: "B", target: "D", directed: true },
    { id: "CD", source: "C", target: "D", directed: true },
    { id: "CE", source: "C", target: "E", directed: true },
    { id: "DF", source: "D", target: "E", directed: true },
    { id: "CF", source: "A", target: "F", directed: true },
  ],
  directed: true,
  weighted: false,
};

// ─── Adjacency List Builder ────────────────────────────────────────────────────

function buildAdjList(
  graph: GraphData
): Map<string, Array<{ node: string; weight: number; edgeId: string }>> {
  const adj = new Map<string, Array<{ node: string; weight: number; edgeId: string }>>();
  graph.nodes.forEach((n) => adj.set(n.id, []));
  graph.edges.forEach((e) => {
    adj.get(e.source)!.push({ node: e.target, weight: e.weight ?? 1, edgeId: e.id });
    if (!graph.directed) {
      adj.get(e.target)!.push({ node: e.source, weight: e.weight ?? 1, edgeId: e.id });
    }
  });
  return adj;
}

// ─── BFS ─────────────────────────────────────────────────────────────────────

export function bfsSteps(graph: GraphData, startId: string): GraphStep[] {
  const steps: GraphStep[] = [];
  const adj = buildAdjList(graph);
  const visited = new Set<string>();
  const queue: string[] = [startId];
  const visitedNodes: string[] = [];
  const activeEdges: string[][] = [];

  visited.add(startId);

  steps.push({
    type: "info",
    visitedNodes: [],
    activeNodes: [startId],
    activeEdges: [],
    highlightedEdges: [],
    description: `BFS starting from node ${startId}. Initializing queue: [${startId}]`,
  });

  while (queue.length > 0) {
    const node = queue.shift()!;
    visitedNodes.push(node);

    steps.push({
      type: "visit",
      visitedNodes: [...visitedNodes],
      activeNodes: [node],
      activeEdges: [...activeEdges],
      highlightedEdges: [],
      description: `Dequeue ${node}. Exploring its neighbors...`,
      auxiliaryData: { queue: [...queue], visited: [...visited] },
    });

    for (const { node: neighbor, edgeId } of adj.get(node) ?? []) {
      if (!visited.has(neighbor)) {
        visited.add(neighbor);
        queue.push(neighbor);
        activeEdges.push([node, neighbor]);

        steps.push({
          type: "enqueue",
          visitedNodes: [...visitedNodes],
          activeNodes: [neighbor],
          activeEdges: [...activeEdges],
          highlightedEdges: [[node, neighbor]],
          description: `Discovered ${neighbor} via ${node}→${neighbor}. Enqueue ${neighbor}. Queue: [${queue.join(", ")}]`,
          auxiliaryData: { queue: [...queue], edgeId },
        });
      }
    }
  }

  steps.push({
    type: "info",
    visitedNodes: [...visitedNodes],
    activeNodes: [],
    activeEdges: [...activeEdges],
    highlightedEdges: [],
    description: `BFS complete! Visit order: ${visitedNodes.join(" → ")}`,
  });

  return steps;
}

// ─── DFS ─────────────────────────────────────────────────────────────────────

export function dfsSteps(graph: GraphData, startId: string): GraphStep[] {
  const steps: GraphStep[] = [];
  const adj = buildAdjList(graph);
  const visited = new Set<string>();
  const visitedNodes: string[] = [];
  const activeEdges: string[][] = [];

  steps.push({
    type: "info",
    visitedNodes: [],
    activeNodes: [startId],
    activeEdges: [],
    highlightedEdges: [],
    description: `DFS starting from node ${startId}. Using recursive exploration.`,
  });

  function dfs(node: string): void {
    visited.add(node);
    visitedNodes.push(node);

    steps.push({
      type: "visit",
      visitedNodes: [...visitedNodes],
      activeNodes: [node],
      activeEdges: [...activeEdges],
      highlightedEdges: [],
      description: `Visiting ${node}. Exploring its unvisited neighbors...`,
    });

    for (const { node: neighbor } of adj.get(node) ?? []) {
      if (!visited.has(neighbor)) {
        activeEdges.push([node, neighbor]);
        steps.push({
          type: "enqueue",
          visitedNodes: [...visitedNodes],
          activeNodes: [neighbor],
          activeEdges: [...activeEdges],
          highlightedEdges: [[node, neighbor]],
          description: `Going deep: ${node} → ${neighbor}`,
        });
        dfs(neighbor);

        steps.push({
          type: "visit",
          visitedNodes: [...visitedNodes],
          activeNodes: [node],
          activeEdges: [...activeEdges],
          highlightedEdges: [],
          description: `Backtracking to ${node}`,
        });
      }
    }
  }

  dfs(startId);

  steps.push({
    type: "info",
    visitedNodes: [...visitedNodes],
    activeNodes: [],
    activeEdges: [...activeEdges],
    highlightedEdges: [],
    description: `DFS complete! Visit order: ${visitedNodes.join(" → ")}`,
  });

  return steps;
}

// ─── Dijkstra ─────────────────────────────────────────────────────────────────

export function dijkstraSteps(graph: GraphData, startId: string): GraphStep[] {
  const steps: GraphStep[] = [];
  const adj = buildAdjList(graph);
  const dist: Record<string, number> = {};
  const visited = new Set<string>();
  const visitedNodes: string[] = [];
  const activeEdges: string[][] = [];

  graph.nodes.forEach((n) => (dist[n.id] = Infinity));
  dist[startId] = 0;

  steps.push({
    type: "info",
    visitedNodes: [],
    activeNodes: [startId],
    activeEdges: [],
    highlightedEdges: [],
    distances: { ...dist },
    description: `Dijkstra from ${startId}. dist[${startId}]=0, all others = ∞`,
  });

  const getMinUnvisited = (): string | null => {
    let minNode: string | null = null;
    let minDist = Infinity;
    for (const node of graph.nodes) {
      if (!visited.has(node.id) && dist[node.id] < minDist) {
        minDist = dist[node.id];
        minNode = node.id;
      }
    }
    return minNode;
  };

  for (let i = 0; i < graph.nodes.length; i++) {
    const u = getMinUnvisited();
    if (!u || dist[u] === Infinity) break;

    visited.add(u);
    visitedNodes.push(u);

    steps.push({
      type: "visit",
      visitedNodes: [...visitedNodes],
      activeNodes: [u],
      activeEdges: [...activeEdges],
      highlightedEdges: [],
      distances: { ...dist },
      description: `Processing ${u} (dist=${dist[u]}). Relaxing its edges...`,
    });

    for (const { node: v, weight } of adj.get(u) ?? []) {
      if (!visited.has(v)) {
        const newDist = dist[u] + weight;

        steps.push({
          type: "compare",
          visitedNodes: [...visitedNodes],
          activeNodes: [v],
          activeEdges: [...activeEdges],
          highlightedEdges: [[u, v]],
          distances: { ...dist },
          description: `Edge ${u}→${v} (w=${weight}): ${dist[u]} + ${weight} = ${newDist} ${newDist < dist[v] ? "<" : "≥"} ${dist[v] === Infinity ? "∞" : dist[v]}`,
        });

        if (newDist < dist[v]) {
          dist[v] = newDist;
          activeEdges.push([u, v]);
          steps.push({
            type: "relax",
            visitedNodes: [...visitedNodes],
            activeNodes: [v],
            activeEdges: [...activeEdges],
            highlightedEdges: [[u, v]],
            distances: { ...dist },
            description: `Relaxed! dist[${v}] updated to ${newDist}`,
          });
        }
      }
    }
  }

  steps.push({
    type: "info",
    visitedNodes: [...visitedNodes],
    activeNodes: [],
    activeEdges: [...activeEdges],
    highlightedEdges: [],
    distances: { ...dist },
    description: `Dijkstra complete! Shortest distances from ${startId}: ${Object.entries(dist).map(([k, v]) => `${k}:${v}`).join(", ")}`,
  });

  return steps;
}

// ─── Kruskal's MST ────────────────────────────────────────────────────────────

export function kruskalSteps(graph: GraphData): GraphStep[] {
  const steps: GraphStep[] = [];
  const visitedNodes: string[] = [];
  const mstEdges: string[][] = [];
  const parent: Record<string, string> = {};
  const rank: Record<string, number> = {};

  graph.nodes.forEach((n) => {
    parent[n.id] = n.id;
    rank[n.id] = 0;
  });

  function find(x: string): string {
    if (parent[x] !== x) parent[x] = find(parent[x]);
    return parent[x];
  }

  function union(x: string, y: string): boolean {
    const rx = find(x), ry = find(y);
    if (rx === ry) return false;
    if (rank[rx] > rank[ry]) parent[ry] = rx;
    else if (rank[rx] < rank[ry]) parent[rx] = ry;
    else { parent[ry] = rx; rank[rx]++; }
    return true;
  }

  const sortedEdges = [...graph.edges].sort((a, b) => (a.weight ?? 0) - (b.weight ?? 0));

  steps.push({
    type: "info",
    visitedNodes: [],
    activeNodes: [],
    activeEdges: [],
    highlightedEdges: [],
    description: `Kruskal's MST. Sorted edges: ${sortedEdges.map((e) => `${e.source}-${e.target}(${e.weight})`).join(", ")}`,
  });

  let totalWeight = 0;

  for (const edge of sortedEdges) {
    const { source, target, weight } = edge;

    steps.push({
      type: "compare",
      visitedNodes: [...visitedNodes],
      activeNodes: [source, target],
      activeEdges: [...mstEdges],
      highlightedEdges: [[source, target]],
      description: `Considering edge ${source}-${target} (w=${weight}). Do they form a cycle?`,
    });

    if (union(source, target)) {
      mstEdges.push([source, target]);
      totalWeight += weight ?? 0;
      if (!visitedNodes.includes(source)) visitedNodes.push(source);
      if (!visitedNodes.includes(target)) visitedNodes.push(target);

      steps.push({
        type: "relax",
        visitedNodes: [...visitedNodes],
        activeNodes: [source, target],
        activeEdges: [...mstEdges],
        highlightedEdges: [[source, target]],
        description: `Added ${source}-${target} to MST. Total weight so far: ${totalWeight}`,
      });
    } else {
      steps.push({
        type: "info",
        visitedNodes: [...visitedNodes],
        activeNodes: [],
        activeEdges: [...mstEdges],
        highlightedEdges: [],
        description: `Skipped ${source}-${target}: would create a cycle (${source} and ${target} already connected)`,
      });
    }
  }

  steps.push({
    type: "info",
    visitedNodes: [...visitedNodes],
    activeNodes: [],
    activeEdges: [...mstEdges],
    highlightedEdges: [...mstEdges],
    description: `MST complete! Total weight = ${totalWeight}. Edges: ${mstEdges.map((e) => e.join("-")).join(", ")}`,
  });

  return steps;
}

// ─── Topological Sort ─────────────────────────────────────────────────────────

export function topologicalSortSteps(graph: GraphData): GraphStep[] {
  const steps: GraphStep[] = [];
  const adj = buildAdjList(graph);
  const visited = new Set<string>();
  const visitedNodes: string[] = [];
  const result: string[] = [];
  const activeEdges: string[][] = [];

  steps.push({
    type: "info",
    visitedNodes: [],
    activeNodes: [],
    activeEdges: [],
    highlightedEdges: [],
    description: `Topological Sort via DFS. Valid only for DAGs (no cycles).`,
  });

  function dfs(node: string): void {
    visited.add(node);
    visitedNodes.push(node);

    steps.push({
      type: "visit",
      visitedNodes: [...visitedNodes],
      activeNodes: [node],
      activeEdges: [...activeEdges],
      highlightedEdges: [],
      description: `Visiting ${node}, exploring descendants...`,
    });

    for (const { node: neighbor } of adj.get(node) ?? []) {
      if (!visited.has(neighbor)) {
        activeEdges.push([node, neighbor]);
        steps.push({
          type: "enqueue",
          visitedNodes: [...visitedNodes],
          activeNodes: [neighbor],
          activeEdges: [...activeEdges],
          highlightedEdges: [[node, neighbor]],
          description: `Going to ${neighbor} via ${node}→${neighbor}`,
        });
        dfs(neighbor);
      }
    }

    result.unshift(node);
    steps.push({
      type: "sorted",
      visitedNodes: [...visitedNodes],
      activeNodes: [],
      activeEdges: [...activeEdges],
      highlightedEdges: [],
      description: `${node} fully explored. Push to front of result → [${result.join(", ")}]`,
      auxiliaryData: { result: [...result] },
    });
  }

  for (const node of graph.nodes) {
    if (!visited.has(node.id)) dfs(node.id);
  }

  steps.push({
    type: "info",
    visitedNodes: [...visitedNodes],
    activeNodes: [],
    activeEdges: [...activeEdges],
    highlightedEdges: [...activeEdges],
    description: `Topological Order: ${result.join(" → ")}`,
    auxiliaryData: { result },
  });

  return steps;
}

// ─── Code Implementations ─────────────────────────────────────────────────────

export const graphImplementations = {
  bfs: {
    typescript: `function bfs(graph: Map<string, string[]>, start: string): string[] {
  const visited = new Set<string>([start]);
  const queue = [start];
  const order: string[] = [];
  
  while (queue.length > 0) {
    const node = queue.shift()!;
    order.push(node);
    
    for (const neighbor of graph.get(node) ?? []) {
      if (!visited.has(neighbor)) {
        visited.add(neighbor);
        queue.push(neighbor);
      }
    }
  }
  
  return order;
}`,
    python: `from collections import deque

def bfs(graph: dict, start: str) -> list:
    visited = {start}
    queue = deque([start])
    order = []
    
    while queue:
        node = queue.popleft()
        order.append(node)
        
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return order`,
    java: `public List<String> bfs(Map<String, List<String>> graph, String start) {
    Set<String> visited = new HashSet<>();
    Queue<String> queue = new LinkedList<>();
    List<String> order = new ArrayList<>();
    
    queue.offer(start);
    visited.add(start);
    
    while (!queue.isEmpty()) {
        String node = queue.poll();
        order.add(node);
        for (String neighbor : graph.getOrDefault(node, List.of())) {
            if (!visited.contains(neighbor)) {
                visited.add(neighbor);
                queue.offer(neighbor);
            }
        }
    }
    return order;
}`,
    cpp: `vector<string> bfs(unordered_map<string, vector<string>>& graph, string start) {
    unordered_set<string> visited = {start};
    queue<string> q;
    q.push(start);
    vector<string> order;
    
    while (!q.empty()) {
        string node = q.front(); q.pop();
        order.push_back(node);
        for (auto& neighbor : graph[node]) {
            if (!visited.count(neighbor)) {
                visited.insert(neighbor);
                q.push(neighbor);
            }
        }
    }
    return order;
}`,
  },
  dfs: {
    typescript: `function dfs(graph: Map<string, string[]>, start: string): string[] {
  const visited = new Set<string>();
  const order: string[] = [];
  
  function explore(node: string): void {
    visited.add(node);
    order.push(node);
    for (const neighbor of graph.get(node) ?? []) {
      if (!visited.has(neighbor)) explore(neighbor);
    }
  }
  
  explore(start);
  return order;
}`,
    python: `def dfs(graph: dict, start: str) -> list:
    visited = set()
    order = []
    
    def explore(node):
        visited.add(node)
        order.append(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                explore(neighbor)
    
    explore(start)
    return order`,
    java: `public List<String> dfs(Map<String, List<String>> graph, String start) {
    Set<String> visited = new HashSet<>();
    List<String> order = new ArrayList<>();
    dfsHelper(graph, start, visited, order);
    return order;
}

private void dfsHelper(Map<String, List<String>> graph, String node,
    Set<String> visited, List<String> order) {
    visited.add(node);
    order.add(node);
    for (String neighbor : graph.getOrDefault(node, List.of())) {
        if (!visited.contains(neighbor)) dfsHelper(graph, neighbor, visited, order);
    }
}`,
    cpp: `vector<string> dfs(unordered_map<string, vector<string>>& graph, string start) {
    unordered_set<string> visited;
    vector<string> order;
    function<void(string)> explore = [&](string node) {
        visited.insert(node);
        order.push_back(node);
        for (auto& nb : graph[node])
            if (!visited.count(nb)) explore(nb);
    };
    explore(start);
    return order;
}`,
  },
  dijkstra: {
    typescript: `function dijkstra(graph: Map<string, [string, number][]>, start: string): Record<string, number> {
  const dist: Record<string, number> = {};
  const visited = new Set<string>();
  // Initialize distances
  for (const node of graph.keys()) dist[node] = Infinity;
  dist[start] = 0;
  
  while (true) {
    // Pick unvisited node with min distance
    let u = null, minD = Infinity;
    for (const [node, d] of Object.entries(dist)) {
      if (!visited.has(node) && d < minD) { minD = d; u = node; }
    }
    if (!u) break;
    visited.add(u);
    for (const [v, w] of graph.get(u) ?? []) {
      if (dist[u] + w < dist[v]) dist[v] = dist[u] + w;
    }
  }
  return dist;
}`,
    python: `import heapq

def dijkstra(graph: dict, start: str) -> dict:
    dist = {node: float('inf') for node in graph}
    dist[start] = 0
    heap = [(0, start)]
    
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]: continue
        for v, w in graph.get(u, []):
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(heap, (dist[v], v))
    return dist`,
    java: `// See standard Dijkstra with PriorityQueue`,
    cpp: `// See standard Dijkstra with priority_queue`,
  },
};
