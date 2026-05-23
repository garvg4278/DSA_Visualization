import type { TreeNode, TreeStep } from "@/types";

// ─── BST Construction ─────────────────────────────────────────────────────────

export function insertBSTNode(root: TreeNode | null, value: number): TreeNode {
  if (!root) {
    return { id: `node-${value}-${Date.now()}`, value, left: null, right: null };
  }
  if (value < root.value) {
    return { ...root, left: insertBSTNode(root.left, value) };
  } else if (value > root.value) {
    return { ...root, right: insertBSTNode(root.right, value) };
  }
  return root; // duplicate
}

export function buildBST(values: number[]): TreeNode | null {
  let root: TreeNode | null = null;
  for (const v of values) root = insertBSTNode(root, v);
  return root;
}

export function treeToArray(root: TreeNode | null): Array<{ id: string; value: number; left: string | null; right: string | null; parent: string | null }> {
  const result: Array<{ id: string; value: number; left: string | null; right: string | null; parent: string | null }> = [];

  function traverse(node: TreeNode | null, parentId: string | null): void {
    if (!node) return;
    result.push({
      id: node.id,
      value: node.value,
      left: node.left?.id ?? null,
      right: node.right?.id ?? null,
      parent: parentId,
    });
    traverse(node.left, node.id);
    traverse(node.right, node.id);
  }

  traverse(root, null);
  return result;
}

// ─── BST Insert Steps ─────────────────────────────────────────────────────────

export function bstInsertSteps(root: TreeNode | null, value: number): TreeStep[] {
  const steps: TreeStep[] = [];

  if (!root) {
    steps.push({
      type: "info",
      highlightedNodes: [],
      activeNode: null,
      description: `Tree is empty. Creating root node with value ${value}.`,
    });
    return steps;
  }

  steps.push({
    type: "info",
    highlightedNodes: [],
    activeNode: root.id,
    description: `Inserting ${value} into BST. Starting at root ${root.value}.`,
  });

  let current: TreeNode | null = root;
  const path: string[] = [];

  while (current) {
    path.push(current.id);
    steps.push({
      type: "compare",
      highlightedNodes: [...path],
      activeNode: current.id,
      description: `At node ${current.value}: ${value} ${value < current.value ? "<" : ">"} ${current.value} → go ${value < current.value ? "left" : "right"}`,
    });

    if (value < current.value) {
      if (!current.left) {
        steps.push({
          type: "insert",
          highlightedNodes: [...path],
          activeNode: current.id,
          description: `Left child of ${current.value} is null → insert ${value} here!`,
          auxiliaryData: { insertedValue: value, parentId: current.id, side: "left" },
        });
        break;
      }
      current = current.left;
    } else if (value > current.value) {
      if (!current.right) {
        steps.push({
          type: "insert",
          highlightedNodes: [...path],
          activeNode: current.id,
          description: `Right child of ${current.value} is null → insert ${value} here!`,
          auxiliaryData: { insertedValue: value, parentId: current.id, side: "right" },
        });
        break;
      }
      current = current.right;
    } else {
      steps.push({
        type: "found",
        highlightedNodes: [...path],
        activeNode: current.id,
        description: `${value} already exists in BST. No duplicate insertion.`,
      });
      break;
    }
  }

  return steps;
}

// ─── BST Search Steps ─────────────────────────────────────────────────────────

export function bstSearchSteps(root: TreeNode | null, value: number): TreeStep[] {
  const steps: TreeStep[] = [];

  if (!root) {
    steps.push({
      type: "not-found",
      highlightedNodes: [],
      activeNode: null,
      description: "Tree is empty.",
    });
    return steps;
  }

  steps.push({
    type: "info",
    highlightedNodes: [],
    activeNode: root.id,
    description: `Searching for ${value} in BST.`,
  });

  let current: TreeNode | null = root;
  const path: string[] = [];

  while (current) {
    path.push(current.id);
    steps.push({
      type: "compare",
      highlightedNodes: [...path],
      activeNode: current.id,
      description: `Checking node ${current.value}: ${value} ${value === current.value ? "===" : value < current.value ? "<" : ">"} ${current.value}`,
    });

    if (value === current.value) {
      steps.push({
        type: "found",
        highlightedNodes: [...path],
        activeNode: current.id,
        description: `✓ Found ${value} at depth ${path.length - 1}!`,
        auxiliaryData: { foundId: current.id, depth: path.length - 1 },
      });
      return steps;
    } else if (value < current.value) {
      current = current.left;
      if (!current) {
        steps.push({
          type: "not-found",
          highlightedNodes: [...path],
          activeNode: null,
          description: `Left child is null. ${value} not in BST.`,
        });
      }
    } else {
      current = current.right;
      if (!current) {
        steps.push({
          type: "not-found",
          highlightedNodes: [...path],
          activeNode: null,
          description: `Right child is null. ${value} not in BST.`,
        });
      }
    }
  }

  return steps;
}

// ─── Tree Traversal Steps ─────────────────────────────────────────────────────

export type TraversalType = "inorder" | "preorder" | "postorder";

export function treeTraversalSteps(
  root: TreeNode | null,
  type: TraversalType
): TreeStep[] {
  const steps: TreeStep[] = [];
  const visited: string[] = [];
  const result: number[] = [];

  if (!root) {
    steps.push({
      type: "info",
      highlightedNodes: [],
      activeNode: null,
      description: "Tree is empty.",
    });
    return steps;
  }

  const typeLabel = { inorder: "Inorder (L→Root→R)", preorder: "Preorder (Root→L→R)", postorder: "Postorder (L→R→Root)" }[type];

  steps.push({
    type: "info",
    highlightedNodes: [],
    activeNode: root.id,
    description: `Starting ${typeLabel} traversal.`,
    auxiliaryData: { result: [] },
  });

  function inorder(node: TreeNode | null): void {
    if (!node) return;
    steps.push({
      type: "visit",
      highlightedNodes: [...visited],
      activeNode: node.id,
      description: `At ${node.value}: recurse left`,
    });
    inorder(node.left);
    visited.push(node.id);
    result.push(node.value);
    steps.push({
      type: "visit",
      highlightedNodes: [...visited],
      activeNode: node.id,
      description: `Visit ${node.value} (inorder). Result: [${result.join(", ")}]`,
      auxiliaryData: { result: [...result] },
    });
    inorder(node.right);
  }

  function preorder(node: TreeNode | null): void {
    if (!node) return;
    visited.push(node.id);
    result.push(node.value);
    steps.push({
      type: "visit",
      highlightedNodes: [...visited],
      activeNode: node.id,
      description: `Visit ${node.value} (preorder). Result: [${result.join(", ")}]`,
      auxiliaryData: { result: [...result] },
    });
    preorder(node.left);
    preorder(node.right);
  }

  function postorder(node: TreeNode | null): void {
    if (!node) return;
    steps.push({
      type: "visit",
      highlightedNodes: [...visited],
      activeNode: node.id,
      description: `At ${node.value}: recurse left then right`,
    });
    postorder(node.left);
    postorder(node.right);
    visited.push(node.id);
    result.push(node.value);
    steps.push({
      type: "visit",
      highlightedNodes: [...visited],
      activeNode: node.id,
      description: `Visit ${node.value} (postorder). Result: [${result.join(", ")}]`,
      auxiliaryData: { result: [...result] },
    });
  }

  if (type === "inorder") inorder(root);
  else if (type === "preorder") preorder(root);
  else postorder(root);

  steps.push({
    type: "sorted",
    highlightedNodes: [...visited],
    activeNode: null,
    description: `${typeLabel} complete: [${result.join(", ")}]${type === "inorder" ? " — Note: Inorder of BST = sorted order!" : ""}`,
    auxiliaryData: { result },
  });

  return steps;
}

// ─── Layout Computation ───────────────────────────────────────────────────────

export interface NodeLayout {
  id: string;
  value: number;
  x: number;
  y: number;
  left: string | null;
  right: string | null;
}

export function computeTreeLayout(
  root: TreeNode | null,
  width: number = 700
): NodeLayout[] {
  const layouts: NodeLayout[] = [];

  function compute(
    node: TreeNode | null,
    x: number,
    y: number,
    spread: number
  ): void {
    if (!node) return;
    layouts.push({
      id: node.id,
      value: node.value,
      x,
      y,
      left: node.left?.id ?? null,
      right: node.right?.id ?? null,
    });
    compute(node.left, x - spread, y + 70, spread / 2);
    compute(node.right, x + spread, y + 70, spread / 2);
  }

  compute(root, width / 2, 40, width / 4);
  return layouts;
}

// ─── Code Implementations ─────────────────────────────────────────────────────

export const treeImplementations = {
  bst: {
    typescript: `class BST {
  root: TreeNode | null = null;
  
  insert(value: number): void {
    this.root = this._insert(this.root, value);
  }
  
  private _insert(node: TreeNode | null, value: number): TreeNode {
    if (!node) return { value, left: null, right: null };
    if (value < node.value) node.left = this._insert(node.left, value);
    else if (value > node.value) node.right = this._insert(node.right, value);
    return node;
  }
  
  search(value: number): boolean {
    let curr = this.root;
    while (curr) {
      if (value === curr.value) return true;
      curr = value < curr.value ? curr.left : curr.right;
    }
    return false;
  }
}`,
    python: `class BST:
    def insert(self, root, value):
        if not root:
            return TreeNode(value)
        if value < root.val:
            root.left = self.insert(root.left, value)
        elif value > root.val:
            root.right = self.insert(root.right, value)
        return root
    
    def search(self, root, value):
        while root:
            if value == root.val: return True
            root = root.left if value < root.val else root.right
        return False`,
    java: `// Standard BST implementation`,
    cpp: `// Standard BST implementation`,
  },
};
