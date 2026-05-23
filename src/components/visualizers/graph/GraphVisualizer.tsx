"use client";
import { useEffect, useRef, useMemo } from "react";
import * as d3 from "d3";
import { C } from "@/lib/utils";
import type { GraphData, GraphStep } from "@/types";

function edgeKey(a: string, b: string) { return [a,b].sort().join("--"); }

function getNodeColors(id: string, step: GraphStep | null) {
  if (!step) return { fill: C.surfaceRaised, stroke: C.border, isSpecial: false };
  if (step.activeNodes.includes(id))  return { fill: C.accent,    stroke: C.accent,  isSpecial: true };
  if (step.visitedNodes.includes(id)) return { fill: C.greenDim,  stroke: C.green,   isSpecial: true };
  return { fill: C.surfaceRaised, stroke: C.border, isSpecial: false };
}

function getEdgeStyle(src: string, tgt: string, step: GraphStep | null) {
  if (!step) return { stroke: C.border, width: 1.5 };
  const k = edgeKey(src, tgt);
  if (step.highlightedEdges.some(([a,b]) => edgeKey(a,b) === k)) return { stroke: C.accent, width: 2.5 };
  if (step.activeEdges.some(([a,b]) => edgeKey(a,b) === k))      return { stroke: C.green,  width: 2 };
  return { stroke: C.border, width: 1.5 };
}

interface GraphVisualizerProps {
  graph: GraphData;
  step: GraphStep | null;
  width?: number;
  height?: number;
}

export function GraphVisualizer({ graph, step, width=680, height=340 }: GraphVisualizerProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const posMap = useMemo(() =>
    new Map(graph.nodes.map(n => [n.id, { x: n.x, y: n.y }])),
    [graph.nodes]
  );

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    // Arrow markers
    if (graph.directed) {
      const defs = svg.append("defs");
      const mkMarker = (id: string, color: string) => {
        defs.append("marker")
          .attr("id", id).attr("markerWidth", 8).attr("markerHeight", 6)
          .attr("refX", 26).attr("refY", 3).attr("orient", "auto")
          .append("polygon").attr("points", "0 0, 8 3, 0 6").attr("fill", color);
      };
      mkMarker("arr-default", C.border);
      mkMarker("arr-active",  C.accent);
      mkMarker("arr-visited", C.green);
    }

    // Edges
    const eg = svg.append("g");
    graph.edges.forEach(edge => {
      const p1 = posMap.get(edge.source);
      const p2 = posMap.get(edge.target);
      if (!p1 || !p2) return;
      const { stroke, width: sw } = getEdgeStyle(edge.source, edge.target, step);
      const isHL  = step?.highlightedEdges.some(([a,b]) => edgeKey(a,b)===edgeKey(edge.source,edge.target));
      const isAct = step?.activeEdges.some(([a,b]) => edgeKey(a,b)===edgeKey(edge.source,edge.target));
      const markerId = isHL ? "arr-active" : isAct ? "arr-visited" : "arr-default";

      eg.append("line")
        .attr("x1",p1.x).attr("y1",p1.y).attr("x2",p2.x).attr("y2",p2.y)
        .attr("stroke",stroke).attr("stroke-width",sw)
        .attr("marker-end", graph.directed ? `url(#${markerId})` : null);

      if (graph.weighted && edge.weight !== undefined) {
        const mx=(p1.x+p2.x)/2, my=(p1.y+p2.y)/2;
        eg.append("rect")
          .attr("x",mx-11).attr("y",my-8).attr("width",22).attr("height",15)
          .attr("rx",4).attr("fill",C.bg).attr("stroke",C.borderSubtle);
        eg.append("text")
          .attr("x",mx).attr("y",my+1).attr("text-anchor","middle").attr("dominant-baseline","middle")
          .attr("font-size",10).attr("font-family","'JetBrains Mono',monospace")
          .attr("fill",C.text3).text(edge.weight);
      }
    });

    // Nodes
    graph.nodes.forEach(node => {
      const pos = posMap.get(node.id);
      if (!pos) return;
      const { fill, stroke, isSpecial } = getNodeColors(node.id, step);
      const isActive  = step?.activeNodes.includes(node.id);
      const isVisited = step?.visitedNodes.includes(node.id);

      const g = svg.append("g").attr("transform",`translate(${pos.x},${pos.y})`);

      // Glow ring
      if (isSpecial) {
        g.append("circle").attr("r",27).attr("fill","none")
          .attr("stroke", isActive ? `${C.accent}35` : `${C.green}22`)
          .attr("stroke-width",7);
      }

      // Main circle
      g.append("circle").attr("r",20)
        .attr("fill",fill).attr("stroke",stroke)
        .attr("stroke-width", isSpecial ? 2.5 : 1.5);

      // Label
      g.append("text")
        .attr("text-anchor","middle").attr("dominant-baseline","middle")
        .attr("font-size",13).attr("font-weight","700")
        .attr("font-family","'Outfit',sans-serif")
        .attr("fill", isSpecial ? "#fff" : C.text2)
        .text(node.label);

      // Distance label (Dijkstra)
      if (step?.distances) {
        const dist = step.distances[node.id];
        g.append("text")
          .attr("text-anchor","middle").attr("y",-30)
          .attr("font-size",10).attr("font-family","'JetBrains Mono',monospace")
          .attr("fill",C.green)
          .text(dist === Infinity ? "∞" : dist);
      }
    });
  }, [graph, step, posMap]);

  return (
    <div style={{
      borderRadius:12, background:C.surface, border:`1px solid ${C.border}`,
      overflow:"hidden", width:"100%",
    }}>
      <svg
        ref={svgRef}
        style={{ display:"block", width:"100%", height:height }}
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="xMidYMid meet"
      />
    </div>
  );
}
