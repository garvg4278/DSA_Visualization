"use client";
import { useEffect, useRef } from "react";
import * as d3 from "d3";
import { C } from "@/lib/utils";
import type { TreeStep } from "@/types";
import type { NodeLayout } from "@/algorithms/trees/index";

function nodeColor(id: string, step: TreeStep | null) {
  if (!step) return { fill: C.surfaceRaised, stroke: C.border, text: C.text2, special: false };
  if (step.activeNode === id) {
    if (step.type === "found")     return { fill:`${C.green}28`,  stroke:C.green,  text:C.green,  special:true };
    if (step.type === "not-found") return { fill:`${C.red}20`,    stroke:C.red,    text:C.red,    special:true };
    return { fill:`${C.accent}22`, stroke:C.accent, text:C.accent, special:true };
  }
  if (step.highlightedNodes.includes(id)) {
    if (step.type === "sorted") return { fill:`${C.green}15`, stroke:`${C.green}60`, text:C.green, special:true };
    return { fill:`${C.amber}12`, stroke:`${C.amber}55`, text:C.amber, special:true };
  }
  return { fill:C.surfaceRaised, stroke:C.border, text:C.text2, special:false };
}

interface TreeVisualizerProps {
  layout: NodeLayout[];
  step: TreeStep | null;
  width?: number;
  height?: number;
}

export function TreeVisualizer({ layout, step, width=700, height=360 }: TreeVisualizerProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    if (!layout.length) return;

    const map = new Map(layout.map(n => [n.id, n]));

    // Edges first (behind nodes)
    const eg = svg.append("g");
    layout.forEach(node => {
      [node.left, node.right].forEach(cid => {
        if (!cid) return;
        const child = map.get(cid);
        if (!child) return;
        eg.append("line")
          .attr("x1",node.x).attr("y1",node.y)
          .attr("x2",child.x).attr("y2",child.y)
          .attr("stroke",C.border).attr("stroke-width",1.5)
          .attr("stroke-linecap","round");
      });
    });

    // Nodes
    layout.forEach(node => {
      const { fill, stroke, text: textColor, special } = nodeColor(node.id, step);
      const g = svg.append("g").attr("transform",`translate(${node.x},${node.y})`);

      if (special) {
        g.append("circle").attr("r",26).attr("fill","none")
          .attr("stroke",stroke).attr("stroke-width",6).attr("opacity",0.2);
      }

      g.append("circle").attr("r",20).attr("fill",fill)
        .attr("stroke",stroke).attr("stroke-width",special?2.5:1.5);

      g.append("text")
        .attr("text-anchor","middle").attr("dominant-baseline","middle")
        .attr("font-size", layout.length > 14 ? 10 : 12)
        .attr("font-weight","700")
        .attr("font-family","'JetBrains Mono',monospace")
        .attr("fill",textColor)
        .text(node.value);
    });
  }, [layout, step]);

  if (!layout.length) {
    return (
      <div style={{
        height, borderRadius:12, background:C.surface, border:`1px solid ${C.border}`,
        display:"flex", alignItems:"center", justifyContent:"center",
        color:C.text4, fontSize:13, fontFamily:"'JetBrains Mono',monospace",
      }}>
        Tree is empty — insert values to begin
      </div>
    );
  }

  return (
    <div style={{
      borderRadius:12, background:C.surface, border:`1px solid ${C.border}`,
      overflow:"hidden", width:"100%",
    }}>
      <svg
        ref={svgRef}
        style={{ display:"block", width:"100%", height }}
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="xMidYMid meet"
      />
    </div>
  );
}
