"use client";
import { useEffect, useCallback } from "react";
import { useParams } from "next/navigation";
import { Network } from "lucide-react";
import { useVisualizerStore } from "@/stores/visualizerStore";
import { useAlgorithmMeta, useCurrentGraphStep } from "@/hooks/useVisualizer";
import { bfsSteps, dfsSteps, dijkstraSteps, kruskalSteps, topologicalSortSteps, DEFAULT_WEIGHTED_GRAPH, DEFAULT_DAG } from "@/algorithms/graph/index";
import { GraphVisualizer } from "@/components/visualizers/graph/GraphVisualizer";
import { PlaybackBar } from "@/components/controls/PlaybackBar";
import { StepDescriptionBar } from "@/components/shared/StepDescriptionBar";
import { ComplexityPanel, DescriptionPanel } from "@/components/panels/ComplexityPanel";
import { PseudocodePanel } from "@/components/panels/PseudocodePanel";
import { C } from "@/lib/utils";

function getSteps(id: string, graph: typeof DEFAULT_WEIGHTED_GRAPH, start: string) {
  switch(id) {
    case "bfs":              return bfsSteps(graph, start);
    case "dfs":              return dfsSteps(graph, start);
    case "dijkstra":         return dijkstraSteps(graph, start);
    case "kruskal":          return kruskalSteps(graph);
    case "topological-sort": return topologicalSortSteps(graph);
    default:                 return bfsSteps(graph, start);
  }
}

export default function GraphPage() {
  const params = useParams();
  const algoId = params.algorithm as string;
  const algo = useAlgorithmMeta(algoId);
  const { graphStartNode, setGraphStartNode, setAlgorithm, setGraphSteps } = useVisualizerStore();
  const currentStep = useCurrentGraphStep();
  const hasSteps = useVisualizerStore(s => s.graphSteps.length > 0);
  const isDAG = algoId === "topological-sort";
  const graph = isDAG ? DEFAULT_DAG : DEFAULT_WEIGHTED_GRAPH;
  const noStart = ["kruskal","topological-sort"].includes(algoId);

  useEffect(() => { setAlgorithm(algoId); }, [algoId, setAlgorithm]);

  const runViz = useCallback(() => {
    setGraphSteps(getSteps(algoId, graph, graphStartNode));
    setTimeout(() => useVisualizerStore.getState().play(), 80);
  }, [algoId, graph, graphStartNode, setGraphSteps]);

  if (!algo) return null;

  return (
    <div style={{ display:"flex", flexDirection:"column", height:"100%", background:C.bg }}>
      <div className="page-header" style={{ padding:"13px 20px", borderBottom:`1px solid ${C.border}`, background:C.bgSubtle, display:"flex", alignItems:"center", gap:11, flexShrink:0 }}>
        <div style={{ width:32, height:32, borderRadius:8, background:`${C.green}18`, border:`1px solid ${C.green}28`, display:"flex", alignItems:"center", justifyContent:"center", flexShrink:0 }}>
          <Network size={15} color={C.green}/>
        </div>
        <div style={{ minWidth:0 }}>
          <h1 style={{ fontSize:15, fontWeight:800, color:C.text, margin:0 }}>{algo.name}</h1>
          <p style={{ fontSize:10, color:C.text3, fontFamily:"'JetBrains Mono',monospace", margin:0 }}>
            graph · {algo.complexity.time.average}{isDAG?" · DAG only":algoId==="dijkstra"?" · non-negative weights":""}
          </p>
        </div>
      </div>
      <div style={{ flex:1, overflowY:"auto", overflowX:"hidden" }}>
        <div className="page-content">
          <div className="control-bar" style={{ background:C.surface, border:`1px solid ${C.border}` }}>
            {!noStart && (
              <div style={{ display:"flex", flexDirection:"column", gap:5 }}>
                <label style={{ fontSize:10, fontWeight:700, letterSpacing:"0.1em", textTransform:"uppercase", color:C.text4 }}>Start Node</label>
                <div style={{ display:"flex", gap:4 }}>
                  {graph.nodes.map(n => (
                    <button key={n.id} onClick={() => setGraphStartNode(n.id)} style={{
                      width:30, height:30, borderRadius:7, border:"none", cursor:"pointer",
                      fontSize:12, fontWeight:700,
                      background: graphStartNode===n.id ? `${C.green}18` : C.bg,
                      color: graphStartNode===n.id ? C.green : C.text3,
                      outline: graphStartNode===n.id ? `1px solid ${C.green}40` : `1px solid ${C.border}`,
                    }}>{n.id}</button>
                  ))}
                </div>
              </div>
            )}
            <div style={{ flex:1, minWidth:10 }}/>
            <PlaybackBar onVisualize={runViz} hasSteps={hasSteps}/>
          </div>
          <StepDescriptionBar/>
          <div className="resp-grid-2col">
            <div style={{ display:"flex", flexDirection:"column", gap:14, minWidth:0, overflow:"hidden" }}>
              <GraphVisualizer graph={graph} step={currentStep} height={360}/>
              <PseudocodePanel algorithmId={algoId}/>
            </div>
            <div style={{ display:"flex", flexDirection:"column", gap:14 }}>
              <ComplexityPanel algo={algo}/>
              <DescriptionPanel algo={algo}/>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
