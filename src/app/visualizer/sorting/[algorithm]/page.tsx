"use client";
import { useEffect, useCallback } from "react";
import { useParams } from "next/navigation";
import { ArrowUpDown, Shuffle } from "lucide-react";
import { useVisualizerStore } from "@/stores/visualizerStore";
import { useAlgorithmMeta, useCurrentArrayStep } from "@/hooks/useVisualizer";
import { generateRandomArray, parseArrayInput } from "@/utils";
import { bubbleSortSteps } from "@/algorithms/sorting/bubbleSort";
import { selectionSortSteps } from "@/algorithms/sorting/selectionSort";
import { insertionSortSteps, mergeSortSteps, quickSortSteps, heapSortSteps, radixSortSteps, countingSortSteps } from "@/algorithms/sorting/index";
import { ArrayVisualizer } from "@/components/visualizers/sorting/ArrayVisualizer";
import { PlaybackBar } from "@/components/controls/PlaybackBar";
import { StepDescriptionBar } from "@/components/shared/StepDescriptionBar";
import { ComplexityPanel, DescriptionPanel } from "@/components/panels/ComplexityPanel";
import { PseudocodePanel } from "@/components/panels/PseudocodePanel";
import { Button } from "@/components/ui/primitives";
import { C } from "@/lib/utils";

function getSteps(id: string, arr: number[], order: "asc"|"desc") {
  switch(id) {
    case "bubble-sort":    return bubbleSortSteps(arr, order);
    case "selection-sort": return selectionSortSteps(arr, order);
    case "insertion-sort": return insertionSortSteps(arr, order);
    case "merge-sort":     return mergeSortSteps(arr, order);
    case "quick-sort":     return quickSortSteps(arr, order);
    case "heap-sort":      return heapSortSteps(arr, order);
    case "radix-sort":     return radixSortSteps(arr);
    case "counting-sort":  return countingSortSteps(arr);
    default:               return bubbleSortSteps(arr, order);
  }
}

const inp: React.CSSProperties = {
  height:32, padding:"0 10px", borderRadius:8, background:C.bg,
  border:`1px solid ${C.border}`, color:C.text, fontSize:12,
  fontFamily:"'JetBrains Mono',monospace", outline:"none", boxSizing:"border-box",
};
const lbl: React.CSSProperties = {
  fontSize:10, fontWeight:700, letterSpacing:"0.1em",
  textTransform:"uppercase" as const, color:C.text4,
};

export default function SortingPage() {
  const params = useParams();
  const algoId = params.algorithm as string;
  const algo = useAlgorithmMeta(algoId);
  const { inputArray, sortOrder, setInputArray, setSortOrder, setAlgorithm, setArraySteps, reset } = useVisualizerStore();
  const currentStep = useCurrentArrayStep();
  const hasSteps = useVisualizerStore(s => s.arraySteps.length > 0);

  useEffect(() => { setAlgorithm(algoId); }, [algoId, setAlgorithm]);

  const runViz = useCallback(() => {
    setArraySteps(getSteps(algoId, inputArray, sortOrder));
    setTimeout(() => useVisualizerStore.getState().play(), 80);
  }, [algoId, inputArray, sortOrder, setArraySteps]);

  if (!algo) return <div style={{ padding:40, color:C.text3 }}>Algorithm not found: {algoId}</div>;

  return (
    <div style={{ display:"flex", flexDirection:"column", height:"100%", background:C.bg }}>
      {/* Header */}
      <div className="page-header" style={{ padding:"13px 20px", borderBottom:`1px solid ${C.border}`,
        background:C.bgSubtle, display:"flex", alignItems:"center", gap:11, flexShrink:0 }}>
        <div style={{ width:32, height:32, borderRadius:8, background:`${C.accent}18`,
          border:`1px solid ${C.accent}28`, display:"flex", alignItems:"center", justifyContent:"center", flexShrink:0 }}>
          <ArrowUpDown size={15} color={C.accent}/>
        </div>
        <div style={{ minWidth:0 }}>
          <h1 style={{ fontSize:15, fontWeight:800, color:C.text, margin:0, whiteSpace:"nowrap", overflow:"hidden", textOverflow:"ellipsis" }}>{algo.name}</h1>
          <p style={{ fontSize:10, color:C.text3, fontFamily:"'JetBrains Mono',monospace", margin:0 }}>
            sorting · {algo.complexity.time.average} avg
          </p>
        </div>
      </div>

      <div style={{ flex:1, overflowY:"auto", overflowX:"hidden" }}>
        <div className="page-content">
          {/* Controls */}
          <div className="control-bar" style={{ background:C.surface, border:`1px solid ${C.border}` }}>
            <div style={{ display:"flex", flexDirection:"column", gap:5 }}>
              <label style={lbl}>Array</label>
              <input type="text" value={inputArray.join(", ")}
                onChange={e => { const p = parseArrayInput(e.target.value); if(p.length>0) setInputArray(p); }}
                style={{ ...inp, width:250 }} placeholder="e.g. 64, 34, 25, 12, 22"/>
            </div>
            <div style={{ display:"flex", flexDirection:"column", gap:5 }}>
              <label style={lbl}>Order</label>
              <div style={{ display:"flex", gap:4 }}>
                {(["asc","desc"] as const).map(o => (
                  <button key={o} onClick={() => setSortOrder(o)} style={{
                    height:32, padding:"0 11px", borderRadius:7, border:"none", cursor:"pointer",
                    fontSize:11, fontWeight:600,
                    background: sortOrder===o ? `${C.accent}18` : C.bg,
                    color: sortOrder===o ? C.accent : C.text3,
                    outline: sortOrder===o ? `1px solid ${C.accent}40` : `1px solid ${C.border}`,
                  }}>{o==="asc"?"Asc ↑":"Desc ↓"}</button>
                ))}
              </div>
            </div>
            <Button variant="secondary" size="sm" onClick={() => { setInputArray(generateRandomArray(12)); reset(); }}>
              <Shuffle size={12}/> Random
            </Button>
            <div style={{ flex:1, minWidth:10 }}/>
            <PlaybackBar onVisualize={runViz} hasSteps={hasSteps}/>
          </div>

          <StepDescriptionBar/>

          {/* Responsive 2-col grid */}
          <div className="resp-grid-2col">
            <div style={{ display:"flex", flexDirection:"column", gap:14, minWidth:0 }}>
              <ArrayVisualizer step={currentStep} baseArray={inputArray} showValues height={300}/>
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
