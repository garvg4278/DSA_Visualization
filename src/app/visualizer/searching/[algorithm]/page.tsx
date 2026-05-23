"use client";
import { useEffect, useCallback } from "react";
import { useParams } from "next/navigation";
import { Search, Shuffle } from "lucide-react";
import { useVisualizerStore } from "@/stores/visualizerStore";
import { useAlgorithmMeta, useCurrentArrayStep } from "@/hooks/useVisualizer";
import { generateRandomArray } from "@/utils";
import { linearSearchSteps, binarySearchSteps, jumpSearchSteps } from "@/algorithms/searching/index";
import { ArrayVisualizer } from "@/components/visualizers/sorting/ArrayVisualizer";
import { PlaybackBar } from "@/components/controls/PlaybackBar";
import { StepDescriptionBar } from "@/components/shared/StepDescriptionBar";
import { ComplexityPanel, DescriptionPanel } from "@/components/panels/ComplexityPanel";
import { PseudocodePanel } from "@/components/panels/PseudocodePanel";
import { Button } from "@/components/ui/primitives";
import { C } from "@/lib/utils";

function getSteps(id: string, arr: number[], target: number) {
  switch(id) {
    case "linear-search": return linearSearchSteps(arr, target);
    case "binary-search": return binarySearchSteps(arr, target);
    case "jump-search":   return jumpSearchSteps(arr, target);
    default: return linearSearchSteps(arr, target);
  }
}
const inp: React.CSSProperties = { height:32, padding:"0 10px", borderRadius:8, background:"hsl(222 18% 7%)", border:`1px solid hsl(220 12% 20%)`, color:"hsl(220 20% 94%)", fontSize:12, fontFamily:"'JetBrains Mono',monospace", outline:"none", boxSizing:"border-box" as const };
const lbl: React.CSSProperties = { fontSize:10, fontWeight:700, letterSpacing:"0.1em", textTransform:"uppercase" as const, color:"hsl(220 8% 28%)" };

export default function SearchingPage() {
  const params = useParams();
  const algoId = params.algorithm as string;
  const algo = useAlgorithmMeta(algoId);
  const { inputArray, searchTarget, setInputArray, setSearchTarget, setAlgorithm, setArraySteps, reset } = useVisualizerStore();
  const currentStep = useCurrentArrayStep();
  const hasSteps = useVisualizerStore(s => s.arraySteps.length > 0);
  const isSorted = algoId !== "linear-search";
  const displayArr = isSorted ? [...inputArray].sort((a,b)=>a-b) : inputArray;

  useEffect(() => { setAlgorithm(algoId); }, [algoId, setAlgorithm]);

  const runViz = useCallback(() => {
    setArraySteps(getSteps(algoId, inputArray, searchTarget));
    setTimeout(() => useVisualizerStore.getState().play(), 80);
  }, [algoId, inputArray, searchTarget, setArraySteps]);

  const randomize = useCallback(() => {
    const arr = generateRandomArray(14,5,80).sort((a,b)=>a-b);
    setInputArray(arr); setSearchTarget(arr[Math.floor(Math.random()*arr.length)]); reset();
  }, [setInputArray, setSearchTarget, reset]);

  if (!algo) return null;

  return (
    <div style={{ display:"flex", flexDirection:"column", height:"100%", background:C.bg }}>
      <div className="page-header" style={{ padding:"13px 20px", borderBottom:`1px solid ${C.border}`, background:C.bgSubtle, display:"flex", alignItems:"center", gap:11, flexShrink:0 }}>
        <div style={{ width:32, height:32, borderRadius:8, background:`${C.blue}18`, border:`1px solid ${C.blue}28`, display:"flex", alignItems:"center", justifyContent:"center", flexShrink:0 }}>
          <Search size={15} color={C.blue}/>
        </div>
        <div style={{ minWidth:0 }}>
          <h1 style={{ fontSize:15, fontWeight:800, color:C.text, margin:0 }}>{algo.name}</h1>
          <p style={{ fontSize:10, color:C.text3, fontFamily:"'JetBrains Mono',monospace", margin:0 }}>
            searching · {algo.complexity.time.average}{isSorted?" · sorted input required":""}
          </p>
        </div>
      </div>
      <div style={{ flex:1, overflowY:"auto", overflowX:"hidden" }}>
        <div className="page-content">
          <div className="control-bar" style={{ background:C.surface, border:`1px solid ${C.border}` }}>
            <div style={{ display:"flex", flexDirection:"column", gap:5 }}>
              <label style={lbl}>Array{isSorted?" (auto-sorted)":""}</label>
              <input type="text" value={inputArray.join(", ")}
                onChange={e=>{const p=e.target.value.split(",").map(s=>parseInt(s.trim())).filter(n=>!isNaN(n));if(p.length>0)setInputArray(p);}}
                style={{ ...inp, width:220 }}/>
            </div>
            <div style={{ display:"flex", flexDirection:"column", gap:5 }}>
              <label style={lbl}>Target</label>
              <input type="number" value={searchTarget} onChange={e=>setSearchTarget(Number(e.target.value))} style={{ ...inp, width:90 }}/>
            </div>
            <Button variant="secondary" size="sm" onClick={randomize}><Shuffle size={12}/> Random</Button>
            <div style={{ flex:1, minWidth:10 }}/>
            <PlaybackBar onVisualize={runViz} hasSteps={hasSteps}/>
          </div>
          <StepDescriptionBar/>
          <div className="resp-grid-2col">
            <div style={{ display:"flex", flexDirection:"column", gap:14, minWidth:0 }}>
              <ArrayVisualizer step={currentStep} baseArray={displayArr} showValues showIndices height={280}/>
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
