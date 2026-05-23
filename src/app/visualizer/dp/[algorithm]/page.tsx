"use client";
import { useEffect, useCallback } from "react";
import { useParams } from "next/navigation";
import { TableProperties } from "lucide-react";
import { useVisualizerStore } from "@/stores/visualizerStore";
import { useAlgorithmMeta, useCurrentDPStep } from "@/hooks/useVisualizer";
import { fibonacciSteps, lcsSteps, knapsackSteps, editDistanceSteps, DEFAULT_KNAPSACK_ITEMS } from "@/algorithms/dp/index";
import { DPTableVisualizer, FibonacciTable } from "@/components/visualizers/dp/DPTableVisualizer";
import { PlaybackBar } from "@/components/controls/PlaybackBar";
import { StepDescriptionBar } from "@/components/shared/StepDescriptionBar";
import { ComplexityPanel, DescriptionPanel } from "@/components/panels/ComplexityPanel";
import { PseudocodePanel } from "@/components/panels/PseudocodePanel";
import { C } from "@/lib/utils";

const inp: React.CSSProperties = { height:32, padding:"0 10px", borderRadius:8, background:C.bg, border:`1px solid ${C.border}`, color:C.text, fontSize:12, fontFamily:"'JetBrains Mono',monospace", outline:"none", boxSizing:"border-box" as const };
const lbl: React.CSSProperties = { fontSize:10, fontWeight:700, letterSpacing:"0.1em", textTransform:"uppercase" as const, color:C.text4 };

export default function DPPage() {
  const params = useParams();
  const algoId = params.algorithm as string;
  const algo = useAlgorithmMeta(algoId);
  const { inputString1, inputString2, nValue, setInputString1, setInputString2, setNValue, setAlgorithm, setDPSteps } = useVisualizerStore();
  const currentStep = useCurrentDPStep();
  const hasSteps = useVisualizerStore(s => s.dpSteps.length > 0);

  useEffect(() => { setAlgorithm(algoId); }, [algoId, setAlgorithm]);

  const runViz = useCallback(() => {
    let steps;
    switch(algoId) {
      case "fibonacci":     steps = fibonacciSteps(nValue); break;
      case "lcs":           steps = lcsSteps(inputString1, inputString2); break;
      case "knapsack":      steps = knapsackSteps(DEFAULT_KNAPSACK_ITEMS, 8); break;
      case "edit-distance": steps = editDistanceSteps(inputString1, inputString2); break;
      default:              steps = fibonacciSteps(nValue);
    }
    setDPSteps(steps);
    setTimeout(() => useVisualizerStore.getState().play(), 80);
  }, [algoId, nValue, inputString1, inputString2, setDPSteps]);

  if (!algo) return null;
  const isFib = algoId === "fibonacci";
  const isStr = algoId === "lcs" || algoId === "edit-distance";
  const isKnap = algoId === "knapsack";
  const s1 = inputString1.slice(0,8), s2 = inputString2.slice(0,8);

  return (
    <div style={{ display:"flex", flexDirection:"column", height:"100%", background:C.bg }}>
      <div className="page-header" style={{ padding:"13px 20px", borderBottom:`1px solid ${C.border}`, background:C.bgSubtle, display:"flex", alignItems:"center", gap:11, flexShrink:0 }}>
        <div style={{ width:32, height:32, borderRadius:8, background:`${C.amber}18`, border:`1px solid ${C.amber}28`, display:"flex", alignItems:"center", justifyContent:"center", flexShrink:0 }}>
          <TableProperties size={15} color={C.amber}/>
        </div>
        <div style={{ minWidth:0 }}>
          <h1 style={{ fontSize:15, fontWeight:800, color:C.text, margin:0 }}>{algo.name}</h1>
          <p style={{ fontSize:10, color:C.text3, fontFamily:"'JetBrains Mono',monospace", margin:0 }}>
            dynamic programming · {algo.complexity.time.average}
          </p>
        </div>
      </div>
      <div style={{ flex:1, overflowY:"auto", overflowX:"hidden" }}>
        <div className="page-content">
          <div className="control-bar" style={{ background:C.surface, border:`1px solid ${C.border}` }}>
            {isFib && (
              <div style={{ display:"flex", flexDirection:"column", gap:5 }}>
                <label style={lbl}>Compute F(N) — max 15</label>
                <input type="number" min={1} max={15} value={nValue}
                  onChange={e=>setNValue(Math.min(15,Math.max(1,Number(e.target.value))))}
                  style={{ ...inp, width:90 }}/>
              </div>
            )}
            {isStr && <>
              <div style={{ display:"flex", flexDirection:"column", gap:5 }}>
                <label style={lbl}>String 1 (max 8)</label>
                <input type="text" maxLength={8} value={inputString1}
                  onChange={e=>setInputString1(e.target.value.toUpperCase())}
                  style={{ ...inp, width:130 }}/>
              </div>
              <div style={{ display:"flex", flexDirection:"column", gap:5 }}>
                <label style={lbl}>String 2 (max 8)</label>
                <input type="text" maxLength={8} value={inputString2}
                  onChange={e=>setInputString2(e.target.value.toUpperCase())}
                  style={{ ...inp, width:130 }}/>
              </div>
            </>}
            {isKnap && (
              <div style={{ fontSize:11, color:C.text3, fontFamily:"'JetBrains Mono',monospace",
                padding:"6px 10px", background:C.bgSubtle, borderRadius:7, border:`1px solid ${C.border}`,
                overflowX:"auto", maxWidth:"100%" }}>
                Items: Gem(w=2,v=6) · Book(w=3,v=4) · Tool(w=4,v=5) · Vase(w=5,v=3) · Cap=8
              </div>
            )}
            <div style={{ flex:1, minWidth:10 }}/>
            <PlaybackBar onVisualize={runViz} hasSteps={hasSteps}/>
          </div>
          <StepDescriptionBar/>
          <div className="resp-grid-2col">
            <div style={{ display:"flex", flexDirection:"column", gap:14, minWidth:0, overflow:"hidden" }}>
              <div className="dp-scroll">
                {isFib
                  ? <FibonacciTable step={currentStep} n={nValue}/>
                  : <DPTableVisualizer step={currentStep}
                      rowLabels={isStr ? ["", ...s1.split("")] : undefined}
                      colLabels={isStr ? ["", ...s2.split("")] : undefined}/>
                }
              </div>
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
