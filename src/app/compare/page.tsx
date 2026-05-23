"use client";
import { useState, useCallback } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { GitCompare, Play, Shuffle, ArrowLeft, Trophy } from "lucide-react";
import { generateRandomArray } from "@/utils";
import { ALGORITHM_REGISTRY, ANIMATION_SPEEDS } from "@/constants";
import { bubbleSortSteps } from "@/algorithms/sorting/bubbleSort";
import { selectionSortSteps } from "@/algorithms/sorting/selectionSort";
import { insertionSortSteps, mergeSortSteps, quickSortSteps, heapSortSteps, radixSortSteps, countingSortSteps } from "@/algorithms/sorting/index";
import { ArrayVisualizer } from "@/components/visualizers/sorting/ArrayVisualizer";
import { Button } from "@/components/ui/primitives";
import type { ArrayStep, AlgorithmMeta } from "@/types";
import { C } from "@/lib/utils";

const SORTABLE = ALGORITHM_REGISTRY.filter(a => a.category === "sorting");

function getSteps(id: string, arr: number[]): ArrayStep[] {
  switch(id) {
    case "bubble-sort":    return bubbleSortSteps(arr, "asc");
    case "selection-sort": return selectionSortSteps(arr, "asc");
    case "insertion-sort": return insertionSortSteps(arr, "asc");
    case "merge-sort":     return mergeSortSteps(arr, "asc");
    case "quick-sort":     return quickSortSteps(arr, "asc");
    case "heap-sort":      return heapSortSteps(arr, "asc");
    case "radix-sort":     return radixSortSteps(arr);
    case "counting-sort":  return countingSortSteps(arr);
    default:               return bubbleSortSteps(arr, "asc");
  }
}

const selectStyle: React.CSSProperties = {
  height: 32, padding: "0 10px", borderRadius: 8, background: C.bg,
  border: `1px solid ${C.border}`, color: C.text, fontSize: 12,
  fontFamily: "'JetBrains Mono', monospace", outline: "none",
  appearance: "none" as const, cursor: "pointer", minWidth: 160,
};

export default function ComparePage() {
  const [leftId, setLeftId]       = useState("bubble-sort");
  const [rightId, setRightId]     = useState("merge-sort");
  const [inputArr, setInputArr]   = useState(() => generateRandomArray(10));
  const [leftSteps, setLeftSteps] = useState<ArrayStep[]>([]);
  const [rightSteps, setRightSteps]= useState<ArrayStep[]>([]);
  const [leftIdx, setLeftIdx]     = useState(0);
  const [rightIdx, setRightIdx]   = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [speed, setSpeed]         = useState(6);

  const leftAlgo  = SORTABLE.find(a => a.id===leftId)!;
  const rightAlgo = SORTABLE.find(a => a.id===rightId)!;
  const hasResults = leftSteps.length > 0 && rightSteps.length > 0;

  const runComparison = useCallback(() => {
    const ls = getSteps(leftId,  inputArr);
    const rs = getSteps(rightId, inputArr);
    setLeftSteps(ls); setRightSteps(rs);
    setLeftIdx(0); setRightIdx(0); setIsRunning(true);
    const delay = ANIMATION_SPEEDS[speed as keyof typeof ANIMATION_SPEEDS] ?? 350;
    const maxLen = Math.max(ls.length, rs.length);
    let i = 0;
    const tick = setInterval(() => {
      i++;
      setLeftIdx(Math.min(i, ls.length-1));
      setRightIdx(Math.min(i, rs.length-1));
      if (i >= maxLen) { clearInterval(tick); setIsRunning(false); }
    }, delay);
  }, [leftId, rightId, inputArr, speed]);

  return (
    <div style={{ minHeight: "100vh", background: C.bg }}>
      {/* Header */}
      <div style={{ borderBottom: `1px solid ${C.border}`, background: C.bgSubtle }}>
        <div style={{ maxWidth: 1200, margin: "0 auto", padding: "14px 24px",
          display: "flex", alignItems: "center", gap: 12 }}>
          <Link href="/" style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 12,
            color: C.text3, textDecoration: "none" }}>
            <ArrowLeft size={13}/> Home
          </Link>
          <div style={{ width: 1, height: 16, background: C.border }}/>
          <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
            <GitCompare size={16} color={C.accent}/>
            <h1 style={{ fontSize: 14, fontWeight: 800, color: C.text, margin: 0 }}>Algorithm Comparison</h1>
          </div>
          <p style={{ fontSize: 12, color: C.text3, margin: 0 }}>Run two algorithms side-by-side on identical input</p>
        </div>
      </div>

      <div style={{ maxWidth: 1200, margin: "0 auto", padding: 24, display: "flex", flexDirection: "column", gap: 14 }}>
        {/* Controls */}
        <div style={{ display: "flex", flexWrap: "wrap", alignItems: "flex-end", gap: 14,
          padding: "14px 16px", borderRadius: 12, background: C.surface, border: `1px solid ${C.border}` }}>
          {/* Left picker */}
          <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
            <label style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase",
              color: C.accent }}>Left Algorithm</label>
            <select value={leftId} onChange={e=>setLeftId(e.target.value)}
              style={{ ...selectStyle, borderColor: `${C.accent}40` }}>
              {SORTABLE.map(a => <option key={a.id} value={a.id}>{a.name}</option>)}
            </select>
          </div>
          <div style={{ fontSize: 14, fontWeight: 800, color: C.text4, paddingBottom: 6 }}>VS</div>
          {/* Right picker */}
          <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
            <label style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase",
              color: C.blue }}>Right Algorithm</label>
            <select value={rightId} onChange={e=>setRightId(e.target.value)}
              style={{ ...selectStyle, borderColor: `${C.blue}40` }}>
              {SORTABLE.map(a => <option key={a.id} value={a.id}>{a.name}</option>)}
            </select>
          </div>
          {/* Array */}
          <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
            <label style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase", color: C.text4 }}>Array</label>
            <input type="text" value={inputArr.join(", ")}
              onChange={e=>{const p=e.target.value.split(",").map(s=>parseInt(s.trim())).filter(n=>!isNaN(n));if(p.length>0)setInputArr(p);}}
              style={{ ...selectStyle, width: 220 }}/>
          </div>
          {/* Speed */}
          <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
            <label style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase", color: C.text4 }}>Speed ({speed}x)</label>
            <input type="range" min={1} max={10} value={speed} onChange={e=>setSpeed(Number(e.target.value))}
              style={{ marginTop: 6, width: 100, accentColor: C.accent, cursor: "pointer" }}/>
          </div>
          <Button variant="secondary" size="sm" onClick={()=>{setInputArr(generateRandomArray(10));setLeftSteps([]);setRightSteps([]);setIsRunning(false);}}>
            <Shuffle size={12}/> Random
          </Button>
          <Button size="sm" loading={isRunning} onClick={runComparison} style={{ marginLeft: "auto" }}>
            <Play size={12}/> {isRunning ? "Running..." : "Compare"}
          </Button>
        </div>

        {/* Visualizers */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
          {[
            { label: leftAlgo.name, steps: leftSteps, idx: leftIdx, color: C.accent },
            { label: rightAlgo.name, steps: rightSteps, idx: rightIdx, color: C.blue },
          ].map(({ label, steps, idx, color }) => (
            <div key={label} style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
                <div style={{ width: 7, height: 7, borderRadius: "50%", background: color }}/>
                <span style={{ fontSize: 12, fontWeight: 700, color }}>{label}</span>
                {hasResults && <span style={{ marginLeft: "auto", fontSize: 10,
                  fontFamily: "'JetBrains Mono', monospace", color: C.text4 }}>
                  {Math.min(idx+1, steps.length)} / {steps.length} steps
                </span>}
              </div>
              <ArrayVisualizer step={steps[idx]??null} baseArray={inputArr} showValues height={220}/>
            </div>
          ))}
        </div>

        {/* Step descriptions */}
        {hasResults && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
            {[{s:leftSteps[leftIdx],c:C.accent},{s:rightSteps[rightIdx],c:C.blue}].map(({s,c},i)=>(
              <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: 8, padding: "8px 12px",
                borderRadius: 9, background: C.surface, border: `1px solid ${C.border}`, minHeight: 38 }}>
                <span style={{ marginTop:4, width:6, height:6, borderRadius:"50%", background:c, flexShrink:0 }}/>
                <span style={{ fontSize:11, fontFamily:"'JetBrains Mono', monospace", color:C.text2 }}>{s?.description??"—"}</span>
              </div>
            ))}
          </div>
        )}

        {/* Results */}
        {hasResults && !isRunning && (
          <motion.div initial={{ opacity:0, y:8 }} animate={{ opacity:1, y:0 }}
            style={{ borderRadius: 12, background: C.surface, border: `1px solid ${C.border}`, padding: 18 }}>
            <p style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase",
              color: C.text4, marginBottom: 14 }}>Comparison Results</p>
            <div style={{ display: "grid", gridTemplateColumns: "1fr auto 1fr", gap: 16, alignItems: "center" }}>
              {[{algo:leftAlgo,steps:leftSteps.length,other:rightSteps.length,color:C.accent},
                {algo:rightAlgo,steps:rightSteps.length,other:leftSteps.length,color:C.blue}].map(({algo,steps,other,color},i)=>{
                const wins = steps < other;
                return (
                  <div key={i} style={{ borderRadius: 10, border: `1px solid ${wins?`${C.green}40`:C.border}`,
                    background: wins?`${C.green}05`:C.surfaceRaised, padding: "14px 16px", textAlign:"center" }}>
                    <p style={{ fontSize:13, fontWeight:700, color, marginBottom:6 }}>{algo.name}</p>
                    <p style={{ fontSize:30, fontWeight:900, fontFamily:"'JetBrains Mono', monospace",
                      color:wins?C.green:C.text2, margin:"0 0 4px" }}>{steps}</p>
                    <p style={{ fontSize:10, color:C.text4, marginBottom:8 }}>total steps</p>
                    {wins && <div style={{ display:"flex", alignItems:"center", justifyContent:"center", gap:4,
                      fontSize:11, fontWeight:600, color:C.green }}>
                      <Trophy size={12}/> Fewer steps!
                    </div>}
                    <div style={{ marginTop:10, paddingTop:10, borderTop:`1px solid ${C.borderSubtle}`,
                      display:"flex", flexDirection:"column", gap:3 }}>
                      {[["avg",algo.complexity.time.average,C.amber],["space",algo.complexity.space,C.blue]].map(([l,v,c])=>(
                        <div key={String(l)} style={{ display:"flex", justifyContent:"space-between", alignItems:"center" }}>
                          <span style={{ fontSize:10, color:C.text4, fontFamily:"monospace" }}>{l}:</span>
                          <span style={{ fontSize:10, fontWeight:700, color:String(c), fontFamily:"monospace" }}>{v}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })}
              <div style={{ textAlign:"center", fontSize:16, fontWeight:800, color:C.text4 }}>VS</div>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
}
