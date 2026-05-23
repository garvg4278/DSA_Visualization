"use client";
import { useState } from "react";
import Link from "next/link";
import { Terminal, ArrowLeft, Play, Shuffle } from "lucide-react";
import { generateRandomArray, parseArrayInput } from "@/utils";
import { ALGORITHM_REGISTRY, ANIMATION_SPEEDS } from "@/constants";
import { ArrayVisualizer } from "@/components/visualizers/sorting/ArrayVisualizer";
import { Button } from "@/components/ui/primitives";
import { bubbleSortSteps } from "@/algorithms/sorting/bubbleSort";
import { selectionSortSteps } from "@/algorithms/sorting/selectionSort";
import { insertionSortSteps, mergeSortSteps, quickSortSteps, heapSortSteps, radixSortSteps, countingSortSteps } from "@/algorithms/sorting/index";
import { linearSearchSteps, binarySearchSteps, jumpSearchSteps } from "@/algorithms/searching/index";
import type { ArrayStep } from "@/types";
import { C } from "@/lib/utils";

const ALL_ARRAY = ALGORITHM_REGISTRY.filter(a => ["sorting","searching"].includes(a.category));

function getSteps(id: string, arr: number[], target: number): ArrayStep[] {
  switch(id) {
    case "bubble-sort":    return bubbleSortSteps(arr,"asc");
    case "selection-sort": return selectionSortSteps(arr,"asc");
    case "insertion-sort": return insertionSortSteps(arr,"asc");
    case "merge-sort":     return mergeSortSteps(arr,"asc");
    case "quick-sort":     return quickSortSteps(arr,"asc");
    case "heap-sort":      return heapSortSteps(arr,"asc");
    case "radix-sort":     return radixSortSteps(arr);
    case "counting-sort":  return countingSortSteps(arr);
    case "linear-search":  return linearSearchSteps(arr,target);
    case "binary-search":  return binarySearchSteps(arr,target);
    case "jump-search":    return jumpSearchSteps(arr,target);
    default: return bubbleSortSteps(arr,"asc");
  }
}

const inp: React.CSSProperties = { height:32, padding:"0 10px", borderRadius:8, background:C.bg,
  border:`1px solid ${C.border}`, color:C.text, fontSize:12,
  fontFamily:"'JetBrains Mono', monospace", outline:"none" };

export default function PlaygroundPage() {
  const [algoId, setAlgoId] = useState("bubble-sort");
  const [arr, setArr]       = useState(() => generateRandomArray(10));
  const [target, setTarget] = useState(42);
  const [speed, setSpeed]   = useState(6);
  const [steps, setSteps]   = useState<ArrayStep[]>([]);
  const [idx, setIdx]       = useState(0);
  const [running, setRunning] = useState(false);

  const algo = ALGORITHM_REGISTRY.find(a => a.id === algoId);
  const isSearch = algo?.category === "searching";
  const step = steps[idx] ?? null;

  const run = () => {
    const s = getSteps(algoId, arr, target);
    setSteps(s); setIdx(0); setRunning(true);
    const delay = ANIMATION_SPEEDS[speed as keyof typeof ANIMATION_SPEEDS] ?? 350;
    let i = 0;
    const t = setInterval(() => {
      i++; setIdx(Math.min(i, s.length-1));
      if (i >= s.length-1) { clearInterval(t); setRunning(false); }
    }, delay);
  };

  return (
    <div style={{ minHeight:"100vh", background:C.bg }}>
      <div style={{ borderBottom:`1px solid ${C.border}`, background:C.bgSubtle }}>
        <div style={{ maxWidth:900, margin:"0 auto", padding:"14px 24px",
          display:"flex", alignItems:"center", gap:12 }}>
          <Link href="/" style={{ display:"flex", alignItems:"center", gap:5, fontSize:12, color:C.text3, textDecoration:"none" }}>
            <ArrowLeft size={13}/> Home
          </Link>
          <div style={{ width:1, height:16, background:C.border }}/>
          <Terminal size={15} color={C.accent}/>
          <h1 style={{ fontSize:14, fontWeight:800, color:C.text, margin:0 }}>Playground</h1>
          <p style={{ fontSize:12, color:C.text3, margin:0 }}>Quick sandbox for any array algorithm</p>
        </div>
      </div>

      <div style={{ maxWidth:900, margin:"0 auto", padding:24, display:"flex", flexDirection:"column", gap:14 }}>
        {/* Controls */}
        <div style={{ display:"flex", flexWrap:"wrap", alignItems:"flex-end", gap:12,
          padding:"14px 16px", borderRadius:12, background:C.surface, border:`1px solid ${C.border}` }}>
          <div style={{ display:"flex", flexDirection:"column", gap:5 }}>
            <label style={{ fontSize:10, fontWeight:700, letterSpacing:"0.1em", textTransform:"uppercase", color:C.text4 }}>Algorithm</label>
            <select value={algoId} onChange={e=>setAlgoId(e.target.value)}
              style={{ ...inp, minWidth:180, appearance:"none", cursor:"pointer" }}>
              {ALL_ARRAY.map(a => <option key={a.id} value={a.id}>{a.name}</option>)}
            </select>
          </div>
          <div style={{ display:"flex", flexDirection:"column", gap:5 }}>
            <label style={{ fontSize:10, fontWeight:700, letterSpacing:"0.1em", textTransform:"uppercase", color:C.text4 }}>Array</label>
            <input type="text" value={arr.join(", ")}
              onChange={e=>{const p=parseArrayInput(e.target.value);if(p.length>0)setArr(p);}}
              style={{ ...inp, width:240 }}/>
          </div>
          {isSearch && (
            <div style={{ display:"flex", flexDirection:"column", gap:5 }}>
              <label style={{ fontSize:10, fontWeight:700, letterSpacing:"0.1em", textTransform:"uppercase", color:C.text4 }}>Target</label>
              <input type="number" value={target} onChange={e=>setTarget(Number(e.target.value))}
                style={{ ...inp, width:90 }}/>
            </div>
          )}
          <div style={{ display:"flex", flexDirection:"column", gap:5 }}>
            <label style={{ fontSize:10, fontWeight:700, letterSpacing:"0.1em", textTransform:"uppercase", color:C.text4 }}>Speed</label>
            <input type="range" min={1} max={10} value={speed} onChange={e=>setSpeed(Number(e.target.value))}
              style={{ marginTop:8, width:90, accentColor:C.accent }}/>
          </div>
          <Button variant="secondary" size="sm" onClick={()=>{setArr(generateRandomArray(10));setSteps([]);setRunning(false);}}>
            <Shuffle size={12}/> Random
          </Button>
          <Button size="sm" loading={running} onClick={run} style={{ marginLeft:"auto" }}>
            <Play size={12}/> Run
          </Button>
        </div>

        {/* Step description */}
        {step && (
          <div style={{ display:"flex", alignItems:"flex-start", gap:8, padding:"9px 12px",
            borderRadius:9, background:C.surface, border:`1px solid ${C.border}` }}>
            <span style={{ marginTop:4, width:7, height:7, borderRadius:"50%", background:C.accent, flexShrink:0 }}/>
            <span style={{ fontSize:12, fontFamily:"'JetBrains Mono', monospace", color:C.text2, flex:1 }}>
              {step.description}
            </span>
            <span style={{ fontSize:10, fontFamily:"'JetBrains Mono', monospace", color:C.text4, flexShrink:0 }}>
              {idx+1}/{steps.length}
            </span>
          </div>
        )}

        <ArrayVisualizer step={step} baseArray={arr} height={300} showValues showIndices/>

        {/* Complexity info bar */}
        {algo && (
          <div style={{ display:"flex", flexWrap:"wrap", gap:12, padding:"10px 14px",
            borderRadius:9, background:C.surface, border:`1px solid ${C.border}`, alignItems:"center" }}>
            <span style={{ fontSize:11, fontWeight:600, color:C.text2 }}>{algo.name}</span>
            <div style={{ width:1, height:14, background:C.border }}/>
            {[["Best",algo.complexity.time.best,C.green],["Avg",algo.complexity.time.average,C.amber],
              ["Worst",algo.complexity.time.worst,C.red],["Space",algo.complexity.space,C.blue]].map(([l,v,c])=>(
              <div key={String(l)} style={{ display:"flex", alignItems:"center", gap:5 }}>
                <span style={{ fontSize:10, color:C.text4, fontFamily:"monospace" }}>{l}:</span>
                <span style={{ fontSize:11, fontWeight:700, color:String(c), fontFamily:"monospace" }}>{v}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
