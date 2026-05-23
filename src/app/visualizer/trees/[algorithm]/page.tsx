"use client";
import { useEffect, useCallback, useState } from "react";
import { useParams } from "next/navigation";
import { GitBranch, Plus, Search, RotateCcw } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useVisualizerStore } from "@/stores/visualizerStore";
import { useAlgorithmMeta, useCurrentTreeStep } from "@/hooks/useVisualizer";
import { buildBST, insertBSTNode, bstInsertSteps, bstSearchSteps, treeTraversalSteps, computeTreeLayout, type TraversalType } from "@/algorithms/trees/index";
import type { TreeNode } from "@/types";
import { TreeVisualizer } from "@/components/visualizers/trees/TreeVisualizer";
import { PlaybackBar } from "@/components/controls/PlaybackBar";
import { StepDescriptionBar } from "@/components/shared/StepDescriptionBar";
import { ComplexityPanel, DescriptionPanel } from "@/components/panels/ComplexityPanel";
import { PseudocodePanel } from "@/components/panels/PseudocodePanel";
import { Button } from "@/components/ui/primitives";
import { C } from "@/lib/utils";

const DEFAULT_VALS = [50,30,70,20,40,60,80,10,35];
const inp: React.CSSProperties = { height:32, padding:"0 10px", borderRadius:8, background:C.bg, border:`1px solid ${C.border}`, color:C.text, fontSize:12, fontFamily:"'JetBrains Mono',monospace", outline:"none", width:90 };

export default function TreesPage() {
  const params = useParams();
  const algoId = params.algorithm as string;
  const algo = useAlgorithmMeta(algoId);
  const { setAlgorithm, setTreeSteps, reset } = useVisualizerStore();
  const [bstRoot, setBstRoot] = useState<TreeNode|null>(() => buildBST(DEFAULT_VALS));
  const [inputVal, setInputVal] = useState(25);
  const [travOrder, setTravOrder] = useState<number[]>([]);
  const [activeTrav, setActiveTrav] = useState<TraversalType|null>(null);
  const [history, setHistory] = useState<string[]>([`Built BST: [${DEFAULT_VALS.join(", ")}]`]);
  const currentStep = useCurrentTreeStep();
  const hasSteps = useVisualizerStore(s => s.treeSteps.length > 0);
  const layout = computeTreeLayout(bstRoot, 680);

  useEffect(() => { setAlgorithm(algoId); }, [algoId, setAlgorithm]);

  const handleInsert = useCallback(() => {
    if (!inputVal) return;
    setTreeSteps(bstInsertSteps(bstRoot, inputVal));
    setBstRoot(prev => insertBSTNode(prev, inputVal));
    setHistory(h => [`Insert ${inputVal}`, ...h.slice(0,9)]);
    setTimeout(() => useVisualizerStore.getState().play(), 80);
  }, [inputVal, bstRoot, setTreeSteps]);

  const handleSearch = useCallback(() => {
    if (!inputVal) return;
    setTreeSteps(bstSearchSteps(bstRoot, inputVal));
    setHistory(h => [`Search ${inputVal}`, ...h.slice(0,9)]);
    setTimeout(() => useVisualizerStore.getState().play(), 80);
  }, [inputVal, bstRoot, setTreeSteps]);

  const handleTraversal = useCallback((type: TraversalType) => {
    const steps = treeTraversalSteps(bstRoot, type);
    setTreeSteps(steps);
    setActiveTrav(type);
    const last = steps[steps.length-1];
    if (last?.auxiliaryData && Array.isArray((last.auxiliaryData as {result:number[]}).result))
      setTravOrder((last.auxiliaryData as {result:number[]}).result);
    setHistory(h => [`${type} traversal`, ...h.slice(0,9)]);
    setTimeout(() => useVisualizerStore.getState().play(), 80);
  }, [bstRoot, setTreeSteps]);

  const handleReset = useCallback(() => {
    setBstRoot(buildBST(DEFAULT_VALS)); setHistory([`Reset BST`]);
    setTravOrder([]); setActiveTrav(null); reset();
  }, [reset]);

  if (!algo) return null;
  const isBST = algoId === "bst";
  const isTrav = algoId === "tree-traversal";

  return (
    <div style={{ display:"flex", flexDirection:"column", height:"100%", background:C.bg }}>
      <div className="page-header" style={{ padding:"13px 20px", borderBottom:`1px solid ${C.border}`, background:C.bgSubtle, display:"flex", alignItems:"center", gap:11, flexShrink:0 }}>
        <div style={{ width:32, height:32, borderRadius:8, background:`${C.amber}18`, border:`1px solid ${C.amber}28`, display:"flex", alignItems:"center", justifyContent:"center", flexShrink:0 }}>
          <GitBranch size={15} color={C.amber}/>
        </div>
        <div style={{ minWidth:0 }}>
          <h1 style={{ fontSize:15, fontWeight:800, color:C.text, margin:0 }}>{algo.name}</h1>
          <p style={{ fontSize:10, color:C.text3, fontFamily:"'JetBrains Mono',monospace", margin:0 }}>trees · {algo.complexity.time.average}</p>
        </div>
      </div>
      <div style={{ flex:1, overflowY:"auto", overflowX:"hidden" }}>
        <div className="page-content">
          <div className="control-bar" style={{ background:C.surface, border:`1px solid ${C.border}` }}>
            <div style={{ display:"flex", flexDirection:"column", gap:5 }}>
              <label style={{ fontSize:10, fontWeight:700, letterSpacing:"0.1em", textTransform:"uppercase", color:C.text4 }}>Value</label>
              <input type="number" value={inputVal} onChange={e=>setInputVal(Number(e.target.value))} style={inp}/>
            </div>
            {isBST && <>
              <Button variant="default" size="sm" onClick={handleInsert}><Plus size={12}/> Insert</Button>
              <Button variant="secondary" size="sm" onClick={handleSearch}><Search size={12}/> Search</Button>
            </>}
            {isTrav && (["inorder","preorder","postorder"] as TraversalType[]).map(t => (
              <Button key={t} size="sm" variant={activeTrav===t?"default":"secondary"} onClick={() => handleTraversal(t)}>
                {t.charAt(0).toUpperCase()+t.slice(1)}
              </Button>
            ))}
            <button onClick={handleReset} style={{ width:30, height:30, borderRadius:7, border:`1px solid ${C.border}`, background:"transparent", cursor:"pointer", color:C.text3, display:"flex", alignItems:"center", justifyContent:"center" }}>
              <RotateCcw size={12}/>
            </button>
            <div style={{ flex:1, minWidth:10 }}/>
            <PlaybackBar hasSteps={hasSteps}/>
          </div>
          <StepDescriptionBar/>
          <div className="resp-grid-2col">
            <div style={{ display:"flex", flexDirection:"column", gap:14, minWidth:0, overflow:"hidden" }}>
              <TreeVisualizer layout={layout} step={currentStep} height={340}/>
              <AnimatePresence>
                {travOrder.length>0 && isTrav && (
                  <motion.div initial={{ opacity:0, y:6 }} animate={{ opacity:1, y:0 }} exit={{ opacity:0 }}
                    style={{ borderRadius:12, background:C.surface, border:`1px solid ${C.border}`, padding:14 }}>
                    <p style={{ fontSize:10, fontWeight:700, letterSpacing:"0.1em", textTransform:"uppercase", color:C.text4, marginBottom:10 }}>
                      {activeTrav} result{activeTrav==="inorder"?" (sorted for BST)":""}
                    </p>
                    <div style={{ display:"flex", alignItems:"center", gap:5, flexWrap:"wrap" }}>
                      {travOrder.map((v,i) => (
                        <motion.div key={`${v}-${i}`} initial={{ scale:0 }} animate={{ scale:1 }} transition={{ delay:i*0.04 }}
                          style={{ display:"flex", alignItems:"center", gap:4 }}>
                          <span style={{ width:32, height:32, borderRadius:7, background:C.surfaceRaised, border:`1px solid ${C.border}`,
                            display:"flex", alignItems:"center", justifyContent:"center",
                            fontFamily:"'JetBrains Mono',monospace", fontSize:11, fontWeight:700, color:C.text2 }}>{v}</span>
                          {i<travOrder.length-1 && <span style={{ fontSize:10, color:C.text4 }}>→</span>}
                        </motion.div>
                      ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
              <div style={{ borderRadius:12, background:C.surface, border:`1px solid ${C.border}`, overflow:"hidden" }}>
                <div style={{ padding:"8px 14px", borderBottom:`1px solid ${C.border}`, background:C.bgSubtle }}>
                  <span style={{ fontSize:11, fontWeight:600, color:C.text2 }}>Operation History</span>
                </div>
                <div style={{ padding:8, display:"flex", flexDirection:"column", gap:2 }}>
                  {history.map((op,i) => (
                    <div key={i} style={{ padding:"4px 10px", borderRadius:5, fontSize:11, fontFamily:"'JetBrains Mono',monospace",
                      background:i===0?`${C.accent}08`:"transparent", color:i===0?C.accent:C.text4,
                      outline:i===0?`1px solid ${C.accent}20`:"none" }}>{op}</div>
                  ))}
                </div>
              </div>
              <PseudocodePanel algorithmId={algoId}/>
            </div>
            <div style={{ display:"flex", flexDirection:"column", gap:14 }}>
              <ComplexityPanel algo={algo}/>
              <DescriptionPanel algo={algo}/>
              <div style={{ borderRadius:12, background:C.surface, border:`1px solid ${C.border}`, padding:14 }}>
                <p style={{ fontSize:10, fontWeight:700, letterSpacing:"0.1em", textTransform:"uppercase", color:C.text4, marginBottom:10 }}>Tree Stats</p>
                <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:8 }}>
                  {[["Nodes",layout.length],["Height",Math.max(1,Math.ceil(Math.log2(layout.length+1)))]].map(([l,v]) => (
                    <div key={String(l)} style={{ borderRadius:8, background:C.surfaceRaised, border:`1px solid ${C.border}`, padding:"10px 8px", textAlign:"center" }}>
                      <p style={{ fontSize:10, color:C.text4, marginBottom:3 }}>{l}</p>
                      <p style={{ fontSize:22, fontWeight:800, fontFamily:"'JetBrains Mono',monospace", color:C.text2 }}>{v}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
