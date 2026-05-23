"use client";
import { Play, Pause, RotateCcw, ChevronLeft, ChevronRight, Gauge } from "lucide-react";
import { C } from "@/lib/utils";
import { Button } from "@/components/ui/primitives";
import { usePlaybackControls } from "@/hooks/useVisualizer";

interface PlaybackBarProps {
  onVisualize?: () => void;
  hasSteps: boolean;
}

export function PlaybackBar({ onVisualize, hasSteps }: PlaybackBarProps) {
  // This hook now registers keyboard shortcuts with ref-based approach
  const {
    playbackState, currentStep, totalSteps, speed, progress,
    isPlaying, canStepForward, canStepBackward,
    reset, stepForward, stepBackward, togglePlay, setSpeed,
  } = usePlaybackControls();

  const isIdle = playbackState === "idle";

  return (
    <div style={{ display:"flex", flexDirection:"column", gap:7 }}>
      {/* Progress bar */}
      {hasSteps && (
        <div style={{ height:3, borderRadius:99, background:C.surfaceRaised, overflow:"hidden" }}>
          <div style={{
            height:"100%", borderRadius:99,
            width:`${progress}%`,
            background:`linear-gradient(90deg, ${C.accent}, ${C.blue})`,
            transition:"width 0.12s linear",
          }}/>
        </div>
      )}

      {/* Controls row */}
      <div style={{ display:"flex", alignItems:"center", gap:5, flexWrap:"wrap" }}>
        {/* Visualize button */}
        {onVisualize && (
          <Button variant="default" size="sm"
            onClick={() => { reset(); setTimeout(() => onVisualize(), 50); }}
            style={{ gap:5 }}>
            <Play size={12}/> {isIdle ? "Visualize" : "Re-run"}
          </Button>
        )}

        {hasSteps && <div style={{ width:1, height:18, background:C.border, margin:"0 1px" }}/>}

        {/* Playback controls */}
        {hasSteps && (
          <>
            <Button variant="ghost" size="icon-sm" onClick={reset} title="Reset (R)">
              <RotateCcw size={12}/>
            </Button>
            <Button variant="ghost" size="icon-sm" onClick={stepBackward}
              disabled={!canStepBackward} title="Prev step (←)">
              <ChevronLeft size={13}/>
            </Button>
            <Button
              variant={isPlaying ? "secondary" : "default"}
              size="icon"
              onClick={togglePlay}
              title="Play / Pause (Space)"
              style={isPlaying ? {
                background:`${C.amber}15`,
                border:`1px solid ${C.amber}40`,
                color:C.amber,
              } : {}}
            >
              {isPlaying
                ? <Pause size={14}/>
                : <Play size={14} style={{ transform:"translateX(1px)" }}/>
              }
            </Button>
            <Button variant="ghost" size="icon-sm" onClick={stepForward}
              disabled={!canStepForward} title="Next step (→)">
              <ChevronRight size={13}/>
            </Button>
          </>
        )}

        <div style={{ flex:1 }}/>

        {/* Step counter */}
        {hasSteps && totalSteps > 0 && (
          <span style={{
            fontFamily:"'JetBrains Mono',monospace",
            fontSize:11, color:C.text3, flexShrink:0,
          }}>
            <span style={{ color:C.text2, fontWeight:600 }}>{currentStep+1}</span>
            {" / "}{totalSteps}
          </span>
        )}

        {/* Speed control */}
        <div style={{
          display:"flex", alignItems:"center", gap:6,
          background:C.surfaceRaised, border:`1px solid ${C.border}`,
          borderRadius:8, padding:"5px 10px", flexShrink:0,
        }}>
          <Gauge size={11} color={C.text3}/>
          <input
            type="range" min={1} max={10} value={speed}
            onChange={e => setSpeed(Number(e.target.value))}
            style={{ width:64, height:3, accentColor:C.accent, cursor:"pointer" }}
          />
          <span style={{
            fontFamily:"'JetBrains Mono',monospace",
            fontSize:11, color:C.text2, minWidth:11,
          }}>{speed}</span>
        </div>
      </div>
    </div>
  );
}
