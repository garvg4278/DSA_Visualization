"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Copy, Check, ChevronDown, Code2, FileText } from "lucide-react";
import { C } from "@/lib/utils";
import { PSEUDOCODE_DATA } from "@/data/pseudocode";
import type { Approach } from "@/data/pseudocode";
import { highlightCode } from "@/lib/highlighter";

type Lang = "cpp" | "java" | "python";

const LANG_CONFIG: Record<Lang, { label: string; color: string }> = {
  cpp:    { label: "C++",    color: "hsl(199 89% 62%)" },
  java:   { label: "Java",   color: "hsl(38  92% 60%)" },
  python: { label: "Python", color: "hsl(142 71% 55%)" },
};

const baseCode: React.CSSProperties = {
  margin: 0, padding: "16px 18px",
  fontSize: 13, lineHeight: 1.75,
  fontFamily: "'JetBrains Mono', 'Fira Code', 'Courier New', monospace",
  background: "hsl(222 20% 5%)",
  color: "hsl(220 15% 75%)",
  overflowX: "auto", overflowY: "auto",
  maxHeight: 340,
  whiteSpace: "pre",
  tabSize: 4,
};

function ApproachCard({
  approach, isActive, onClick,
}: {
  approach: Approach;
  isActive: boolean;
  onClick: () => void;
}) {
  const [lang, setLang] = useState<Lang>("cpp");
  const [copied, setCopied] = useState(false);
  const [view, setView] = useState<"pseudo" | "code">("pseudo");

  const handleCopy = async () => {
    const text = view === "pseudo" ? approach.pseudocode : approach.code[lang];
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const highlightedCode = highlightCode(approach.code[lang], lang);

  return (
    <div style={{
      border: `1px solid ${isActive ? C.accent + "50" : C.border}`,
      background: isActive ? `${C.accent}05` : C.surface,
      transition: "border-color 0.18s",
      borderRadius: 0,
    }}>
      {/* Accordion Header */}
      <button
        onClick={onClick}
        style={{
          width: "100%", display: "flex", alignItems: "center",
          justifyContent: "space-between", padding: "13px 16px",
          background: "transparent", border: "none", cursor: "pointer",
          borderBottom: isActive ? `1px solid ${C.border}` : "none",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{
            fontSize: 13, fontWeight: 700,
            color: isActive ? C.accent : C.text2,
          }}>
            {approach.label}
          </span>
          <span style={{
            fontSize: 10, fontFamily: "'JetBrains Mono', monospace",
            fontWeight: 600, padding: "2px 8px", borderRadius: 99,
            background: `${approach.badgeColor}22`,
            color: approach.badgeColor,
            border: `1px solid ${approach.badgeColor}35`,
          }}>
            {approach.badge}
          </span>
        </div>
        <ChevronDown
          size={14} color={C.text4}
          style={{
            transition: "transform 0.2s",
            transform: isActive ? "rotate(180deg)" : "rotate(0deg)",
          }}
        />
      </button>

      <AnimatePresence>
        {isActive && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
            style={{ overflow: "hidden" }}
          >
            {/* Description */}
            <p style={{
              margin: 0, padding: "12px 16px 10px",
              fontSize: 12.5, color: C.text3, lineHeight: 1.65,
              borderBottom: `1px solid ${C.border}`,
            }}>
              {approach.description}
            </p>

            {/* Toolbar: pseudo/code toggle + language + copy */}
            <div style={{
              display: "flex", alignItems: "center", justifyContent: "space-between",
              padding: "7px 12px",
              background: "hsl(222 18% 8%)",
              borderBottom: `1px solid ${C.border}`,
              flexWrap: "wrap", gap: 6,
            }}>
              {/* View toggle */}
              <div style={{
                display: "flex", gap: 2,
                background: C.surface, borderRadius: 7, padding: 2,
                border: `1px solid ${C.border}`,
              }}>
                {([["pseudo", <FileText size={11} key="f" />, "Pseudocode"],
                   ["code",   <Code2 size={11} key="c" />,   "Code"      ]] as const).map(([v, icon, label]) => (
                  <button
                    key={v}
                    onClick={() => setView(v as "pseudo" | "code")}
                    style={{
                      display: "flex", alignItems: "center", gap: 5,
                      padding: "4px 10px", borderRadius: 5, border: "none",
                      cursor: "pointer", fontSize: 11, fontWeight: 600,
                      background: view === v ? C.surfaceRaised : "transparent",
                      color: view === v ? C.text : C.text4,
                      transition: "all 0.12s",
                    }}
                  >
                    {icon} {label}
                  </button>
                ))}
              </div>

              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                {/* Language picker — only show in code view */}
                {view === "code" && (
                  <div style={{ display: "flex", gap: 2 }}>
                    {(["cpp", "java", "python"] as Lang[]).map((l) => (
                      <button
                        key={l}
                        onClick={() => setLang(l)}
                        style={{
                          padding: "3px 9px", borderRadius: 5,
                          border: "none", cursor: "pointer",
                          fontSize: 11, fontWeight: 600,
                          fontFamily: "'JetBrains Mono', monospace",
                          background: lang === l ? C.surfaceOverlay : "transparent",
                          color: lang === l ? LANG_CONFIG[l].color : C.text4,
                          transition: "all 0.1s",
                        }}
                      >
                        {LANG_CONFIG[l].label}
                      </button>
                    ))}
                  </div>
                )}

                {/* Copy */}
                <button
                  onClick={handleCopy}
                  style={{
                    display: "flex", alignItems: "center", gap: 4,
                    padding: "4px 10px", borderRadius: 6,
                    border: `1px solid ${C.border}`, background: C.surfaceRaised,
                    cursor: "pointer", fontSize: 10,
                    color: copied ? C.green : C.text3,
                    fontFamily: "'JetBrains Mono', monospace",
                  }}
                >
                  {copied ? <Check size={10} /> : <Copy size={10} />}
                  {copied ? "Copied!" : "Copy"}
                </button>
              </div>
            </div>

            {/* Content */}
            <AnimatePresence mode="wait">
              {view === "pseudo" ? (
                <motion.pre
                  key="pseudo"
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  transition={{ duration: 0.1 }}
                  style={{
                    ...baseCode,
                    color: "hsl(220 15% 80%)",
                    fontStyle: "normal",
                  }}
                >
                  {approach.pseudocode}
                </motion.pre>
              ) : (
                <motion.div
                  key={`code-${lang}`}
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  transition={{ duration: 0.1 }}
                >
                  <pre
                    style={baseCode}
                    dangerouslySetInnerHTML={{ __html: highlightedCode }}
                  />
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

interface PseudocodePanelProps {
  algorithmId: string;
}

export function PseudocodePanel({ algorithmId }: PseudocodePanelProps) {
  const data = PSEUDOCODE_DATA[algorithmId];
  const [activeIdx, setActiveIdx] = useState(0);

  if (!data) {
    return (
      <div style={{
        borderRadius: 12, background: C.surface, border: `1px solid ${C.border}`,
        padding: 20, textAlign: "center",
        color: C.text3, fontSize: 12, fontFamily: "'JetBrains Mono', monospace",
      }}>
        Pseudocode coming soon for this algorithm.
      </div>
    );
  }

  return (
    <div style={{ borderRadius: 12, overflow: "hidden", border: `1px solid ${C.border}` }}>
      {/* Panel header */}
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "10px 16px", background: C.bgSubtle,
        borderBottom: `1px solid ${C.border}`,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
          <FileText size={13} color={C.text3} />
          <span style={{ fontSize: 12, fontWeight: 600, color: C.text2 }}>
            Pseudocode &amp; Implementations
          </span>
        </div>
        <span style={{
          fontSize: 10, fontFamily: "'JetBrains Mono', monospace", color: C.text4,
          background: C.surfaceRaised, border: `1px solid ${C.border}`,
          borderRadius: 99, padding: "2px 8px",
        }}>
          {data.approaches.length} approach{data.approaches.length > 1 ? "es" : ""}
        </span>
      </div>

      {/* Approaches */}
      <div style={{ background: C.surface }}>
        {data.approaches.map((approach, i) => (
          <div
            key={i}
            style={{
              borderBottom: i < data.approaches.length - 1
                ? `1px solid ${C.border}` : "none",
            }}
          >
            <ApproachCard
              approach={approach}
              isActive={activeIdx === i}
              onClick={() => setActiveIdx(activeIdx === i ? -1 : i)}
            />
          </div>
        ))}
      </div>
    </div>
  );
}
