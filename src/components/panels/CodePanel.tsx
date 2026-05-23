"use client";
import { useState, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Copy, Check, Code2 } from "lucide-react";
import { C } from "@/lib/utils";
import type { CodeLanguage } from "@/types";

const LANGS: { id: CodeLanguage; label: string; color: string }[] = [
  { id: "typescript", label: "TS",     color: C.blue },
  { id: "python",     label: "Python", color: C.green },
  { id: "java",       label: "Java",   color: C.amber },
  { id: "cpp",        label: "C++",    color: C.purple },
];

function highlight(code: string, lang: CodeLanguage): string {
  const kws: Record<CodeLanguage, string[]> = {
    typescript: ["function","const","let","var","return","if","else","while","for","of","in","new","class","type","interface","export","import","from","break","true","false","null","undefined","void"],
    python: ["def","return","if","else","elif","while","for","in","not","and","or","import","from","class","True","False","None","len","range","self"],
    java: ["public","private","static","void","int","boolean","String","return","if","else","while","for","new","class","import","true","false","null","List","Arrays","Math"],
    cpp: ["int","bool","void","return","if","else","while","for","auto","vector","string","true","false","nullptr","new","class","struct","using","namespace","std","size_t","long","short"],
  };
  return code
    .replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")
    .replace(/(\/\/[^\n]*)/g,'<span class="token-comment">$1</span>')
    .replace(/(#[^\n]*)/g,'<span class="token-comment">$1</span>')
    .replace(/("([^"\\]|\\.)*"|'([^'\\]|\\.)*'|`([^`\\]|\\.)*`)/g,'<span class="token-string">$1</span>')
    .replace(/\b(\d+)\b/g,'<span class="token-number">$1</span>')
    .replace(new RegExp(`\\b(${(kws[lang]??[]).join("|")})\\b`,"g"),'<span class="token-keyword">$1</span>')
    .replace(/\b([a-zA-Z_][a-zA-Z0-9_]*)\s*(?=\()/g,'<span class="token-function">$1</span>');
}

export function CodePanel({ implementations, title }: { implementations: Record<string, string>; title?: string }) {
  const [lang, setLang] = useState<CodeLanguage>("typescript");
  const [copied, setCopied] = useState(false);
  const code = implementations[lang] ?? "// Not available for this language";
  const highlighted = useMemo(() => highlight(code, lang), [code, lang]);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true); setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div style={{ borderRadius: 12, background: C.surface, border: `1px solid ${C.border}`, overflow: "hidden" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "8px 14px", borderBottom: `1px solid ${C.border}`, background: C.bgSubtle }}>
        <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
          <Code2 size={13} color={C.text3} />
          <span style={{ fontSize: 12, fontWeight: 600, color: C.text2 }}>{title ?? "Implementation"}</span>
        </div>
        <div style={{ display: "flex", gap: 3 }}>
          {LANGS.map(({ id, label, color }) => (
            <button key={id} onClick={() => setLang(id)} style={{
              padding: "3px 9px", borderRadius: 6, border: "none", cursor: "pointer",
              fontSize: 10.5, fontWeight: 600, fontFamily: "'JetBrains Mono', monospace",
              background: lang === id ? C.surfaceOverlay : "transparent",
              color: lang === id ? color : C.text4, transition: "all 0.12s",
            }}>{label}</button>
          ))}
        </div>
      </div>
      <div style={{ position: "relative" }}>
        <AnimatePresence mode="wait">
          <motion.pre key={lang} initial={{ opacity: 0, y: 3 }} animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -3 }} transition={{ duration: 0.13 }}
            style={{ margin: 0, padding: "14px 16px", fontSize: 12, lineHeight: 1.7,
              fontFamily: "'JetBrains Mono', monospace", color: C.text2,
              background: C.bg, overflowX: "auto", maxHeight: 300 }}
            dangerouslySetInnerHTML={{ __html: highlighted }} />
        </AnimatePresence>
        <button onClick={handleCopy} style={{
          position: "absolute", top: 10, right: 10,
          display: "flex", alignItems: "center", gap: 4, padding: "3px 9px",
          borderRadius: 6, border: `1px solid ${C.border}`, background: C.surfaceRaised,
          cursor: "pointer", fontSize: 10, color: copied ? C.green : C.text3,
          fontFamily: "'JetBrains Mono', monospace",
        }}>
          {copied ? <Check size={10} /> : <Copy size={10} />}
          {copied ? "Copied!" : "Copy"}
        </button>
      </div>
    </div>
  );
}
