"use client";
import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowRight, GitBranch, Network, TableProperties, ArrowUpDown, Search, Play, TrendingUp, GitCompare, Terminal, Code2, BarChart3, ChevronRight } from "lucide-react";
import { ALGORITHMS_BY_CATEGORY, CATEGORY_CONFIG, APP_CONFIG } from "@/constants";
import { C } from "@/lib/utils";
import type { AlgorithmCategory } from "@/types";

const fadeUp = {
  hidden: { opacity: 0, y: 20 },
  visible: (i: number) => ({ opacity: 1, y: 0, transition: { delay: i*0.07, duration: 0.45, ease: [0.16,1,0.3,1] } }),
};

const ICON_MAP: Record<AlgorithmCategory, React.ReactNode> = {
  sorting:     <ArrowUpDown size={18}/>,
  searching:   <Search size={18}/>,
  graph:       <Network size={18}/>,
  trees:       <GitBranch size={18}/>,
  dp:          <TableProperties size={18}/>,
  recursion:   <Terminal size={18}/>,
  backtracking:<Terminal size={18}/>,
};

function CategoryCard({ category }: { category: AlgorithmCategory }) {
  const config = CATEGORY_CONFIG[category];
  const algos = ALGORITHMS_BY_CATEGORY[category] ?? [];
  const first = algos[0];
  return (
    <motion.div whileHover={{ y: -3 }} transition={{ duration: 0.18 }}>
      <Link href={first ? `/visualizer/${category}/${first.id}` : "/"} style={{ textDecoration: "none", display: "block" }}>
        <div style={{ borderRadius: 14, background: C.surface, border: `1px solid ${C.border}`, padding: 20,
          position: "relative", overflow: "hidden", transition: "border-color 0.2s" }}
          onMouseEnter={e => (e.currentTarget.style.borderColor = `${config.color}50`)}
          onMouseLeave={e => (e.currentTarget.style.borderColor = C.border)}>
          <div style={{ position: "absolute", top: -24, right: -24, width: 80, height: 80,
            borderRadius: "50%", background: config.color, opacity: 0.07, filter: "blur(20px)" }} />
          <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: 14 }}>
            <div style={{ width: 40, height: 40, borderRadius: 11, display: "flex", alignItems: "center",
              justifyContent: "center", background: `${config.color}18`, border: `1px solid ${config.color}30`,
              color: config.color }}>
              {ICON_MAP[category]}
            </div>
            <span style={{ fontSize: 10, fontFamily: "'JetBrains Mono', monospace", color: C.text4,
              background: C.surfaceRaised, border: `1px solid ${C.border}`, borderRadius: 99, padding: "3px 9px" }}>
              {algos.length} algos
            </span>
          </div>
          <h3 style={{ fontSize: 14, fontWeight: 800, color: C.text, margin: "0 0 5px" }}>{config.label}</h3>
          <p style={{ fontSize: 12, color: C.text3, lineHeight: 1.6, margin: "0 0 14px" }}>{config.description}</p>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginBottom: 14 }}>
            {algos.slice(0,3).map(a => (
              <span key={a.id} style={{ fontSize: 10, padding: "2px 7px", borderRadius: 4, fontFamily: "'JetBrains Mono', monospace",
                background: C.surfaceRaised, border: `1px solid ${C.borderSubtle}`, color: C.text4 }}>{a.name}</span>
            ))}
            {algos.length > 3 && <span style={{ fontSize: 10, padding: "2px 7px", borderRadius: 4,
              background: C.surfaceRaised, border: `1px solid ${C.borderSubtle}`, color: C.text4 }}>+{algos.length-3}</span>}
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 11, fontWeight: 600, color: config.color }}>
            Explore <ChevronRight size={13}/>
          </div>
        </div>
      </Link>
    </motion.div>
  );
}

const DEMO_BARS = [17, 28, 38, 45, 55, 62, 72, 77, 84, 91];
const DEMO_UNSORTED = [45, 72, 28, 91, 55, 38, 84, 62, 17, 77];

function HeroDemo() {
  return (
    <div style={{ borderRadius: 16, background: C.surface, border: `1px solid ${C.border}`, padding: 22, position: "relative", overflow: "hidden" }}>
      {/* Terminal dots */}
      <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 18 }}>
        {[C.red, C.amber, C.green].map((c,i) => <div key={i} style={{ width: 10, height: 10, borderRadius: "50%", background: c, opacity: 0.7 }}/>)}
        <span style={{ marginLeft: 10, fontSize: 11, fontFamily: "'JetBrains Mono', monospace", color: C.text4 }}>bubble-sort.ts — step 7/28</span>
        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 5,
          background: `${C.green}15`, border: `1px solid ${C.green}30`, borderRadius: 8, padding: "3px 9px" }}>
          <div style={{ width: 5, height: 5, borderRadius: "50%", background: C.green, animation: "pulse-subtle 1.5s infinite" }}/>
          <span style={{ fontSize: 10, fontFamily: "'JetBrains Mono', monospace", color: C.green }}>Playing</span>
        </div>
      </div>
      {/* Bars */}
      <div style={{ display: "flex", alignItems: "flex-end", gap: 5, height: 110 }}>
        {DEMO_BARS.map((val, i) => {
          const color = i < 5 ? C.vizSorted : i < 7 ? C.vizCompare : C.vizDefault;
          return (
            <motion.div key={i} initial={{ height: (DEMO_UNSORTED[i]/100)*110 }}
              animate={{ height: (val/100)*110 }}
              transition={{ duration: 0.7, delay: i*0.05, ease: [0.16,1,0.3,1] }}
              style={{ flex: 1, borderRadius: "3px 3px 0 0", background: color,
                boxShadow: i < 5 ? `0 0 8px ${color}60` : undefined }} />
          );
        })}
      </div>
      {/* Step description */}
      <div style={{ marginTop: 14, padding: "8px 12px", borderRadius: 8, background: C.bgSubtle,
        border: `1px solid ${C.border}`, display: "flex", alignItems: "center", gap: 8 }}>
        <div style={{ width: 6, height: 6, borderRadius: "50%", background: C.amber, flexShrink: 0 }}/>
        <span style={{ fontSize: 11, fontFamily: "'JetBrains Mono', monospace", color: C.text2 }}>
          Pass 2: Comparing [55] and [62] → no swap needed
        </span>
      </div>
    </div>
  );
}

const FEATURES = [
  { icon: <Play size={15}/>,       title: "Step-by-step Playback",   desc: "Play, pause, step forward/backward through every operation with keyboard shortcuts." },
  { icon: <TrendingUp size={15}/>, title: "Complexity Analysis",      desc: "Best, average, worst time + space complexity with visual bars for every algorithm." },
  { icon: <Code2 size={15}/>,      title: "Pseudocode + 3 Languages", desc: "Pseudocode and implementations in C++, Java, and Python — brute force to optimal." },
  { icon: <GitCompare size={15}/>, title: "Algorithm Comparison",     desc: "Run two algorithms side-by-side on the same input. See step counts and winners." },
  { icon: <BarChart3 size={15}/>,  title: "DP Table Visualization",   desc: "Watch 2D DP tables fill cell-by-cell with dependency highlighting." },
  { icon: <Network size={15}/>,    title: "Interactive Graphs",       desc: "BFS, DFS, Dijkstra, Kruskal — animated traversals with edge weights and distances." },
];

export default function HomePage() {
  const cats: AlgorithmCategory[] = ["sorting","searching","graph","trees","dp"];
  const total = Object.values(ALGORITHMS_BY_CATEGORY).reduce((s,a)=>s+a.length, 0);

  return (
    <div style={{ minHeight: "100vh", background: C.bg }}>
      {/* Nav */}
      <nav style={{ borderBottom: `1px solid ${C.border}`, background: `${C.bgSubtle}CC`,
        backdropFilter: "blur(12px)", position: "sticky", top: 0, zIndex: 50 }}>
        <div style={{ maxWidth: 1100, margin: "0 auto", padding: "0 24px", height: 52,
          display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
            <div style={{ width: 30, height: 30, borderRadius: 8, overflow: "hidden", flexShrink: 0 }}>
              <img src="/assets/logo/Favicon.png" alt="AlgoVista" style={{ width: "100%", height: "100%", objectFit: "cover" }}/>
            </div>
            <span className="gradient-text" style={{ fontSize: 15, fontWeight: 800 }}>{APP_CONFIG.name}</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <Link href="/compare" style={{ fontSize: 12, fontWeight: 500, color: C.text3, textDecoration: "none",
              padding: "6px 12px", borderRadius: 7, transition: "color 0.15s" }}
              onMouseEnter={e => (e.currentTarget.style.color = C.text2)}
              onMouseLeave={e => (e.currentTarget.style.color = C.text3)}>Compare</Link>
            <Link href="/visualizer/sorting/bubble-sort" style={{
              fontSize: 12, fontWeight: 600, color: "#fff", textDecoration: "none",
              padding: "7px 16px", borderRadius: 8, display: "flex", alignItems: "center", gap: 5,
              background: `linear-gradient(135deg, ${C.accent}, ${C.accentDim})`,
              boxShadow: `0 0 18px ${C.accent}30` }}>
              Launch App <ArrowRight size={12}/>
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <section style={{ position: "relative", overflow: "hidden" }}>
        <div className="dot-grid-bg" style={{ position: "absolute", inset: 0, opacity: 0.4 }}/>
        <div style={{ position: "absolute", top: "30%", left: "60%", width: 400, height: 400, borderRadius: "50%",
          background: `${C.accent}08`, filter: "blur(80px)", pointerEvents: "none" }}/>
        <div style={{ maxWidth: 1100, margin: "0 auto", padding: "80px 24px 72px",
          display: "grid", gridTemplateColumns: "1fr 420px", gap: 60, alignItems: "center" }}>
          <div>
            <motion.div initial={{ opacity:0, y:12 }} animate={{ opacity:1, y:0 }} transition={{ duration:0.45 }}
              style={{ display: "inline-flex", alignItems: "center", gap: 7, padding: "5px 12px",
                borderRadius: 99, background: `${C.accent}15`, border: `1px solid ${C.accent}30`,
                color: C.accent, fontSize: 11, fontWeight: 600, marginBottom: 22 }}>
              ⚡ Industry-grade DSA Platform
            </motion.div>
            <motion.h1 initial={{ opacity:0, y:18 }} animate={{ opacity:1, y:0 }} transition={{ delay:0.05, duration:0.55 }}
              className="gradient-text"
              style={{ fontSize: 54, fontWeight: 900, lineHeight: 1.05, letterSpacing: -1.5, margin: "0 0 18px" }}>
              {APP_CONFIG.tagline}
            </motion.h1>
            <motion.p initial={{ opacity:0, y:12 }} animate={{ opacity:1, y:0 }} transition={{ delay:0.1, duration:0.45 }}
              style={{ fontSize: 15, color: C.text2, lineHeight: 1.75, marginBottom: 30, maxWidth: 520 }}>
              Step through {total}+ algorithms frame-by-frame. Animated visualizations, complexity analysis,
              pseudocode, and multi-language implementations — all in one place.
            </motion.p>
            <motion.div initial={{ opacity:0, y:10 }} animate={{ opacity:1, y:0 }} transition={{ delay:0.15, duration:0.4 }}
              style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
              <Link href="/visualizer/sorting/bubble-sort" style={{
                display: "inline-flex", alignItems: "center", gap: 7, padding: "11px 22px",
                borderRadius: 10, background: `linear-gradient(135deg, ${C.accent}, ${C.accentDim})`,
                color: "#fff", fontWeight: 700, fontSize: 13, textDecoration: "none",
                boxShadow: `0 0 24px ${C.accent}30, 0 0 48px ${C.accent}10` }}>
                <Play size={14}/> Start Visualizing
              </Link>
              <Link href="/compare" style={{
                display: "inline-flex", alignItems: "center", gap: 7, padding: "11px 22px",
                borderRadius: 10, background: C.surface, border: `1px solid ${C.border}`,
                color: C.text2, fontWeight: 600, fontSize: 13, textDecoration: "none" }}>
                <GitCompare size={14}/> Compare Algorithms
              </Link>
            </motion.div>
            {/* Stats row */}
            <motion.div initial={{ opacity:0 }} animate={{ opacity:1 }} transition={{ delay:0.25 }}
              style={{ display: "flex", gap: 32, marginTop: 36, paddingTop: 28, borderTop: `1px solid ${C.borderSubtle}` }}>
              {[["22+","Algorithms"],["5","Categories"],["3","Languages"],["∞","Steps"]].map(([v,l]) => (
                <div key={l} style={{ textAlign: "center" }}>
                  <div className="gradient-text" style={{ fontSize: 26, fontWeight: 900, fontFamily: "'JetBrains Mono', monospace" }}>{v}</div>
                  <div style={{ fontSize: 11, color: C.text3, marginTop: 2 }}>{l}</div>
                </div>
              ))}
            </motion.div>
          </div>
          <motion.div initial={{ opacity:0, scale:0.96, y:16 }} animate={{ opacity:1, scale:1, y:0 }}
            transition={{ delay:0.18, duration:0.55, ease:[0.16,1,0.3,1] }}>
            <HeroDemo/>
          </motion.div>
        </div>
      </section>

      {/* Categories */}
      <section style={{ maxWidth: 1100, margin: "0 auto", padding: "60px 24px" }}>
        <div style={{ marginBottom: 32 }}>
          <p style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.14em", textTransform: "uppercase",
            color: C.text4, marginBottom: 10 }}>Algorithm Categories</p>
          <h2 style={{ fontSize: 28, fontWeight: 900, color: C.text, margin: 0 }}>Everything you need to master DSA</h2>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 14 }}>
          {cats.map((cat, i) => (
            <motion.div key={cat} custom={i} variants={fadeUp} initial="hidden" whileInView="visible" viewport={{ once: true }}>
              <CategoryCard category={cat}/>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Features */}
      <section style={{ background: C.bgSubtle, borderTop: `1px solid ${C.border}`, borderBottom: `1px solid ${C.border}` }}>
        <div style={{ maxWidth: 1100, margin: "0 auto", padding: "60px 24px" }}>
          <h2 style={{ fontSize: 28, fontWeight: 900, color: C.text, textAlign: "center", marginBottom: 36 }}>
            Built for serious learners
          </h2>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12 }}>
            {FEATURES.map((f, i) => (
              <motion.div key={f.title} custom={i} variants={fadeUp} initial="hidden" whileInView="visible" viewport={{ once: true }}
                style={{ borderRadius: 12, background: C.surface, border: `1px solid ${C.border}`, padding: 18 }}>
                <div style={{ width: 36, height: 36, borderRadius: 9, background: `${C.accent}15`, border: `1px solid ${C.accent}25`,
                  display: "flex", alignItems: "center", justifyContent: "center", color: C.accent, marginBottom: 14 }}>
                  {f.icon}
                </div>
                <h3 style={{ fontSize: 13, fontWeight: 700, color: C.text, margin: "0 0 6px" }}>{f.title}</h3>
                <p style={{ fontSize: 12, color: C.text3, lineHeight: 1.65, margin: 0 }}>{f.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section style={{ maxWidth: 1100, margin: "0 auto", padding: "72px 24px", textAlign: "center" }}>
        <div style={{ display: "inline-flex", alignItems: "center", gap: 6, padding: "5px 12px",
          borderRadius: 99, background: `${C.green}15`, border: `1px solid ${C.green}30`,
          color: C.green, fontSize: 11, fontWeight: 600, marginBottom: 20 }}>
          <div style={{ width: 6, height: 6, borderRadius: "50%", background: C.green, animation: "pulse-subtle 1.5s infinite" }}/>
          Ready to use · No login required
        </div>
        <h2 style={{ fontSize: 36, fontWeight: 900, color: C.text, margin: "0 0 28px" }}>
          Start with Bubble Sort.<br/>
          <span className="gradient-text">End with mastery.</span>
        </h2>
        <Link href="/visualizer/sorting/bubble-sort" style={{
          display: "inline-flex", alignItems: "center", gap: 8, padding: "13px 32px", borderRadius: 12,
          background: `linear-gradient(135deg, ${C.accent}, ${C.accentDim})`, color: "#fff",
          fontWeight: 700, fontSize: 14, textDecoration: "none",
          boxShadow: `0 0 28px ${C.accent}30, 0 0 56px ${C.accent}12` }}>
          Open Visualizer <ArrowRight size={15}/>
        </Link>
      </section>

      {/* Footer */}
      <footer style={{ borderTop: `1px solid ${C.border}`, padding: "20px 24px" }}>
        <div style={{ maxWidth: 1100, margin: "0 auto", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <img src="/assets/logo/Favicon.png" alt="AlgoVista" style={{ width: 18, height: 18, borderRadius: 4, objectFit: "cover" }}/>
            <span style={{ fontSize: 12, fontWeight: 600, color: C.text3 }}>{APP_CONFIG.name}</span>
          </div>
          <span style={{ fontSize: 11, color: C.text4, fontFamily: "'JetBrains Mono', monospace" }}>
            Next.js 15 · TypeScript · D3.js · Framer Motion · Zustand
          </span>
        </div>
      </footer>
    </div>
  );
}
