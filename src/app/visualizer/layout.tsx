import { Sidebar } from "@/components/layout/Sidebar";

export default function VisualizerLayout({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      display: "flex",
      height: "100vh",
      overflow: "hidden",
      background: "hsl(222 18% 7%)",
    }}>
      <Sidebar />
      <main style={{
        flex: 1,
        overflowY: "auto",
        overflowX: "hidden",
        minWidth: 0, // critical - prevents flex child overflow
      }}>
        {children}
      </main>
    </div>
  );
}
