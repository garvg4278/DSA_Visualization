import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: {
    template: "%s | AlgoVista",
    default: "AlgoVista — DSA Visualization Platform",
  },
  description:
    "An industry-grade platform for visualizing Data Structures & Algorithms. Step through 22+ algorithms with animated visualizations, pseudocode, complexity analysis, and multi-language code.",
  keywords: [
    "DSA visualizer", "algorithm visualization", "data structures",
    "sorting algorithms", "graph algorithms", "dynamic programming",
    "binary search tree", "computer science", "Next.js", "TypeScript",
  ],
  authors: [{ name: "AlgoVista" }],
  metadataBase: new URL("https://dsa-visualization-ten.vercel.app"),
  openGraph: {
    title: "AlgoVista — DSA Visualization Platform",
    description: "Visualize. Understand. Master.",
    type: "website",
    images: [{ url: "/og-image.svg", width: 1200, height: 630 }],
  },
  twitter: {
    card: "summary_large_image",
    title: "AlgoVista — DSA Visualization Platform",
    description: "Visualize. Understand. Master.",
    images: ["/og-image.svg"],
  },
  icons: {
    icon: [
      { url: "/favicon.png", type: "image/png", sizes: "32x32" },
      { url: "/favicon.svg", type: "image/svg+xml" },
    ],
    apple: "/assets/logo/Favicon.png",
    shortcut: "/favicon.png",
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#0d0f17",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        {/* Google Fonts loaded via link tag — works in all environments */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Outfit:wght@300;400;500;600;700;800;900&display=swap"
          rel="stylesheet"
        />
      </head>
      <body style={{ fontFamily: "'Outfit', system-ui, sans-serif", margin: 0, padding: 0 }}>
        {children}
      </body>
    </html>
  );
}
