import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// CSS variable helpers for inline styles in Tailwind v4
export const C = {
  bg: "hsl(222 18% 7%)",
  bgSubtle: "hsl(222 18% 9%)",
  surface: "hsl(222 16% 11%)",
  surfaceRaised: "hsl(222 14% 14%)",
  surfaceOverlay: "hsl(222 14% 17%)",
  border: "hsl(220 12% 20%)",
  borderSubtle: "hsl(220 10% 15%)",
  text: "hsl(220 20% 94%)",
  text2: "hsl(220 12% 65%)",
  text3: "hsl(220 10% 42%)",
  text4: "hsl(220 8% 28%)",
  accent: "hsl(258 90% 70%)",
  accentDim: "hsl(258 60% 55%)",
  green: "hsl(142 71% 52%)",
  greenDim: "hsl(142 50% 40%)",
  amber: "hsl(38 92% 58%)",
  red: "hsl(346 87% 65%)",
  blue: "hsl(199 89% 60%)",
  purple: "hsl(262 83% 70%)",
  vizDefault: "hsl(220 14% 35%)",
  vizCompare: "hsl(38 92% 58%)",
  vizSwap: "hsl(346 87% 65%)",
  vizPivot: "hsl(262 83% 70%)",
  vizSorted: "hsl(142 71% 52%)",
  vizFound: "hsl(142 71% 52%)",
} as const;
