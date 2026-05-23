/**
 * Syntax highlighter — uses inline styles instead of CSS classes
 * so colors always render correctly regardless of CSS loading order.
 */

type Lang = "cpp" | "java" | "python";

const COLORS = {
  keyword:  "hsl(262 83% 72%)",
  function: "hsl(199 89% 65%)",
  string:   "hsl(142 71% 55%)",
  number:   "hsl(38 92% 62%)",
  comment:  "hsl(220 10% 50%)",
  type:     "hsl(346 87% 68%)",
};

const KEYWORDS: Record<Lang, string[]> = {
  cpp: [
    "void","int","bool","long","short","unsigned","const","static",
    "return","if","else","while","for","do","break","continue",
    "auto","vector","string","pair","map","set","queue","stack","deque","priority_queue",
    "true","false","nullptr","new","delete","class","struct","public","private","protected",
    "include","using","namespace","std","template","typename","size_t",
  ],
  java: [
    "public","private","protected","static","final","abstract","synchronized",
    "void","int","long","double","boolean","char","byte","short",
    "String","Integer","Long","Double","Boolean",
    "return","if","else","while","for","do","break","continue","new",
    "class","interface","extends","implements","import","package",
    "true","false","null","this","super","instanceof",
    "List","ArrayList","HashMap","HashSet","Queue","LinkedList","Deque","ArrayDeque",
    "Arrays","Collections","Math","System","StringBuilder",
  ],
  python: [
    "def","return","if","elif","else","while","for","in","not","and","or","is",
    "import","from","as","class","pass","break","continue","yield","with",
    "True","False","None","self","lambda","global","nonlocal","del","raise","try","except","finally",
    "len","range","print","input","int","str","float","list","dict","set","tuple","enumerate","zip",
    "append","extend","pop","insert","remove","sort","sorted","reversed","min","max","sum","abs",
    "heapq","collections","deque","defaultdict","Counter",
  ],
};

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function span(color: string, content: string): string {
  return `<span style="color:${color}">${content}</span>`;
}

export function highlightCode(rawCode: string, lang: Lang): string {
  // Process line by line to handle comments correctly
  const lines = rawCode.split("\n");
  const kws = KEYWORDS[lang];
  const kwSet = new Set(kws);

  const processed = lines.map(line => {
    // Find comment start (but not inside strings)
    let commentStart = -1;
    let inStr = false;
    let strChar = "";
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (inStr) {
        if (ch === strChar && line[i-1] !== "\\") inStr = false;
      } else if (ch === '"' || ch === "'") {
        inStr = true; strChar = ch;
      } else if (ch === "/" && line[i+1] === "/") {
        commentStart = i; break;
      } else if (ch === "#" && lang === "python") {
        commentStart = i; break;
      }
    }

    let codePart = commentStart >= 0 ? line.slice(0, commentStart) : line;
    const commentPart = commentStart >= 0 ? line.slice(commentStart) : "";

    // Tokenize the code part
    const result: string[] = [];
    let i = 0;
    while (i < codePart.length) {
      // String literals
      if (codePart[i] === '"' || codePart[i] === "'") {
        const q = codePart[i];
        let j = i + 1;
        while (j < codePart.length && !(codePart[j] === q && codePart[j-1] !== "\\")) j++;
        result.push(span(COLORS.string, escapeHtml(codePart.slice(i, j+1))));
        i = j + 1;
        continue;
      }

      // Numbers
      if (/[0-9]/.test(codePart[i]) && (i === 0 || !/[a-zA-Z_]/.test(codePart[i-1]))) {
        let j = i;
        while (j < codePart.length && /[0-9._xXa-fA-FlLuU]/.test(codePart[j])) j++;
        result.push(span(COLORS.number, escapeHtml(codePart.slice(i, j))));
        i = j;
        continue;
      }

      // Identifiers / keywords
      if (/[a-zA-Z_]/.test(codePart[i])) {
        let j = i;
        while (j < codePart.length && /[a-zA-Z0-9_]/.test(codePart[j])) j++;
        const word = codePart.slice(i, j);
        // Check if next non-space is '('  → function
        let k = j;
        while (k < codePart.length && codePart[k] === " ") k++;
        if (codePart[k] === "(" && !kwSet.has(word)) {
          result.push(span(COLORS.function, escapeHtml(word)));
        } else if (kwSet.has(word)) {
          result.push(span(COLORS.keyword, escapeHtml(word)));
        } else {
          result.push(escapeHtml(word));
        }
        i = j;
        continue;
      }

      result.push(escapeHtml(codePart[i]));
      i++;
    }

    const commentHtml = commentPart
      ? span(COLORS.comment, escapeHtml(commentPart))
      : "";

    return result.join("") + commentHtml;
  });

  return processed.join("\n");
}
