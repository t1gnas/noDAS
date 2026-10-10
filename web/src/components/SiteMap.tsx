import { memo } from "react";

type Box = [number, number, number, number];

const BORDER = "M-200 236 C 120 226, 260 248, 450 236 S 760 224, 1200 240";

const TREES: [number, number, number][] = Array.from({ length: 260 }, (_, i) => {
  const x = ((i * 137.5) % 1100) - 50;
  const y = 10 + ((i * 61.7) % 205);
  const r = 7 + ((i * 13) % 9);
  return [x, y, r];
});

const BUILDINGS: Box[] = [
  [64, 326, 88, 58],
  [108, 392, 64, 50],
  [168, 650, 64, 34],
  [252, 652, 46, 30],
  [560, 648, 78, 34],
  [662, 652, 40, 28],
  [990, 330, 60, 70],
];

const FIELDS: [Box, string][] = [
  [[180, 330, 300, 240], "#e7efd6"],
  [[500, 330, 410, 250], "#eef2e0"],
  [[180, 590, 290, 18], "#eef2e0"],
];

function Building({ b }: { b: Box }) {
  const [x, y, w, h] = b;
  return (
    <g>
      <rect x={x + 3} y={y + 4} width={w} height={h} rx={5} fill="#d8d3c9" />
      <rect x={x} y={y} width={w} height={h} rx={5} fill="#e7e3db" stroke="#d9d4ca" strokeWidth={1} />
    </g>
  );
}

function Road({ d, w }: { d: string; w: number }) {
  return (
    <g fill="none" strokeLinecap="round" strokeLinejoin="round">
      <path d={d} stroke="#e1ddd4" strokeWidth={w + 4} />
      <path d={d} stroke="#ffffff" strokeWidth={w} />
    </g>
  );
}

function Label({ x, y, text, size = 10, color = "#8e8e93", spacing = 0.4, weight = 500 }: { x: number; y: number; text: string; size?: number; color?: string; spacing?: number; weight?: number }) {
  return (
    <text x={x} y={y} fontSize={size} fill={color} fontWeight={weight} textAnchor="middle" style={{ letterSpacing: spacing }}>
      {text}
    </text>
  );
}

function Field({ b, color }: { b: Box; color: string }) {
  const [x, y, w, h] = b;
  const rows = Array.from({ length: Math.floor(h / 16) }, (_, i) => y + 8 + i * 16);
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={14} fill={color} />
      {rows.map((ry) => (
        <line key={ry} x1={x + 10} x2={x + w - 10} y1={ry} y2={ry} stroke="#ffffff" strokeOpacity={0.45} strokeWidth={2} />
      ))}
    </g>
  );
}

function Fence() {
  const posts = Array.from({ length: 101 }, (_, i) => i * 10);
  return (
    <g>
      <rect x={-200} y={240} width={1400} height={28} fill="#f4e7e4" opacity={0.8} />
      <line x1={-200} x2={1200} y1={268} y2={268} stroke="#7c7c80" strokeWidth={1.6} />
      {posts.map((x) => (
        <line key={x} x1={x} x2={x} y1={264} y2={272} stroke="#7c7c80" strokeWidth={1} />
      ))}
    </g>
  );
}

function Tower({ x }: { x: number }) {
  return (
    <g>
      <rect x={x - 6} y={248} width={12} height={12} rx={3} fill="#ffffff" stroke="#8e8e93" strokeWidth={1.2} />
      <circle cx={x} cy={254} r={2} fill="#ff9f0a" />
    </g>
  );
}

function Checkpoint() {
  return (
    <g>
      <rect x={14} y={312} width={54} height={40} rx={4} fill="#ffffff" stroke="#d9d4ca" />
      <line x1={22} x2={58} y1={262} y2={262} stroke="#ff3b30" strokeWidth={3} strokeDasharray="6 4" />
      <line x1={22} x2={58} y1={306} y2={306} stroke="#ff3b30" strokeWidth={3} strokeDasharray="6 4" />
    </g>
  );
}

function SiteMapBase() {
  return (
    <g>
      <rect x={-2000} y={-2000} width={5000} height={5000} fill="#f3f1ec" />
      <path d={`${BORDER} L1200 -2000 L-200 -2000 Z`} fill="#d9ead0" />
      {TREES.map(([x, y, r]) => (
        <circle key={`${x}-${y}`} cx={x} cy={y} r={r} fill="#c4ddb3" />
      ))}
      <Label x={520} y={120} text="NEIGHBOURING COUNTRY" size={15} color="#7f9a72" spacing={4} weight={600} />
      {FIELDS.map(([b, c]) => (
        <Field key={b.join()} b={b} color={c} />
      ))}
      <Fence />
      <path d={BORDER} fill="none" stroke="#9b8fc7" strokeWidth={2.5} strokeDasharray="10 6" />
      <Label x={640} y={222} text="STATE BORDER" size={9} color="#8a7fb8" spacing={1.5} weight={600} />
      <Label x={360} y={259} text="restricted zone" size={8} color="#c0928a" />
      <Road d="M-200 300 H1200" w={10} />
      <Road d="M40 -200 V900" w={16} />
      <Road d="M962 300 V900" w={12} />
      <Road d="M-200 632 H1200" w={12} />
      <Checkpoint />
      {[260, 600, 880].map((x) => (
        <Tower key={x} x={x} />
      ))}
      {BUILDINGS.map((b) => (
        <Building key={b.join()} b={b} />
      ))}
      <Label x={700} y={316} text="PATROL ROAD" size={9} spacing={1.2} />
      <Label x={116} y={462} text="Border checkpoint" size={11} color="#6e6e73" />
      <Label x={330} y={460} text="Farmland" size={11} />
    </g>
  );
}

export const SiteMap = memo(SiteMapBase);
