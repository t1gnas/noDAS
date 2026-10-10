export type Pt = [number, number];

export const SITE = { width: 1000, height: 680 };

export const ROUTE: Pt[] = [
  [92, 400],
  [92, 282],
  [940, 282],
  [940, 610],
  [499, 610],
];

const ROAD_SIDE: Pt[] = [
  [0, 0],
  [0, 1],
  [1, 0],
  [0, 1],
];

export const INTERROGATOR: Pt = ROUTE[0];

const SEGMENTS = ROUTE.slice(1).map((p, i) => {
  const a = ROUTE[i];
  return { a, b: p, len: Math.hypot(p[0] - a[0], p[1] - a[1]), side: ROAD_SIDE[i] };
});

export const ROUTE_LENGTH = SEGMENTS.reduce((s, g) => s + g.len, 0);

function locate(fraction: number) {
  let d = Math.min(Math.max(fraction, 0), 1) * ROUTE_LENGTH;
  for (const g of SEGMENTS) {
    if (d <= g.len) return { g, t: d / g.len };
    d -= g.len;
  }
  const g = SEGMENTS[SEGMENTS.length - 1];
  return { g, t: 1 };
}

export function pointAt(fraction: number): Pt {
  const { g, t } = locate(fraction);
  return [g.a[0] + (g.b[0] - g.a[0]) * t, g.a[1] + (g.b[1] - g.a[1]) * t];
}

export function besideAt(fraction: number, offset: number): Pt {
  const [x, y] = pointAt(fraction);
  const { g } = locate(fraction);
  return [x + g.side[0] * offset, y + g.side[1] * offset];
}

export function subPath(f0: number, f1: number, steps = 60): string {
  const pts = Array.from({ length: steps + 1 }, (_, i) => pointAt(f0 + ((f1 - f0) * i) / steps));
  return pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");
}

export const ROUTE_PATH = ROUTE.map((p, i) => `${i ? "L" : "M"}${p[0]},${p[1]}`).join(" ");
