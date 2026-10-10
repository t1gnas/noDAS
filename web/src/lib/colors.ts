type RGB = [number, number, number];

const STOPS: [number, RGB][] = [
  [0, [11, 11, 16]],
  [0.35, [22, 44, 84]],
  [0.65, [40, 132, 214]],
  [0.85, [120, 210, 250]],
  [1, [255, 255, 255]],
];

function lerp(a: RGB, b: RGB, t: number): RGB {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

function sample(v: number): RGB {
  for (let i = 1; i < STOPS.length; i++) {
    const [p1, c1] = STOPS[i];
    const [p0, c0] = STOPS[i - 1];
    if (v <= p1) return lerp(c0, c1, (v - p0) / (p1 - p0));
  }
  return STOPS[STOPS.length - 1][1];
}

export const COLORMAP: RGB[] = Array.from({ length: 256 }, (_, i) => sample(i / 255));

export function hexToRgb(hex: string): RGB {
  const n = parseInt(hex.replace("#", ""), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

export function decode(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}
