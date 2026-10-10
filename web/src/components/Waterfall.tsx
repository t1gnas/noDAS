"use client";

import { Ref, useImperativeHandle, useRef } from "react";
import { Frame, Meta } from "@/lib/api";
import { COLORMAP, decode, hexToRgb } from "@/lib/colors";

export const WINDOW_SECONDS = 30;

export type WaterfallHandle = { reset: (meta: Meta) => void; push: (frame: Frame, meta: Meta) => void };

type Palette = Map<number, [number, number, number]>;

function shiftDown(ctx: CanvasRenderingContext2D, w: number, h: number, rows: number) {
  ctx.drawImage(ctx.canvas, 0, 0, w, h - rows, 0, rows, w, h - rows);
}

function paint(raw: ImageData, ai: ImageData, inten: Uint8Array, over: Uint8Array, palette: Palette, rows: number, w: number) {
  for (let i = 0; i < rows; i++) {
    const y = rows - 1 - i;
    for (let x = 0; x < w; x++) {
      const v = inten[i * w + x];
      const [r, g, b] = COLORMAP[v];
      const p = (y * w + x) * 4;
      raw.data.set([r, g, b, 255], p);
      const cls = palette.get(over[i * w + x]);
      const a = cls ? 0.82 : 0;
      ai.data.set(cls ? [r * (1 - a) + cls[0] * a, g * (1 - a) + cls[1] * a, b * (1 - a) + cls[2] * a, 255] : [r * 0.55, g * 0.55, b * 0.55, 255], p);
    }
  }
}

export function Waterfall({ ref, showAi }: { ref: Ref<WaterfallHandle>; showAi: boolean }) {
  const rawRef = useRef<HTMLCanvasElement>(null);
  const aiRef = useRef<HTMLCanvasElement>(null);
  const paletteRef = useRef<Palette>(new Map());

  useImperativeHandle(ref, () => ({
    reset(meta) {
      const h = Math.round(WINDOW_SECONDS / meta.row_seconds);
      paletteRef.current = new Map(meta.classes.filter((c) => c.id > 0).map((c) => [c.id, hexToRgb(c.color)]));
      for (const c of [rawRef.current, aiRef.current]) {
        if (!c) continue;
        c.width = meta.width;
        c.height = h;
        const ctx = c.getContext("2d")!;
        ctx.fillStyle = "#0b0b10";
        ctx.fillRect(0, 0, c.width, c.height);
      }
    },
    push(frame, meta) {
      const raw = rawRef.current?.getContext("2d");
      const ai = aiRef.current?.getContext("2d");
      if (!raw || !ai || !frame.rows || !frame.intensity || !frame.overlay) return;
      const w = meta.width;
      const h = raw.canvas.height;
      shiftDown(raw, w, h, frame.rows);
      shiftDown(ai, w, h, frame.rows);
      const rawImg = raw.createImageData(w, frame.rows);
      const aiImg = ai.createImageData(w, frame.rows);
      paint(rawImg, aiImg, decode(frame.intensity), decode(frame.overlay), paletteRef.current, frame.rows, w);
      raw.putImageData(rawImg, 0, 0);
      ai.putImageData(aiImg, 0, 0);
    },
  }));

  return (
    <div className="relative w-full h-full">
      <canvas ref={rawRef} className="absolute inset-0 w-full h-full" />
      <canvas
        ref={aiRef}
        className="absolute inset-0 w-full h-full transition-opacity duration-500"
        style={{ opacity: showAi ? 1 : 0 }}
      />
    </div>
  );
}
