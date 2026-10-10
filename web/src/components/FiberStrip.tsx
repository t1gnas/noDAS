"use client";

import { AlertItem } from "@/lib/useLive";

type Props = { start: number; end: number; viewStart: number; viewEnd: number; alerts: AlertItem[]; elapsed: number };

const LINGER_S = 2.5;
const LINE_Y = 62;

function km(m: number) {
  return m >= 1000 ? `${(m / 1000).toFixed(1)} km` : `${Math.round(m)} m`;
}

function VerifiedPin({ a, left, row }: { a: AlertItem; left: string; row: number }) {
  const top = row ? 20 : 0;
  return (
    <div
      className="absolute transition-[left,opacity] duration-700 ease-out"
      style={{ left, top: 0, height: LINE_Y + 8, opacity: a.open ? 1 : 0 }}
    >
      <div className="absolute -translate-x-1/2 flex flex-col items-center" style={{ top }}>
        <span className="slide-in px-2.5 py-1 rounded-full text-white text-[11px] font-semibold whitespace-nowrap shadow-[0_3px_10px_rgba(0,0,0,0.18)]" style={{ background: a.color }}>
          {a.label} · {Math.round(a.pos_m)} m
        </span>
        <span className="w-[2px] rounded-full" style={{ height: LINE_Y - top - 30, background: a.color }} />
      </div>
      <span
        className="absolute -translate-x-1/2 -translate-y-1/2 size-3.5 rounded-full border-[2.5px] border-white shadow"
        style={{ top: LINE_Y, background: a.color }}
      />
    </div>
  );
}

function UnverifiedDot({ left }: { left: string }) {
  return (
    <span
      className="absolute -translate-x-1/2 -translate-y-1/2 size-2.5 rounded-full border-2 border-[#8e8e93] bg-white/80 transition-[left] duration-700"
      style={{ left, top: LINE_Y }}
    />
  );
}

export function FiberStrip({ start, end, viewStart, viewEnd, alerts, elapsed }: Props) {
  const pct = (m: number) => `${(100 * (m - start)) / Math.max(end - start, 1)}%`;
  const verified = alerts.filter((a) => a.verified && (a.open || elapsed - a.t < LINGER_S)).sort((x, y) => x.pos_m - y.pos_m);
  const unverified = alerts.filter((a) => !a.verified && a.open);
  return (
    <div className="w-full">
      <div className="relative w-full" style={{ height: LINE_Y + 14 }}>
        <div className="absolute inset-x-0 h-1.5 rounded-full bg-[#d2d2d7] -translate-y-1/2" style={{ top: LINE_Y }} />
        <div
          className="absolute h-1.5 rounded-full bg-blue/80 -translate-y-1/2"
          style={{ top: LINE_Y, left: pct(viewStart), width: `calc(${pct(viewEnd)} - ${pct(viewStart)})` }}
        />
        {unverified.map((a) => (
          <UnverifiedDot key={a.id} left={pct(a.pos_m)} />
        ))}
        {verified.map((a, i) => (
          <VerifiedPin key={a.id} a={a} left={pct(a.pos_m)} row={i % 2} />
        ))}
      </div>
      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1 text-[11px] text-ink-3 tabular">
        <span>{km(start)}</span>
        <span className="flex items-center gap-1.5">
          <span className="size-2.5 rounded-full bg-[#0a84ff] border-2 border-white shadow" />
          Detection confirmed by annotation
        </span>
        <span className="flex items-center gap-1.5">
          <span className="size-2.5 rounded-full border-2 border-[#8e8e93] bg-white" />
          Vibration on unlabelled cable (unverified)
        </span>
        <span className="text-blue font-medium">
          Monitored {km(viewStart)} – {km(viewEnd)}
        </span>
        <span>{km(end)}</span>
      </div>
    </div>
  );
}
