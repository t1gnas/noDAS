"use client";

import { AlertItem } from "@/lib/useLive";

type Props = { start: number; end: number; viewStart: number; viewEnd: number; alerts: AlertItem[] };

const W = 1000;

function km(m: number) {
  return m >= 1000 ? `${(m / 1000).toFixed(1)} km` : `${Math.round(m)} m`;
}

export function FiberStrip({ start, end, viewStart, viewEnd, alerts }: Props) {
  const x = (m: number) => ((m - start) / Math.max(end - start, 1)) * W;
  const open = alerts.filter((a) => a.open);
  return (
    <div className="w-full">
      <svg viewBox={`0 0 ${W} 44`} className="w-full h-11 overflow-visible">
        <line x1={0} x2={W} y1={22} y2={22} stroke="#d2d2d7" strokeWidth={6} strokeLinecap="round" />
        <line x1={x(viewStart)} x2={x(viewEnd)} y1={22} y2={22} stroke="#0071e3" strokeWidth={6} strokeLinecap="round" />
        {open.map((a) => (
          <g key={a.id}>
            <circle cx={x(a.pos_m)} cy={22} r={6} fill={a.color} className="pulse-ring" />
            <circle cx={x(a.pos_m)} cy={22} r={6} fill={a.color} stroke="white" strokeWidth={2} />
          </g>
        ))}
      </svg>
      <div className="flex justify-between text-[11px] text-ink-3 tabular -mt-1">
        <span>{km(start)}</span>
        <span className="text-blue font-medium">
          Monitored section {km(viewStart)} – {km(viewEnd)}
        </span>
        <span>{km(end)}</span>
      </div>
    </div>
  );
}
