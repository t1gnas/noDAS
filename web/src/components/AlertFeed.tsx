"use client";

import { Bell, CircleCheck } from "lucide-react";
import { AlertItem } from "@/lib/useLive";

function Banner({ a }: { a: AlertItem }) {
  const mismatch = a.verified && a.truth_label && a.truth_label !== a.label;
  return (
    <li
      className={`slide-in rounded-[20px] px-3.5 py-3 flex gap-3 items-start border border-line transition-opacity duration-700 ${
        a.open ? "bg-white shadow-[0_4px_18px_rgba(0,0,0,0.08)]" : "bg-white/60 opacity-60"
      }`}
    >
      <span className="grid place-items-center size-9 shrink-0 rounded-[10px] text-white" style={{ background: a.color }}>
        <Bell size={17} strokeWidth={2.4} />
      </span>
      <div className="min-w-0 flex-1">
        <div className="flex items-center justify-between gap-2">
          <span className="font-semibold text-[14px] truncate">{a.label}</span>
          <span className="text-[11px] text-ink-3 tabular shrink-0">{a.opened_t.toFixed(1)} s</span>
        </div>
        <div className="text-[12.5px] text-ink-2 tabular">
          {Math.round(a.pos_m)} m along fiber · {Math.round(a.confidence * 100)}% confident
        </div>
        <div className="mt-1 flex items-center gap-1.5 text-[11.5px]">
          {a.verified ? (
            <span className="inline-flex items-center gap-1 text-green font-medium">
              <CircleCheck size={13} strokeWidth={2.6} />
              {mismatch ? `Annotated as ${a.truth_label}` : "Verified by annotation"}
            </span>
          ) : (
            <span className="text-ink-3">No annotation here · model guess: {a.model_label}</span>
          )}
          {!a.open && <span className="text-ink-3">· cleared after {a.duration_s.toFixed(1)} s</span>}
        </div>
      </div>
    </li>
  );
}

export function AlertFeed({ alerts }: { alerts: AlertItem[] }) {
  if (!alerts.length)
    return (
      <div className="h-full grid place-items-center text-center text-ink-3 text-[13px] px-6">
        Alerts appear here the moment the fiber hears something.
      </div>
    );
  return (
    <ul className="flex flex-col gap-2.5">
      {alerts.slice(0, 40).map((a) => (
        <Banner key={a.id} a={a} />
      ))}
    </ul>
  );
}
