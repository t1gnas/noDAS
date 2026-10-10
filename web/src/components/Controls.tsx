"use client";

import { Pause, Play, RotateCcw, Sparkles } from "lucide-react";
import { Status } from "@/lib/useLive";

type Props = {
  status: Status;
  speed: number;
  speeds: number[];
  showAi: boolean;
  disabled: boolean;
  onPlay: () => void;
  onPause: () => void;
  onSpeed: (s: number) => void;
  onToggleAi: () => void;
};

function Segmented({ value, options, onChange }: { value: number; options: number[]; onChange: (v: number) => void }) {
  return (
    <div className="flex p-1 rounded-full bg-black/[0.05]">
      {options.map((o) => (
        <button
          key={o}
          onClick={() => onChange(o)}
          className={`px-3 py-1 rounded-full text-[12.5px] font-medium tabular transition-all ${
            o === value ? "bg-white shadow-sm text-ink" : "text-ink-2"
          }`}
        >
          {o}×
        </button>
      ))}
    </div>
  );
}

export function Controls(p: Props) {
  const playing = p.status === "playing" || p.status === "connecting";
  const MainIcon = playing ? Pause : p.status === "ended" ? RotateCcw : Play;
  return (
    <div className="flex flex-wrap items-center gap-3">
      <button
        disabled={p.disabled}
        onClick={playing ? p.onPause : p.onPlay}
        className="grid place-items-center size-12 rounded-full bg-blue text-white shadow-[0_6px_20px_rgba(0,113,227,0.35)] transition-transform active:scale-95 disabled:opacity-40"
        aria-label={playing ? "Pause" : "Play"}
      >
        <MainIcon size={22} fill={MainIcon === RotateCcw ? "none" : "currentColor"} strokeWidth={MainIcon === RotateCcw ? 2.4 : 0} />
      </button>
      <Segmented value={p.speed} options={p.speeds} onChange={p.onSpeed} />
      <button
        onClick={p.onToggleAi}
        className={`flex items-center gap-1.5 px-3.5 py-2 rounded-full text-[12.5px] font-medium transition-all ${
          p.showAi ? "bg-[#5e5ce6] text-white shadow-[0_4px_14px_rgba(94,92,230,0.35)]" : "bg-black/[0.05] text-ink-2"
        }`}
      >
        <Sparkles size={14} strokeWidth={2.4} />
        AI view {p.showAi ? "on" : "off"}
      </button>
    </div>
  );
}
