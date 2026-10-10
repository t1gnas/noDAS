"use client";

import { Car, CircleDot, Construction, Fence, Footprints, LucideIcon, Zap } from "lucide-react";
import { Scenario } from "@/lib/api";

const ICONS: Record<string, [LucideIcon, string]> = {
  car: [Car, "#FF9F0A"],
  walk: [Footprints, "#0A84FF"],
  fence: [Fence, "#FF453A"],
  construction: [Construction, "#FFB800"],
  running: [Zap, "#5E5CE6"],
  manhole: [CircleDot, "#FF375F"],
};

type Props = { scenarios: Scenario[]; selected: string | null; onSelect: (id: string) => void };

export function ScenarioPicker({ scenarios, selected, onSelect }: Props) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-3">
      {scenarios.map((s) => {
        const [Icon, color] = ICONS[s.icon] ?? [CircleDot, "#8E8E93"];
        const active = s.id === selected;
        return (
          <button
            key={s.id}
            onClick={() => onSelect(s.id)}
            className={`card text-left p-4 transition-all duration-300 hover:-translate-y-0.5 active:scale-[0.98] ${
              active ? "ring-[2.5px] ring-blue" : "ring-0"
            }`}
          >
            <span className="grid place-items-center size-10 rounded-[12px] text-white mb-3" style={{ background: color }}>
              <Icon size={20} strokeWidth={2.2} />
            </span>
            <div className="font-semibold text-[15px] leading-tight">{s.title}</div>
            <div className="text-[12px] text-ink-2 mt-1 leading-snug line-clamp-2">{s.subtitle}</div>
            <div className="text-[11px] text-ink-3 mt-2 tabular">{Math.round(s.duration_s)} s recording</div>
          </button>
        );
      })}
    </div>
  );
}
