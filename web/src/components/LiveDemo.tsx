"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { ClassInfo, getJson, ScenarioList } from "@/lib/api";
import { AlertItem, useLive } from "@/lib/useLive";
import { AlertFeed } from "./AlertFeed";
import { Controls } from "./Controls";
import { FiberStrip } from "./FiberStrip";
import { ScenarioPicker } from "./ScenarioPicker";
import { StatTile } from "./StatTile";
import { WINDOW_SECONDS, Waterfall, WaterfallHandle } from "./Waterfall";

function ticks(a: number, b: number, n = 5) {
  return Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1));
}

function counts(alerts: AlertItem[]) {
  const verified = alerts.filter((a) => a.verified);
  const correct = verified.filter((a) => !a.truth_label || a.truth_label === a.label);
  return { total: alerts.length, verified: verified.length, correct: correct.length, active: alerts.filter((a) => a.open).length };
}

function Legend({ classes }: { classes: ClassInfo[] }) {
  return (
    <div className="flex flex-wrap gap-2">
      {classes
        .filter((c) => c.id > 0)
        .map((c) => (
          <span key={c.id} className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-black/[0.04] text-[11.5px] text-ink-2">
            <span className="size-2.5 rounded-full" style={{ background: c.color }} />
            {c.label}
          </span>
        ))}
    </div>
  );
}

export function LiveDemo() {
  const [list, setList] = useState<ScenarioList | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [showAi, setShowAi] = useState(true);
  const fall = useRef<WaterfallHandle>(null);
  const live = useLive({ onMeta: (m) => fall.current?.reset(m), onFrame: (f, m) => fall.current?.push(f, m) });

  useEffect(() => {
    getJson<ScenarioList>("/api/scenarios").then(setList, (e: Error) => setLoadError(e.message));
  }, []);

  const scenario = list?.scenarios.find((s) => s.id === selected) ?? null;
  const stats = useMemo(() => counts(live.alerts), [live.alerts]);
  const duration = live.meta?.duration_s ?? scenario?.duration_s ?? 0;
  const play = (id: string) => {
    setSelected(id);
    live.start(id);
  };
  const onPlay = () => (live.status === "paused" ? live.resume() : selected && live.start(selected));

  return (
    <div className="flex flex-col gap-8 pt-10">
      <section className="text-center max-w-3xl mx-auto">
        <h1 className="text-[40px] sm:text-[56px] font-semibold tracking-[-0.03em] leading-[1.05]">Your fiber can hear.</h1>
        <p className="mt-4 text-[17px] sm:text-[19px] text-ink-2 leading-relaxed">
          Every metre of an ordinary telecom cable becomes a microphone. Our model listens in real time and tells you what is happening, and where.
        </p>
      </section>

      {loadError && <div className="card p-5 text-[14px] text-[#d70015]">Backend unavailable: {loadError}</div>}
      {list && <ScenarioPicker scenarios={list.scenarios} selected={selected} onSelect={play} />}

      <section className="card p-5 sm:p-7 flex flex-col gap-5">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-[12px] font-medium text-ink-3 uppercase tracking-wide">
              {live.status === "playing" && <span className="live-dot size-2 rounded-full bg-[#ff3b30]" />}
              {live.status === "playing" ? "Live replay" : live.status === "paused" ? "Paused" : live.status === "ended" ? "Replay finished" : "Ready"}
            </div>
            <h2 className="text-[24px] font-semibold tracking-tight">{scenario?.title ?? "Choose a scenario above"}</h2>
          </div>
          <Controls
            status={live.status}
            speed={live.speed}
            speeds={list?.speeds ?? [1, 2, 5, 10]}
            showAi={showAi}
            disabled={!selected}
            onPlay={onPlay}
            onPause={live.pause}
            onSpeed={live.setSpeed}
            onToggleAi={() => setShowAi((v) => !v)}
          />
        </div>

        <div className="flex items-center gap-3 text-[12px] text-ink-3 tabular">
          <span className="w-12">{live.elapsed.toFixed(1)} s</span>
          <div className="flex-1 h-1.5 rounded-full bg-black/[0.06] overflow-hidden">
            <div className="h-full bg-blue rounded-full transition-[width] duration-300" style={{ width: `${duration ? (100 * live.elapsed) / duration : 0}%` }} />
          </div>
          <span className="w-12 text-right">{duration.toFixed(0)} s</span>
        </div>

        {scenario && (
          <FiberStrip start={scenario.fiber_start_m} end={scenario.fiber_end_m} viewStart={scenario.view_start_m} viewEnd={scenario.view_end_m} alerts={live.alerts} />
        )}

        <div className="grid lg:grid-cols-[1fr_380px] gap-5">
          <div className="flex flex-col gap-2">
            <div className="relative h-[440px] rounded-[22px] overflow-hidden bg-screen">
              <Waterfall ref={fall} showAi={showAi} />
              {!selected && (
                <div className="absolute inset-0 grid place-items-center text-white/50 text-[14px]">Pick a scenario to start listening</div>
              )}
              <div className="absolute left-3 top-2 bottom-2 flex flex-col justify-between text-[10.5px] text-white/55 tabular pointer-events-none">
                <span>now</span>
                <span>−{WINDOW_SECONDS / 2} s</span>
                <span>−{WINDOW_SECONDS} s</span>
              </div>
              <div className="absolute right-3 top-2.5 px-2.5 py-1 rounded-full bg-white/10 backdrop-blur text-[10.5px] text-white/80 pointer-events-none">
                {showAi ? "AI classification" : "Raw vibration energy"}
              </div>
            </div>
            {scenario && (
              <div className="flex justify-between text-[11px] text-ink-3 tabular px-1">
                {ticks(scenario.view_start_m, scenario.view_end_m).map((m) => (
                  <span key={m}>{Math.round(m)} m</span>
                ))}
              </div>
            )}
          </div>
          <div className="h-[440px] overflow-y-auto rounded-[22px] bg-bg p-3">
            <AlertFeed alerts={live.alerts} />
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <StatTile label="Alerts raised" value={stats.total} hint={`${stats.active} active now`} />
          <StatTile label="Verified by annotation" value={stats.verified} hint="overlap a labelled event" />
          <StatTile
            label="Correct class"
            value={stats.verified ? `${Math.round((100 * stats.correct) / stats.verified)}%` : "–"}
            hint={`${stats.correct} of ${stats.verified} verified alerts`}
            accent
          />
          <StatTile label="Unverified" value={stats.total - stats.verified} hint="activity on unlabelled cable" />
        </div>

        {live.error && <div className="text-[13px] text-[#d70015]">{live.error}</div>}
        {list && <Legend classes={list.classes} />}
      </section>
    </div>
  );
}
