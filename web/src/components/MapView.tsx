"use client";

import { useEffect, useRef, useState } from "react";
import { AudioWaveform, Car, Footprints, LocateFixed, LucideIcon, Map as MapIcon, Pause, PersonStanding, Play, RotateCcw } from "lucide-react";
import { Actor, getJson, Results, Scenario, ScenarioList } from "@/lib/api";
import { INTERROGATOR, Pt, ROUTE_PATH, SITE, besideAt, pointAt, subPath } from "@/lib/route";
import { AlertItem, useLive } from "@/lib/useLive";
import { AlertFeed } from "./AlertFeed";
import { SiteMap } from "./SiteMap";

const MAP_SCENARIOS = ["vehicle", "walking"];
const FOLLOW_WIDTH_M = 160;
const MIN_HALO_PX = 7;

type View = { cx: number; cy: number; w: number };

function useSize() {
  const ref = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 1000, h: 620 });
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => {
      const { width, height } = e.contentRect;
      if (width > 0 && height > 0) setSize({ w: width, h: height });
    });
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return { ref, size };
}

function overview(size: { w: number; h: number }): View {
  const w = Math.max(SITE.width + 40, ((SITE.height + 40) * size.w) / size.h);
  return { cx: SITE.width / 2, cy: SITE.height / 2, w };
}

function useCamera(target: View) {
  const [view, setView] = useState(target);
  const targetRef = useRef(target);
  useEffect(() => {
    targetRef.current = target;
  });
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      setView((v) => {
        const t = targetRef.current;
        const k = 0.1;
        const next = { cx: v.cx + (t.cx - v.cx) * k, cy: v.cy + (t.cy - v.cy) * k, w: v.w * Math.pow(t.w / v.w, k) };
        return Math.abs(next.w - v.w) < 0.01 && Math.abs(next.cx - v.cx) < 0.01 && Math.abs(next.cy - v.cy) < 0.01 ? v : next;
      });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);
  return view;
}

function fraction(s: Scenario, m: number) {
  return (m - s.fiber_start_m) / Math.max(s.fiber_end_m - s.fiber_start_m, 1);
}

function isVehicle(a: Actor) {
  return a.label === "Vehicle";
}

function actorPoint(s: Scenario, a: Actor): Pt {
  return besideAt(fraction(s, a.pos_m), isVehicle(a) ? 18 : -7);
}

function Trail({ pts, color, scale }: { pts: Pt[]; color: string; scale: number }) {
  return (
    <g>
      {pts.slice(1).map((p, i) => (
        <line
          key={i}
          x1={pts[i][0]}
          y1={pts[i][1]}
          x2={p[0]}
          y2={p[1]}
          stroke={color}
          strokeOpacity={(0.55 * (i + 1)) / pts.length}
          strokeWidth={5 * scale}
          strokeLinecap="round"
        />
      ))}
    </g>
  );
}

function ActorMarker({ a, at, cable, scale, halo }: { a: Actor; at: Pt; cable: Pt | null; scale: number; halo: number }) {
  const Icon: LucideIcon = isVehicle(a) ? Car : PersonStanding;
  const tag = a.detected ? `Detected · ${Math.round((a.confidence ?? 0) * 100)}%` : "Not detected yet";
  const width = tag.length * 6.4 + 22;
  return (
    <g>
      {a.detected && cable && (
        <g>
          <line x1={at[0]} y1={at[1]} x2={cable[0]} y2={cable[1]} stroke={a.color} strokeWidth={1.5 * scale} strokeDasharray={`${3 * scale} ${3 * scale}`} />
          <circle cx={cable[0]} cy={cable[1]} r={Math.max(halo, MIN_HALO_PX * scale)} fill={a.color} fillOpacity={0.22} stroke={a.color} strokeWidth={1.2 * scale} />
        </g>
      )}
      <g style={{ transform: `translate(${at[0]}px, ${at[1]}px)`, transition: "transform 420ms linear" }}>
        <g transform={`scale(${scale})`}>
          {a.detected ? (
            <circle r={30} fill={a.color} fillOpacity={0.16} stroke={a.color} strokeWidth={2.5} className="detect-pulse" />
          ) : (
            <circle r={27} fill="none" stroke="#8e8e93" strokeWidth={2} strokeDasharray="5 5" className="spin-slow" />
          )}
          <g className={isVehicle(a) ? "drive-bob" : "walk-bob"}>
            <rect x={-17} y={-17} width={34} height={34} rx={11} fill="#ffffff" stroke={a.detected ? a.color : "#d1d1d6"} strokeWidth={2} style={{ filter: "drop-shadow(0 3px 6px rgba(0,0,0,0.22))" }} />
            <Icon x={-11} y={-11} width={22} height={22} color={a.detected ? a.color : "#6e6e73"} strokeWidth={2.4} />
          </g>
          <g transform="translate(0 44)">
            <rect x={-width / 2} y={-11} width={width} height={22} rx={11} fill={a.detected ? a.color : "#ffffff"} stroke={a.detected ? "none" : "#d1d1d6"} />
            <text textAnchor="middle" y={4} fontSize={11} fontWeight={600} fill={a.detected ? "#ffffff" : "#6e6e73"}>
              {tag}
            </text>
          </g>
        </g>
      </g>
    </g>
  );
}

function AlertDot({ a, at, scale, halo }: { a: AlertItem; at: Pt; scale: number; halo: number }) {
  const r = Math.max(halo, MIN_HALO_PX * scale);
  return a.verified ? (
    <circle cx={at[0]} cy={at[1]} r={r} fill={a.color} fillOpacity={0.25} stroke={a.color} strokeWidth={1.2 * scale} />
  ) : (
    <circle cx={at[0]} cy={at[1]} r={4 * scale} fill="#ffffff" stroke="#8e8e93" strokeWidth={1.8 * scale} opacity={0.75} />
  );
}

function ScaleBar({ mPerPx }: { mPerPx: number }) {
  const nice = [5, 10, 20, 50, 100, 200, 500].find((m) => m / mPerPx >= 60) ?? 500;
  return (
    <div className="flex flex-col gap-1">
      <div className="h-1.5 border-x-2 border-b-2 border-ink/70 rounded-b-sm" style={{ width: nice / mPerPx }} />
      <span className="text-[10.5px] font-medium text-ink-2">{nice} m</span>
    </div>
  );
}

function GlassButton({ onClick, active, children, label }: { onClick: () => void; active?: boolean; children: React.ReactNode; label: string }) {
  return (
    <button
      onClick={onClick}
      aria-label={label}
      className={`grid place-items-center size-10 rounded-[12px] shadow-[0_2px_10px_rgba(0,0,0,0.12)] transition-all active:scale-95 ${active ? "bg-blue text-white" : "glass text-ink"}`}
    >
      {children}
    </button>
  );
}

function ActorList({ actors }: { actors: Actor[] }) {
  if (!actors.length) return <div className="text-[13px] text-ink-3 py-3">Nobody near the fence right now.</div>;
  return (
    <ul className="flex flex-col gap-2">
      {actors.map((a) => (
        <li key={a.id} className="flex items-center gap-3 rounded-[16px] bg-bg px-3 py-2.5">
          <span className="grid place-items-center size-8 rounded-[10px] text-white" style={{ background: a.detected ? a.color : "#aeaeb2" }}>
            {isVehicle(a) ? <Car size={16} /> : <PersonStanding size={16} />}
          </span>
          <div className="flex-1 min-w-0">
            <div className="text-[13.5px] font-semibold">
              {isVehicle(a) ? "Vehicle" : "Person"} #{a.id}
            </div>
            <div className="text-[11.5px] text-ink-2 tabular">{Math.round(a.pos_m)} m along the fence</div>
          </div>
          <span className={`text-[11.5px] font-semibold ${a.detected ? "text-green" : "text-ink-3"}`}>
            {a.detected ? `Detected ${Math.round((a.confidence ?? 0) * 100)}%` : "Searching…"}
          </span>
        </li>
      ))}
    </ul>
  );
}

export function MapView() {
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [selected, setSelected] = useState("walking");
  const [halo, setHalo] = useState<number | null>(null);
  const [follow, setFollow] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const live = useLive({ onMeta: () => {}, onFrame: () => {} });
  const { ref, size } = useSize();

  useEffect(() => {
    getJson<ScenarioList>("/api/scenarios").then((l) => setScenarios(l.scenarios.filter((s) => MAP_SCENARIOS.includes(s.id))), (e: Error) => setLoadError(e.message));
    getJson<Results>("/api/results").then((r) => setHalo(r.test.cnn.events.loc_err_mean_m), () => setHalo(null));
  }, []);

  const scenario = scenarios.find((s) => s.id === selected) ?? null;
  const lead = [...live.actors].sort((x, y) => x.id - y.id)[0];
  const focus = follow && lead && scenario ? actorPoint(scenario, lead) : null;
  const view = useCamera(focus ? { cx: focus[0], cy: focus[1], w: FOLLOW_WIDTH_M } : overview(size));
  const vh = (view.w * size.h) / size.w;
  const mPerPx = view.w / size.w;
  const toPx = (p: Pt): Pt => [((p[0] - (view.cx - view.w / 2)) / view.w) * size.w, ((p[1] - (view.cy - vh / 2)) / vh) * size.h];
  const playing = live.status === "playing" || live.status === "connecting";
  const start = (id: string) => {
    setSelected(id);
    live.start(id);
  };
  const onMain = () => (playing ? live.pause() : live.status === "paused" ? live.resume() : start(selected));
  const MainIcon = playing ? Pause : live.status === "ended" ? RotateCcw : Play;
  const haloM = halo ?? 3;
  const linked = new Set(live.actors.filter((a) => a.detected).map((a) => a.alert_id));
  const looseAlerts = live.alerts.filter((a) => a.open && !linked.has(a.id));
  const [ix, iy] = toPx(INTERROGATOR);

  return (
    <div className="flex flex-col gap-6 pt-10">
      <section className="text-center max-w-3xl mx-auto">
        <h1 className="text-[40px] sm:text-[56px] font-semibold tracking-[-0.03em] leading-[1.05]">Guard the border with a cable.</h1>
        <p className="mt-4 text-[17px] sm:text-[19px] text-ink-2">
          A fiber along the fence hears every step and every engine. Watch the AI pick them up as they move, accurate to ±{haloM.toFixed(1)} m.
        </p>
      </section>

      {loadError && <div className="card p-5 text-[14px] text-[#d70015]">Backend unavailable: {loadError}</div>}

      <div className="grid lg:grid-cols-[1fr_360px] gap-5">
        <div ref={ref} className="card relative h-[620px] overflow-hidden !rounded-[28px]">
          <svg className="absolute inset-0 w-full h-full" viewBox={`${view.cx - view.w / 2} ${view.cy - vh / 2} ${view.w} ${vh}`}>
            <defs>
              <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation={3 * mPerPx} />
              </filter>
            </defs>
            <SiteMap />
            <path d={ROUTE_PATH} fill="none" stroke="#0a84ff" strokeOpacity={0.3} strokeWidth={2.5 * mPerPx} strokeLinecap="round" strokeLinejoin="round" />
            {scenario && (
              <g>
                <path d={subPath(fraction(scenario, scenario.view_start_m), fraction(scenario, scenario.view_end_m))} fill="none" stroke="#0a84ff" strokeWidth={8 * mPerPx} strokeLinecap="round" filter="url(#glow)" />
                <path d={subPath(fraction(scenario, scenario.view_start_m), fraction(scenario, scenario.view_end_m))} fill="none" stroke="#0a84ff" strokeWidth={3.5 * mPerPx} strokeLinecap="round" />
                <path
                  d={subPath(fraction(scenario, scenario.view_start_m), fraction(scenario, scenario.view_end_m))}
                  fill="none"
                  stroke="#ffffff"
                  strokeOpacity={0.9}
                  strokeWidth={1.4 * mPerPx}
                  strokeDasharray={`${6 * mPerPx} ${18 * mPerPx}`}
                  className="fiber-flow"
                  strokeLinecap="round"
                />
              </g>
            )}
            {scenario && looseAlerts.map((a) => <AlertDot key={a.id} a={a} at={pointAt(fraction(scenario, a.pos_m))} scale={mPerPx} halo={haloM} />)}
            {scenario &&
              live.actors.map((a) => (
                <Trail key={a.id} pts={(live.trails[a.id] ?? []).map((m) => actorPoint(scenario, { ...a, pos_m: m }))} color={a.color} scale={mPerPx} />
              ))}
            {scenario &&
              live.actors.map((a) => (
                <ActorMarker
                  key={a.id}
                  a={a}
                  at={actorPoint(scenario, a)}
                  cable={a.alert_pos_m !== undefined ? pointAt(fraction(scenario, a.alert_pos_m)) : null}
                  scale={mPerPx}
                  halo={haloM}
                />
              ))}
          </svg>

          <div className="absolute pointer-events-none" style={{ left: ix, top: iy }}>
            <div className="-translate-x-1/2 -translate-y-1/2 flex flex-col items-center">
              <span className="grid place-items-center size-8 rounded-[10px] bg-gradient-to-br from-[#34c759] to-[#30b0c7] text-white border-2 border-white shadow-md">
                <AudioWaveform size={16} strokeWidth={2.4} />
              </span>
              <span className="mt-1 px-2 py-0.5 rounded-full glass text-[10.5px] font-semibold whitespace-nowrap shadow-sm">DAS interrogator</span>
            </div>
          </div>

          <div className="absolute left-4 top-4 flex items-center gap-2">
            <div className="glass flex p-1 rounded-full shadow-[0_2px_10px_rgba(0,0,0,0.12)]">
              {scenarios.map((s) => (
                <button
                  key={s.id}
                  onClick={() => start(s.id)}
                  className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-full text-[13px] font-medium transition-all ${s.id === selected ? "bg-white shadow-sm text-ink" : "text-ink-2"}`}
                >
                  {s.icon === "car" ? <Car size={14} /> : <Footprints size={14} />}
                  {s.icon === "car" ? "Vehicle on patrol road" : "Person at the fence"}
                </button>
              ))}
            </div>
            <button onClick={onMain} aria-label="Play or pause" className="grid place-items-center size-10 rounded-full bg-blue text-white shadow-[0_4px_14px_rgba(0,113,227,0.4)] active:scale-95 transition-transform">
              <MainIcon size={18} fill={MainIcon === RotateCcw ? "none" : "currentColor"} strokeWidth={MainIcon === RotateCcw ? 2.4 : 0} />
            </button>
          </div>

          <div className="absolute right-4 top-4 flex flex-col gap-2">
            <GlassButton label="Follow the intruder" active={follow} onClick={() => setFollow(true)}>
              <LocateFixed size={18} />
            </GlassButton>
            <GlassButton label="Show whole border" active={!follow} onClick={() => setFollow(false)}>
              <MapIcon size={18} />
            </GlassButton>
            <div className="glass grid place-items-center size-10 rounded-full shadow-[0_2px_10px_rgba(0,0,0,0.12)] text-[12px] font-bold text-[#ff3b30]">N</div>
          </div>

          <div className="absolute left-4 bottom-4 glass rounded-[16px] px-3.5 py-2.5 shadow-[0_2px_10px_rgba(0,0,0,0.12)] flex items-center gap-4">
            <ScaleBar mPerPx={mPerPx} />
            <div className="text-[12px] text-ink-2 tabular">
              <div className="flex items-center gap-1.5 font-medium text-ink">
                {playing && <span className="live-dot size-2 rounded-full bg-[#ff3b30]" />}
                {playing ? "Live" : live.status === "paused" ? "Paused" : live.status === "ended" ? "Finished" : "Press play"} · {live.elapsed.toFixed(1)} s
              </div>
              {live.actors.filter((a) => a.detected).length} of {live.actors.length} detected · halo ±{haloM.toFixed(1)} m
            </div>
          </div>

          <div className="absolute right-4 bottom-4 max-w-[270px] text-right text-[10.5px] text-ink-3 leading-snug">
            Illustrative border scene. Movement replays annotated field recordings; positions are distances along the fiber.
          </div>
        </div>

        <div className="flex flex-col gap-4">
          <div className="card p-5">
            <div className="text-[12px] font-medium text-ink-2">Near the fence now</div>
            <div className="mt-3">
              <ActorList actors={live.actors} />
            </div>
            <p className="text-[11.5px] text-ink-3 mt-3 leading-snug">
              Icons move along the annotated path. The coloured ring appears when the model raises an alert at that spot; the small halo on the cable is our ±{haloM.toFixed(1)} m location accuracy.
            </p>
          </div>
          <div className="card p-3 h-[300px] overflow-y-auto !rounded-[28px]">
            <AlertFeed alerts={live.alerts} />
          </div>
        </div>
      </div>
    </div>
  );
}
