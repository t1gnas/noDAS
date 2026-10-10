"use client";

import { useEffect, useState } from "react";
import { ClassInfo, getJson, Results, ScenarioList } from "@/lib/api";
import { StatTile } from "./StatTile";

type Data = { results: Results; classes: ClassInfo[] };

const CHART_HIDDEN = ["openclose"];

const pct = (v: number) => `${(100 * v).toFixed(1)}%`;

function Bar({ label, color, value }: { label: string; color: string; value: number }) {
  return (
    <div className="grid grid-cols-[130px_1fr_48px] items-center gap-3">
      <span className="text-[13px] text-ink-2 truncate">{label}</span>
      <div className="relative h-3 rounded-full bg-black/[0.05]">
        <div className="absolute inset-y-0 left-0 rounded-full transition-[width] duration-700" style={{ width: `${value * 100}%`, background: color }} />
      </div>
      <span className="text-[13px] font-semibold tabular text-right">{value.toFixed(2)}</span>
    </div>
  );
}

function loroMeans(rows: Results["loro"]["cnn"]) {
  const by = new Map<string, number[]>();
  rows.forEach((r) => by.set(r.class, [...(by.get(r.class) ?? []), r.recall]));
  return [...by.entries()].map(([cls, v]) => ({ cls, recall: v.reduce((a, b) => a + b, 0) / v.length }));
}

function scoredClasses({ results, classes }: Data) {
  const cells = results.test.cnn.cells;
  return classes.slice(0, results.classes.length).filter((c, i) => !c.hidden && cells.support[i] > 0);
}

function ClassF1(data: Data) {
  const f1 = data.results.test.cnn.cells.f1;
  return (
    <section className="card p-6 sm:p-8">
      <h2 className="text-[22px] font-semibold tracking-tight">Per-class accuracy</h2>
      <p className="text-[14px] text-ink-2 mt-1 mb-6">F1 score on the held-out test period of each recording.</p>
      <div className="flex flex-col gap-3.5">
        {scoredClasses(data)
          .filter((c) => !CHART_HIDDEN.includes(c.name))
          .map((c) => (
            <Bar key={c.id} label={c.label} color={c.id === 0 ? "#34c759" : c.color} value={f1[c.id]} />
          ))}
      </div>
    </section>
  );
}

function Generalisation({ results, classes }: Data) {
  const label = (name: string) => classes.find((c) => c.name === name)?.label ?? name;
  const color = (name: string) => classes.find((c) => c.name === name)?.color ?? "#0071e3";
  return (
    <section className="card p-6 sm:p-8">
      <h2 className="text-[22px] font-semibold tracking-tight">The honest test: a recording it has never heard</h2>
      <p className="text-[14px] text-ink-2 mt-1 mb-6 max-w-2xl">
        We retrain the model without one recording, then ask it to classify that recording. This is closer to a brand-new site than the test split. Bars show recall averaged over the held-out recordings of each kind.
      </p>
      <div className="flex flex-col gap-3.5">
        {loroMeans(results.loro.cnn).map((r) => (
          <Bar key={r.cls} label={label(r.cls)} color={color(r.cls)} value={r.recall} />
        ))}
      </div>
    </section>
  );
}

export function ResultsView() {
  const [data, setData] = useState<Data | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([getJson<Results>("/api/results"), getJson<ScenarioList>("/api/scenarios")]).then(
      ([results, list]) => setData({ results, classes: list.classes }),
      (e: Error) => setError(e.message),
    );
  }, []);

  if (error) return <div className="card p-6 mt-10 text-[14px] text-[#d70015]">Could not load results: {error}</div>;
  if (!data) return <div className="mt-20 text-center text-ink-3">Loading results…</div>;
  const cnn = data.results.test.cnn;
  const n = scoredClasses(data).length;
  return (
    <div className="flex flex-col gap-6 pt-10">
      <section className="text-center max-w-3xl mx-auto mb-2">
        <h1 className="text-[40px] sm:text-[56px] font-semibold tracking-[-0.03em] leading-[1.05]">Proven on real field recordings.</h1>
        <p className="mt-4 text-[17px] sm:text-[19px] text-ink-2">
          15 recordings, {n - 1} kinds of activity plus quiet background. Scored on a period of each recording the model never trained on.
        </p>
      </section>
      <section className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <StatTile big accent label="Activity vs quiet" value={pct(cnn.tvb.accuracy)} hint="tells real activity from background" />
        <StatTile big label="Overall accuracy" value={pct(cnn.cells.accuracy)} hint={`across ${n} classes`} />
        <StatTile big label="Events detected" value={pct(cnn.events.pod)} hint={`${cnn.events.n_gt} annotated events`} />
        <StatTile big label="Location error" value={`±${cnn.events.loc_err_mean_m.toFixed(1)} m`} hint={`mean along the fiber · median ${cnn.events.loc_err_m.toFixed(1)} m`} />
      </section>
      <section className="card p-6 sm:p-8 flex flex-wrap items-center justify-between gap-6">
        <div>
          <div className="text-[13px] text-ink-2 font-medium">Macro-F1 across {n} classes</div>
          <div className="text-[52px] font-semibold tracking-tight tabular leading-none mt-2">{cnn.cells.macro_f1.toFixed(2)}</div>
        </div>
        <div className="flex gap-8 text-[14px]">
          <div>
            <div className="text-ink-3">Threat detection rate</div>
            <div className="text-[24px] font-semibold tabular">{pct(cnn.tvb.pod)}</div>
          </div>
          <div>
            <div className="text-ink-3">Nuisance alarm rate</div>
            <div className="text-[24px] font-semibold tabular">{pct(cnn.tvb.nar)}</div>
          </div>
        </div>
      </section>
      <ClassF1 {...data} />
      <Generalisation {...data} />
    </div>
  );
}
