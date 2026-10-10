"use client";

import { useEffect, useState } from "react";
import { ClassInfo, getJson, Results, ScenarioList } from "@/lib/api";
import { StatTile } from "./StatTile";

type Data = { results: Results; classes: ClassInfo[] };

const pct = (v: number) => `${(100 * v).toFixed(1)}%`;

function Bar({ label, color, value, baseline, baselineLabel }: { label: string; color: string; value: number; baseline?: number; baselineLabel: string }) {
  return (
    <div className="grid grid-cols-[130px_1fr_48px] items-center gap-3">
      <span className="text-[13px] text-ink-2 truncate">{label}</span>
      <div className="relative h-3 rounded-full bg-black/[0.05]">
        <div className="absolute inset-y-0 left-0 rounded-full transition-[width] duration-700" style={{ width: `${value * 100}%`, background: color }} />
        {baseline !== undefined && (
          <div className="absolute -top-1 -bottom-1 w-[3px] rounded-full bg-ink/70" style={{ left: `calc(${baseline * 100}% - 1.5px)` }} title={`${baselineLabel} ${baseline.toFixed(2)}`} />
        )}
      </div>
      <span className="text-[13px] font-semibold tabular text-right">{value.toFixed(2)}</span>
    </div>
  );
}

function loroMeans(rows: Results["loro"]["cnn"]) {
  const by = new Map<string, number[]>();
  rows.forEach((r) => by.set(r.class, [...(by.get(r.class) ?? []), r.recall]));
  return [...by.entries()].map(([cls, v]) => ({ cls, recall: v.reduce((a, b) => a + b, 0) / v.length, n: v.length }));
}

function ClassF1({ results, classes }: Data) {
  const cnn = results.test.cnn.cells.f1;
  const lgbm = results.test.lgbm.cells.f1;
  return (
    <section className="card p-6 sm:p-8">
      <h2 className="text-[22px] font-semibold tracking-tight">Per-class accuracy</h2>
      <p className="text-[14px] text-ink-2 mt-1 mb-6">F1 score on the held-out test period. The dark tick is the LightGBM baseline.</p>
      <div className="flex flex-col gap-3.5">
        {classes.slice(0, results.classes.length).map((c, i) => (
          <Bar key={c.id} label={c.label} color={c.color === "#8E8E93" ? "#34c759" : c.color} value={cnn[i]} baseline={lgbm[i]} baselineLabel="LightGBM" />
        ))}
      </div>
    </section>
  );
}

function Generalisation({ results, classes }: Data) {
  const ref = results.reference.loro_recall;
  const label = (name: string) => classes.find((c) => c.name === name)?.label ?? name;
  const color = (name: string) => classes.find((c) => c.name === name)?.color ?? "#0071e3";
  return (
    <section className="card p-6 sm:p-8">
      <h2 className="text-[22px] font-semibold tracking-tight">The honest test: a recording it has never heard</h2>
      <p className="text-[14px] text-ink-2 mt-1 mb-6 max-w-2xl">
        We retrain the model without one recording, then ask it to classify that recording. This is closer to a new site than the test split. The dark tick is the reference baseline.
      </p>
      <div className="flex flex-col gap-3.5">
        {loroMeans(results.loro.cnn).map((r) => (
          <Bar key={r.cls} label={`${label(r.cls)} (${r.n})`} color={color(r.cls)} value={r.recall} baseline={ref[r.cls]} baselineLabel="Reference" />
        ))}
      </div>
      <p className="text-[13px] text-ink-3 mt-5">
        Walking transfers well to new recordings. Vehicles, fences and longboards still confuse the model on some unseen sites, so more varied recordings are the next step.
      </p>
    </section>
  );
}

function Limits() {
  const items = [
    "Test periods come from the same sessions as training; the never-heard test above is the stricter check.",
    "Quiet background comes from a single recording, so false-alarm rates are indicative only.",
    "Unlabelled stretches of cable cannot be scored, so live alerts there are shown as unverified.",
    "Inputs are six frequency bands every 0.1 s; no phase or speed information yet.",
  ];
  return (
    <section className="card p-6 sm:p-8">
      <h2 className="text-[22px] font-semibold tracking-tight">What we are careful about</h2>
      <ul className="mt-4 flex flex-col gap-2.5">
        {items.map((t) => (
          <li key={t} className="flex gap-3 text-[14px] text-ink-2">
            <span className="mt-2 size-1.5 rounded-full bg-ink-3 shrink-0" />
            {t}
          </li>
        ))}
      </ul>
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
  const ref = data.results.reference.test;
  return (
    <div className="flex flex-col gap-6 pt-10">
      <section className="text-center max-w-3xl mx-auto mb-2">
        <h1 className="text-[40px] sm:text-[56px] font-semibold tracking-[-0.03em] leading-[1.05]">Proven on real field recordings.</h1>
        <p className="mt-4 text-[17px] sm:text-[19px] text-ink-2">16 recordings, 486 annotated scenes, 8 kinds of activity. Scored on a period the model never trained on.</p>
      </section>
      <section className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <StatTile big accent label="Activity vs quiet" value={pct(cnn.tvb.accuracy)} hint={`baseline ${pct(ref.bg_acc)}`} />
        <StatTile big label="Overall accuracy" value={pct(cnn.cells.accuracy)} hint={`baseline ${pct(ref.accuracy)}`} />
        <StatTile big label="Events detected" value={pct(cnn.events.pod)} hint={`${cnn.events.n_gt} annotated events`} />
        <StatTile big label="Location error" value={`${cnn.events.loc_err_m.toFixed(1)} m`} hint="median, along the fiber" />
      </section>
      <section className="card p-6 sm:p-8 flex flex-wrap items-center justify-between gap-6">
        <div>
          <div className="text-[13px] text-ink-2 font-medium">Macro-F1 across all 9 classes</div>
          <div className="text-[52px] font-semibold tracking-tight tabular leading-none mt-2">{cnn.cells.macro_f1.toFixed(2)}</div>
        </div>
        <div className="flex gap-8 text-[14px]">
          <div>
            <div className="text-ink-3">LightGBM baseline</div>
            <div className="text-[24px] font-semibold tabular">{data.results.test.lgbm.cells.macro_f1.toFixed(2)}</div>
          </div>
          <div>
            <div className="text-ink-3">Reference</div>
            <div className="text-[24px] font-semibold tabular">{ref.macro_f1.toFixed(2)}</div>
          </div>
        </div>
      </section>
      <ClassF1 {...data} />
      <Generalisation {...data} />
      <Limits />
    </div>
  );
}
