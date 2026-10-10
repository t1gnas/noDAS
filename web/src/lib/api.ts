export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export const wsUrl = (path: string) => API_URL.replace(/^http/, "ws") + path;

export type ClassInfo = { id: number; name: string; label: string; color: string; hidden?: boolean };

export type Scenario = {
  id: string;
  title: string;
  subtitle: string;
  icon: string;
  duration_s: number;
  view_start_m: number;
  view_end_m: number;
  fiber_start_m: number;
  fiber_end_m: number;
};

export type Meta = {
  type: "meta";
  scenario: string;
  title: string;
  width: number;
  hop_rows: number;
  row_seconds: number;
  duration_s: number;
  speed: number;
  classes: ClassInfo[];
  view_start_m: number;
  view_end_m: number;
  fiber_start_m: number;
  fiber_end_m: number;
};

export type AlertEvent = {
  kind: "open" | "update" | "clear";
  id: number;
  t: number;
  pos_m: number;
  c0: number;
  c1: number;
  confidence: number;
  duration_s: number;
  verified: boolean;
  label: string;
  color: string;
  model_label: string;
  truth_label: string | null;
};

export type Actor = {
  id: number;
  pos_m: number;
  label: string;
  color: string;
  detected: boolean;
  alert_id?: number;
  alert_pos_m?: number;
  confidence?: number;
};

export type Frame = {
  type: "frame";
  t: number;
  end: number;
  rows: number;
  intensity?: string;
  overlay?: string;
  truth?: string;
  actors?: Actor[];
  events: AlertEvent[];
};

export type Summary = {
  type: "end";
  alerts: number;
  verified: number;
  unverified: number;
  correct: number;
  by_class: Record<string, number>;
};

export type ErrorMessage = { type: "error"; message: string };

export type LiveMessage = Meta | Frame | Summary | ErrorMessage;

export type ScenarioList = { scenarios: Scenario[]; classes: ClassInfo[]; speeds: number[] };

export type CellReport = {
  precision: number[];
  recall: number[];
  f1: number[];
  support: number[];
  macro_f1: number;
  accuracy: number;
  confusion: number[][];
};

export type ModelResults = {
  cells: CellReport;
  coarse: CellReport;
  tvb: { pod: number; nar: number; fnr: number; accuracy: number };
  events: { n_gt: number; pod: number; class_acc: number; loc_err_m: number; loc_err_mean_m: number };
  false_alarms: number;
};

export type LoroRow = { id: string; class: string; recall: number; event_pod: number | null; dist: number[] };

export type Results = {
  classes: string[];
  test: Record<"cnn" | "lgbm", ModelResults>;
  loro: { cnn: LoroRow[] };
  reference: {
    test: { macro_f1: number; accuracy: number; bg_acc: number };
    f1: Record<string, number>;
    loro_recall: Record<string, number>;
  };
};

export async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(API_URL + path, { cache: "no-store" });
  if (!res.ok) throw new Error(`${path}: ${res.status} ${await res.text()}`);
  return res.json() as Promise<T>;
}
