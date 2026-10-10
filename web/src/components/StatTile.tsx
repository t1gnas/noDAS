type Props = { label: string; value: string | number; hint?: string; accent?: boolean; big?: boolean };

export function StatTile({ label, value, hint, accent, big }: Props) {
  return (
    <div className={`rounded-[22px] p-4 sm:p-5 ${accent ? "bg-gradient-to-br from-[#0a84ff] to-[#5e5ce6] text-white" : "bg-bg"}`}>
      <div className={`text-[12px] font-medium ${accent ? "text-white/80" : "text-ink-2"}`}>{label}</div>
      <div className={`${big ? "text-[44px]" : "text-[30px]"} font-semibold tracking-tight tabular leading-tight mt-1`}>{value}</div>
      {hint && <div className={`text-[11.5px] mt-0.5 ${accent ? "text-white/75" : "text-ink-3"}`}>{hint}</div>}
    </div>
  );
}
