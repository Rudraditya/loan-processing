export default function StatusPill({ verdict }) {
  const isHighRisk = verdict === 1;
  return (
    <span
      className={
        "inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-medium tabular-nums " +
        (isHighRisk
          ? "bg-red-500/10 text-red-400 ring-1 ring-inset ring-red-500/20"
          : "bg-emerald-500/10 text-emerald-400 ring-1 ring-inset ring-emerald-500/20")
      }
    >
      {isHighRisk ? "Flagged" : "Proceed"}
    </span>
  );
}
