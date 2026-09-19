import { useEffect, useRef } from "react";
import { animate } from "animejs";
import { useLoanStore } from "../store/useLoanStore";
import { prefersReducedMotion, useCountUp } from "../lib/anime";
import GlowCard from "./GlowCard";
import StatusPill from "./StatusPill";
import PageHeader from "./PageHeader";
import OrbitRing from "./OrbitRing";

const inr = new Intl.NumberFormat("en-IN", {
  style: "currency",
  currency: "INR",
  maximumFractionDigits: 0,
  notation: "compact",
});

function StatTile({ index, label, value, decimals = 0, prefix = "", suffix = "" }) {
  const countRef = useCountUp(value, { decimals, prefix, suffix });
  return (
    <GlowCard index={index} className="p-5">
      <div className="text-[10px] font-medium uppercase tracking-[0.14em] text-zinc-400">{label}</div>
      <div ref={countRef} className="mt-2 font-display text-3xl font-bold tabular-nums text-white">
        {prefix}0{suffix}
      </div>
    </GlowCard>
  );
}

function Meter({ label, pct, value, decimals = 1, suffix = "%" }) {
  const barRef = useRef(null);
  const clamped = Math.min(Math.max(pct, 0), 100);
  const countRef = useCountUp(value, { decimals, suffix });

  useEffect(() => {
    if (!barRef.current) return;
    if (prefersReducedMotion()) {
      barRef.current.style.width = `${clamped}%`;
      return;
    }
    const anim = animate(barRef.current, {
      width: `${clamped}%`,
      duration: 1000,
      ease: "outExpo",
    });
    return () => anim.pause();
  }, [clamped]);

  return (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between">
        <span className="text-xs text-zinc-200">{label}</span>
        <span ref={countRef} className="font-mono text-xs tabular-nums text-zinc-300">
          0{suffix}
        </span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-zinc-800">
        <div ref={barRef} className="h-full rounded-full bg-gradient-to-r from-blue-500 to-accent" style={{ width: 0 }} />
      </div>
    </div>
  );
}

export default function DashboardOverview() {
  const applicants = useLoanStore((state) => state.applicants);
  const selectApplicant = useLoanStore((state) => state.selectApplicant);

  const applicantCount = applicants.length;
  const pendingCount = applicants.filter((a) => a.Review_Status === "pending").length;
  const approvedCount = applicants.filter((a) => a.Review_Status === "approved").length;
  const flaggedCount = applicants.filter((a) => a.Classification_Verdict === 1).length;

  const avgExtractionConfidence =
    applicants.reduce((sum, a) => sum + a.Extraction_Confidence, 0) / (applicantCount || 1);
  const crossRefMatchRate =
    (applicants.filter((a) => a.Consistency_Flag).length / (applicantCount || 1)) * 100;
  const avgLatency = applicants.reduce((sum, a) => sum + a.Processing_Latency, 0) / (applicantCount || 1);

  const recent = [...applicants].slice(-6).reverse();

  return (
    <div className="relative overflow-hidden">
      <div className="pointer-events-none absolute right-0 -top-16 hidden translate-x-1/4 opacity-[0.35] xl:block">
        <OrbitRing size={260} />
      </div>

      <PageHeader
        eyebrow="Portfolio Overview"
        title="Dashboard"
        description="Live portfolio summary across all submitted applications this session."
      />

      <div className="mb-5 grid grid-cols-2 gap-4 lg:grid-cols-4">
        <StatTile index={0} label="Total Applicants" value={applicantCount} />
        <StatTile index={1} label="Pending Review" value={pendingCount} />
        <StatTile index={2} label="Approved" value={approvedCount} />
        <StatTile index={3} label="High Risk Flagged" value={flaggedCount} />
      </div>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-12">
        <GlowCard index={4} className="p-5 lg:col-span-7">
          <div className="mb-4 flex items-baseline justify-between">
            <span className="text-[11px] font-bold uppercase tracking-wide text-zinc-200">Recent Submissions</span>
            <span className="text-[10px] text-zinc-400">Latest {recent.length} records</span>
          </div>
          {recent.length === 0 ? (
            <p className="px-3 py-8 text-center text-[13px] text-zinc-500">
              No applications submitted yet this session.
            </p>
          ) : (
            <div className="space-y-1.5">
              {recent.map((a) => (
                <button
                  key={a.Applicant_ID}
                  onClick={() => selectApplicant(a.Applicant_ID)}
                  className="flex w-full items-center justify-between rounded-lg px-3 py-2.5 text-left transition-colors hover:bg-zinc-800/40"
                >
                  <div className="flex items-center gap-3">
                    <span className="font-mono text-xs tabular-nums text-zinc-300">{a.Applicant_ID}</span>
                    <span className="text-xs text-zinc-300">{a.Employment_Type}</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="font-mono text-xs tabular-nums text-zinc-200">
                      {inr.format(a.Requested_Loan_Amount)}
                    </span>
                    <StatusPill verdict={a.Classification_Verdict} />
                  </div>
                </button>
              ))}
            </div>
          )}
        </GlowCard>

        <GlowCard index={5} className="p-5 lg:col-span-5">
          <div className="mb-5 text-[11px] font-bold uppercase tracking-wide text-zinc-200">System Diagnostics</div>
          <div className="space-y-5">
            <Meter label="NLP Document Parsing Accuracy" pct={avgExtractionConfidence} value={avgExtractionConfidence} />
            <Meter label="Cross-Reference Matching Rate" pct={crossRefMatchRate} value={crossRefMatchRate} />
            <Meter
              label="Avg. Processing Latency"
              pct={(avgLatency / 5) * 100}
              value={avgLatency}
              decimals={2}
              suffix="s"
            />
          </div>
        </GlowCard>
      </div>
    </div>
  );
}
