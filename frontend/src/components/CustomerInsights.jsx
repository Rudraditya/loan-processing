import { useEffect, useMemo } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  MinusCircle,
  MousePointerClick,
  ShieldAlert,
} from "lucide-react";
import { motion, useMotionValue, useTransform, animate, useReducedMotion } from "motion/react";
import { useLoanStore } from "../store/useLoanStore";
import GlowCard from "./GlowCard";

const inr = new Intl.NumberFormat("en-IN", {
  style: "currency",
  currency: "INR",
  maximumFractionDigits: 0,
});

const RECOMMENDATION_TONES = {
  amber: "bg-amber-500/10 text-amber-400 ring-amber-500/25",
  red: "bg-red-500/10 text-red-400 ring-red-500/25",
  emerald: "bg-emerald-500/10 text-emerald-400 ring-emerald-500/25",
};

const VERIFICATION_STATUS = {
  matched: { icon: CheckCircle2, className: "text-emerald-400" },
  mismatch: { icon: AlertTriangle, className: "text-amber-400" },
  unavailable: { icon: MinusCircle, className: "text-zinc-600" },
};

function getRecommendation(applicant) {
  // Consistency_Flag: true = extraction cross-check verified, false = mismatch found.
  if (!applicant.Consistency_Flag) {
    return { label: "FLAG FOR MANUAL REVIEW", tone: "amber" };
  }
  if (applicant.Classification_Verdict === 1) {
    return { label: "DECLINE - HIGH RISK", tone: "red" };
  }
  return { label: "APPROVE - LOW RISK", tone: "emerald" };
}

function buildVerificationLog(applicant) {
  return [
    {
      field: "Monthly Net Income",
      source: "Salary Slip",
      value: inr.format(applicant.Monthly_Net_Income),
      status: "matched",
    },
    { field: "Employer Name", source: "Salary Slip", value: "Cross-referenced", status: "matched" },
    {
      field: "Avg. Monthly Balance",
      source: "Bank Statement",
      value: inr.format(applicant.Average_Monthly_Bank_Balance),
      status: applicant.Consistency_Flag ? "matched" : "mismatch",
    },
    {
      field: "Existing EMIs",
      source: "Bank Statement",
      value: inr.format(applicant.Total_Existing_EMIs),
      status: "matched",
    },
    {
      field: "Bounced Transactions (6M)",
      source: "Bank Statement",
      value: "Not extractable from documents",
      status: "unavailable",
    },
  ];
}

// Deterministic 6-point balance trend around the applicant's stored average, for illustration only.
function buildCashFlowSeries(applicant) {
  let seed = 0;
  for (const ch of applicant.Applicant_ID) seed = (seed * 31 + ch.charCodeAt(0)) >>> 0;
  const base = applicant.Average_Monthly_Bank_Balance;
  const points = [];
  for (let i = 0; i < 6; i++) {
    seed = (seed * 1103515245 + 12345) >>> 0;
    const wobble = ((seed % 2000) / 1000 - 1) * 0.18;
    points.push(Math.max(base * (1 + wobble), base * 0.4));
  }
  return points;
}

function Sparkline({ points }) {
  const width = 100;
  const height = 36;
  const max = Math.max(...points);
  const min = Math.min(...points);
  const range = max - min || 1;
  const coords = points.map((p, i) => {
    const x = (i / (points.length - 1)) * width;
    const y = height - ((p - min) / range) * height;
    return [x, y];
  });
  const path = coords.map(([x, y], i) => `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  const areaPath = `${path} L${width},${height} L0,${height} Z`;

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="h-16 w-full" preserveAspectRatio="none">
      <path d={areaPath} fill="rgba(59,130,246,0.12)" />
      <path d={path} fill="none" stroke="#3b82f6" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function Field({ label, value }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wide text-zinc-500">{label}</div>
      <div className="mt-1 font-mono text-sm tabular-nums text-zinc-200">{value}</div>
    </div>
  );
}

function RiskGauge({ probability }) {
  const reduce = useReducedMotion();
  const mv = useMotionValue(0);

  const displayPct = useTransform(mv, (v) => `${(v * 100).toFixed(1)}%`);
  const gaugeBackground = useTransform(mv, (v) => {
    const angle = Math.min(Math.max(v, 0), 1) * 360;
    const color = v >= 0.5 ? "#f87171" : v >= 0.25 ? "#fbbf24" : "#34d399";
    return `conic-gradient(${color} ${angle}deg, rgb(39 39 42) ${angle}deg)`;
  });

  useEffect(() => {
    if (reduce) {
      mv.set(probability);
      return;
    }
    const controls = animate(mv, probability, { duration: 1.2, ease: [0.16, 1, 0.3, 1] });
    return () => controls.stop();
  }, [probability, reduce, mv]);

  return (
    <motion.div
      className="relative flex h-16 w-16 shrink-0 items-center justify-center rounded-full"
      style={{ background: gaugeBackground }}
    >
      <div className="flex h-[46px] w-[46px] flex-col items-center justify-center rounded-full bg-zinc-950">
        <motion.span className="font-mono text-[13px] font-semibold tabular-nums text-zinc-100">
          {displayPct}
        </motion.span>
      </div>
    </motion.div>
  );
}

export default function CustomerInsights() {
  const activeApplicantId = useLoanStore((state) => state.activeApplicantId);
  const applicants = useLoanStore((state) => state.applicants);
  const setCurrentTab = useLoanStore((state) => state.setCurrentTab);

  const applicant = applicants.find((a) => a.Applicant_ID === activeApplicantId) ?? null;

  const verificationLog = useMemo(() => (applicant ? buildVerificationLog(applicant) : []), [applicant]);
  const cashFlow = useMemo(() => (applicant ? buildCashFlowSeries(applicant) : []), [applicant]);

  if (!applicant) {
    return (
      <div className="flex flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-zinc-800 py-32 text-center">
        <MousePointerClick size={22} strokeWidth={1.5} className="text-zinc-600" />
        <p className="text-sm text-zinc-500">Select an applicant from the Ledger to view their risk assessment.</p>
        <button
          onClick={() => setCurrentTab("ledger")}
          className="mt-1 rounded-md border border-zinc-800 px-3.5 py-1.5 text-[13px] text-zinc-300 transition-colors hover:border-accent hover:text-accent"
        >
          Go to Ledger
        </button>
      </div>
    );
  }

  const recommendation = getRecommendation(applicant);
  const dtiPct = Math.round((applicant.Total_Existing_EMIs / applicant.Monthly_Net_Income) * 1000) / 10;

  return (
    <div>
      <div className="mb-4 flex items-baseline justify-between px-1">
        <h1 className="text-base font-semibold text-zinc-100">Customer Insights &amp; Risk Assessment</h1>
        <span className="font-mono text-xs tabular-nums text-zinc-500">{applicant.Applicant_ID}</span>
      </div>

      <div key={applicant.Applicant_ID} className="grid grid-cols-1 gap-4 lg:grid-cols-12">
        {/* Applicant Profile */}
        <GlowCard index={0} className="p-5 lg:col-span-4">
          <div className="mb-4 text-[11px] font-medium uppercase tracking-wide text-zinc-500">
            Applicant Profile
          </div>
          <div className="grid grid-cols-2 gap-x-5 gap-y-4">
            <Field label="Age" value={applicant.Age} />
            <Field label="Employment" value={applicant.Employment_Type} />
            <Field label="Monthly Income" value={inr.format(applicant.Monthly_Net_Income)} />
            <Field label="Existing EMIs" value={inr.format(applicant.Total_Existing_EMIs)} />
            <Field label="EMI / Income" value={`${dtiPct}%`} />
            <Field label="Requested Loan" value={inr.format(applicant.Requested_Loan_Amount)} />
            <Field label="Tenure" value={`${applicant.Requested_Tenure_Months} mo`} />
          </div>
        </GlowCard>

        {/* ML Risk Meter — compact, gauge beside text */}
        <GlowCard index={1} className="flex flex-col justify-between p-5 lg:col-span-4">
          <div className="mb-4 text-[11px] font-medium uppercase tracking-wide text-zinc-500">
            ML Risk Meter
          </div>
          <div className="flex flex-1 items-center gap-4">
            <RiskGauge probability={applicant.Default_Probability} />
            <div className="flex flex-col gap-2">
              <span className="text-[10px] uppercase tracking-wide text-zinc-500">Default Probability</span>
              <span
                className={
                  "w-fit rounded-full px-2.5 py-1 text-center text-[11px] font-semibold tracking-wide ring-1 ring-inset " +
                  RECOMMENDATION_TONES[recommendation.tone]
                }
              >
                {recommendation.label}
              </span>
            </div>
          </div>
        </GlowCard>

        {/* Explainable AI & Flags */}
        <GlowCard index={2} className="p-5 lg:col-span-4">
          <div className="mb-4 text-[11px] font-medium uppercase tracking-wide text-zinc-500">
            Explainable AI &amp; Flags
          </div>

          {applicant.Consistency_Flag ? (
            <div className="mb-4 flex items-start gap-2 rounded-lg bg-emerald-500/10 px-3 py-2 text-[12px] text-emerald-400 ring-1 ring-inset ring-emerald-500/20">
              <CheckCircle2 size={14} strokeWidth={2} className="mt-0.5 shrink-0" />
              <span>Extraction cross-check verified.</span>
            </div>
          ) : (
            <div className="mb-4 flex items-start gap-2 rounded-lg bg-amber-500/10 px-3 py-2 text-[12px] text-amber-400 ring-1 ring-inset ring-amber-500/20">
              <AlertTriangle size={14} strokeWidth={2} className="mt-0.5 shrink-0" />
              <span>Consistency check failed for a source parameter.</span>
            </div>
          )}

          <div className="text-[10px] uppercase tracking-wide text-zinc-500">Top Risk Drivers</div>
          <div className="mt-2.5 flex flex-wrap gap-1.5">
            {applicant.Top_Risk_Drivers.length === 0 ? (
              <span className="text-[13px] text-zinc-500">No material risk drivers identified.</span>
            ) : (
              applicant.Top_Risk_Drivers.map((driver) => (
                <span
                  key={driver}
                  className="inline-flex items-center gap-1.5 rounded-full border border-zinc-800 px-2.5 py-1 text-[11px] text-zinc-300"
                >
                  <ShieldAlert size={11} strokeWidth={2} className="text-zinc-500" />
                  {driver}
                </span>
              ))
            )}
          </div>
        </GlowCard>

        {/* Cash Flow Trends */}
        <GlowCard index={3} className="p-5 lg:col-span-6">
          <div className="mb-4 flex items-baseline justify-between">
            <span className="text-[11px] font-medium uppercase tracking-wide text-zinc-500">
              Cash Flow Trends
            </span>
            <span className="text-[10px] text-zinc-600">Illustrative, 6-point trend</span>
          </div>
          <Sparkline points={cashFlow} />
          <div className="mt-3 flex justify-between text-[10px] tabular-nums text-zinc-600">
            <span>-5 mo</span>
            <span>Current</span>
          </div>
        </GlowCard>

        {/* Cross-Verification Log */}
        <GlowCard index={4} className="p-5 lg:col-span-6">
          <div className="mb-4 text-[11px] font-medium uppercase tracking-wide text-zinc-500">
            Cross-Verification Log
          </div>
          <div className="space-y-2.5">
            {verificationLog.map((row) => {
              const { icon: Icon, className } = VERIFICATION_STATUS[row.status];
              return (
                <div key={row.field} className="flex items-center gap-2.5 text-[12px]">
                  <Icon size={13} strokeWidth={2} className={"shrink-0 " + className} />
                  <span className="w-36 shrink-0 text-zinc-400">{row.field}</span>
                  <span className="w-28 shrink-0 text-zinc-600">{row.source}</span>
                  <span className="truncate font-mono tabular-nums text-zinc-300">{row.value}</span>
                </div>
              );
            })}
          </div>
        </GlowCard>
      </div>
    </div>
  );
}
