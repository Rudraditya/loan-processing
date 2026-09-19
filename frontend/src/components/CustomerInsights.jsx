import { useEffect, useMemo, useRef } from "react";
import { AlertTriangle, CheckCircle2, MousePointerClick, ShieldAlert } from "lucide-react";
import { animate } from "animejs";
import { prefersReducedMotion } from "../lib/anime";
import { useLoanStore } from "../store/useLoanStore";
import GlowCard from "./GlowCard";
import PageHeader from "./PageHeader";

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
  ];
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
      <path
        d={path}
        fill="none"
        stroke="#3b82f6"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeDasharray="6,3"
      />
    </svg>
  );
}

function Field({ label, value }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wide text-zinc-300">{label}</div>
      <div className="mt-1 font-mono text-sm tabular-nums text-zinc-200">{value}</div>
    </div>
  );
}

const GAUGE_RADIUS = 26;
const GAUGE_CIRCUMFERENCE = 2 * Math.PI * GAUGE_RADIUS;

function riskColor(v) {
  return v >= 0.5 ? "#f87171" : v >= 0.25 ? "#fbbf24" : "#34d399";
}

function RiskGauge({ probability }) {
  const circleRef = useRef(null);
  const textRef = useRef(null);

  useEffect(() => {
    const circle = circleRef.current;
    const text = textRef.current;
    if (!circle || !text) return;

    const clamped = Math.min(Math.max(probability, 0), 1);

    function render(v) {
      circle.style.strokeDashoffset = GAUGE_CIRCUMFERENCE - v * GAUGE_CIRCUMFERENCE;
      circle.style.stroke = riskColor(v);
      text.textContent = `${(v * 100).toFixed(1)}%`;
    }

    if (prefersReducedMotion()) {
      render(clamped);
      return;
    }

    const counter = { v: 0 };
    const anim = animate(counter, {
      v: clamped,
      duration: 1200,
      ease: "outExpo",
      onUpdate: () => render(counter.v),
    });
    return () => anim.pause();
  }, [probability]);

  return (
    <div className="relative flex h-16 w-16 shrink-0 items-center justify-center">
      <svg viewBox="0 0 64 64" className="absolute inset-0 -rotate-90">
        <circle cx="32" cy="32" r={GAUGE_RADIUS} fill="none" stroke="#27272a" strokeWidth="6" />
        <circle
          ref={circleRef}
          cx="32"
          cy="32"
          r={GAUGE_RADIUS}
          fill="none"
          strokeWidth="6"
          strokeLinecap="round"
          strokeDasharray={GAUGE_CIRCUMFERENCE}
        />
      </svg>
      <span ref={textRef} className="relative font-mono text-[13px] font-semibold tabular-nums text-zinc-100">
        0.0%
      </span>
    </div>
  );
}

export default function CustomerInsights() {
  const activeApplicantId = useLoanStore((state) => state.activeApplicantId);
  const applicants = useLoanStore((state) => state.applicants);
  const setCurrentTab = useLoanStore((state) => state.setCurrentTab);

  const applicant = applicants.find((a) => a.Applicant_ID === activeApplicantId) ?? null;

  const verificationLog = useMemo(() => (applicant ? buildVerificationLog(applicant) : []), [applicant]);
  const cashFlow = useMemo(
    () => (applicant?.Cash_Flow_Trend?.length === 6 ? applicant.Cash_Flow_Trend : null),
    [applicant]
  );

  if (!applicant) {
    return (
      <div className="flex flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-zinc-800 py-32 text-center">
        <MousePointerClick size={22} strokeWidth={1.5} className="text-zinc-400" />
        <p className="text-sm text-zinc-300">Select an applicant from the Ledger to view their risk assessment.</p>
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
  // The requested loan's own estimated installment against income - distinct
  // from dtiPct above (existing debt burden). Mirrors Requested_EMI_to_Income_Ratio,
  // the single strongest feature added to the model (see CLAUDE.md), which
  // was otherwise never surfaced to a human reviewer.
  const newEmiToIncomePct =
    Math.round((applicant.Estimated_Monthly_EMI / applicant.Monthly_Net_Income) * 1000) / 10;

  return (
    <div>
      <PageHeader
        eyebrow="Risk Assessment"
        title="Customer Insights"
        description="Extraction verification, risk meter, and cash-flow trend for the selected applicant."
        right={
          <span className="rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-1.5 font-mono text-xs tabular-nums text-zinc-200">
            {applicant.Applicant_ID}
          </span>
        }
      />

      <div key={applicant.Applicant_ID} className="grid grid-cols-1 gap-4 lg:grid-cols-12">
        {/* Applicant Profile */}
        <GlowCard index={0} className="p-5 lg:col-span-4">
          <div className="mb-4 text-[11px] font-bold uppercase tracking-wide text-zinc-200">
            Applicant Profile
          </div>
          <div className="grid grid-cols-2 gap-x-5 gap-y-4">
            <Field label="Age" value={applicant.Age} />
            <Field label="Employment" value={applicant.Employment_Type} />
            <Field label="Monthly Income" value={inr.format(applicant.Monthly_Net_Income)} />
            <Field label="Existing EMIs" value={inr.format(applicant.Total_Existing_EMIs)} />
            <Field label="EMI as % of Income" value={`${dtiPct}%`} />
            <Field label="Requested Loan" value={inr.format(applicant.Requested_Loan_Amount)} />
            <Field label="Tenure" value={`${applicant.Requested_Tenure_Months} mo`} />
            <Field label="Est. Monthly EMI (New)" value={inr.format(applicant.Estimated_Monthly_EMI)} />
            <Field label="New EMI as % of Income" value={`${newEmiToIncomePct}%`} />
          </div>
        </GlowCard>

        {/* Risk Meter — compact, gauge beside text */}
        <GlowCard index={1} className="flex flex-col justify-between p-5 lg:col-span-4">
          <div className="mb-4 text-[11px] font-bold uppercase tracking-wide text-zinc-200">
            Risk Meter
          </div>
          <div className="flex flex-1 items-center gap-4">
            <RiskGauge probability={applicant.Default_Probability} />
            <div className="flex flex-col gap-2">
              <span className="text-[10px] uppercase tracking-wide text-zinc-300">Chance of Default</span>
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

        {/* Key Indicators */}
        <GlowCard index={2} className="p-5 lg:col-span-4">
          <div className="mb-4 text-[11px] font-bold uppercase tracking-wide text-zinc-200">
            Key Indicators
          </div>

          {applicant.Consistency_Flag ? (
            <div className="mb-4 flex items-start gap-2 rounded-lg bg-emerald-500/10 px-3 py-2 text-[12px] text-emerald-400 ring-1 ring-inset ring-emerald-500/20">
              <CheckCircle2 size={14} strokeWidth={2} className="mt-0.5 shrink-0" />
              <span>All details matched successfully.</span>
            </div>
          ) : (
            <div className="mb-4 flex items-start gap-2 rounded-lg bg-amber-500/10 px-3 py-2 text-[12px] text-amber-400 ring-1 ring-inset ring-amber-500/20">
              <AlertTriangle size={14} strokeWidth={2} className="mt-0.5 shrink-0" />
              <span>Some details couldn't be automatically verified — review before deciding.</span>
            </div>
          )}

          <div className="text-[10px] uppercase tracking-wide text-zinc-300">Key Factors</div>
          <div className="mt-2.5 flex flex-wrap gap-1.5">
            {applicant.Top_Risk_Drivers.length === 0 ? (
              <span className="text-[13px] text-zinc-300">No notable risk factors found.</span>
            ) : (
              applicant.Top_Risk_Drivers.map((driver) => (
                <span
                  key={driver}
                  className="inline-flex items-center gap-1.5 rounded-full border border-zinc-800 px-2.5 py-1 text-[11px] text-zinc-300"
                >
                  <ShieldAlert size={11} strokeWidth={2} className="text-zinc-400" />
                  {driver}
                </span>
              ))
            )}
          </div>
        </GlowCard>

        {/* Cash Flow Trends */}
        <GlowCard index={3} className="p-5 lg:col-span-4">
          <div className="mb-4 flex items-baseline justify-between">
            <span className="text-[11px] font-bold uppercase tracking-wide text-zinc-200">
              Cash Flow Trends
            </span>
            <span className="text-[10px] text-zinc-400">6-month trend</span>
          </div>
          {cashFlow ? (
            <>
              <Sparkline points={cashFlow} />
              <div className="mt-3 flex justify-between text-[10px] tabular-nums text-zinc-400">
                <span>-5 mo</span>
                <span>Current</span>
              </div>
            </>
          ) : (
            <p className="text-[12px] text-zinc-400">Cash flow trend not available for this applicant.</p>
          )}
        </GlowCard>

        {/* Verification Details */}
        <GlowCard index={4} className="p-5 lg:col-span-8">
          <div className="mb-4 text-[11px] font-bold uppercase tracking-wide text-zinc-200">
            Verification Details
          </div>
          <div className="space-y-2.5">
            {verificationLog.map((row) => {
              const { icon: Icon, className } = VERIFICATION_STATUS[row.status];
              return (
                <div key={row.field} className="flex items-center gap-2.5 text-[12px]">
                  <Icon size={13} strokeWidth={2} className={"shrink-0 " + className} />
                  <span className="w-36 shrink-0 text-zinc-200">{row.field}</span>
                  <span className="w-28 shrink-0 text-zinc-400">{row.source}</span>
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
