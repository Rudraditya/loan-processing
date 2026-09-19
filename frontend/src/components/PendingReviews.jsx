import { useState } from "react";
import { AlertTriangle, CheckCircle2, ClipboardCheck, FileSearch, ShieldAlert } from "lucide-react";
import { useLoanStore } from "../store/useLoanStore";
import { saveApplicantToExcel } from "../lib/saveApplicant";
import GlowCard from "./GlowCard";
import ManualReviewModal from "./ManualReviewModal";
import PageHeader from "./PageHeader";

const inr = new Intl.NumberFormat("en-IN", {
  style: "currency",
  currency: "INR",
  maximumFractionDigits: 0,
});

// Literal class names throughout (not runtime string-splicing) so Tailwind's
// JIT scanner - which only picks up class names it can see verbatim in the
// source - actually generates these.
function riskTone(probability) {
  if (probability >= 0.5) {
    return { badgeClassName: "bg-red-500/10 text-red-400 ring-red-500/20", barClassName: "bg-red-500", label: "High Risk" };
  }
  if (probability >= 0.25) {
    return { badgeClassName: "bg-amber-500/10 text-amber-400 ring-amber-500/20", barClassName: "bg-amber-500", label: "Medium Risk" };
  }
  return { badgeClassName: "bg-emerald-500/10 text-emerald-400 ring-emerald-500/20", barClassName: "bg-emerald-500", label: "Low Risk" };
}

function RiskMeter({ probability }) {
  const tone = riskTone(probability);
  return (
    <div className="flex items-center gap-2.5">
      <div className="h-1.5 w-24 overflow-hidden rounded-full bg-zinc-800">
        <div
          className={"h-full rounded-full " + tone.barClassName}
          style={{ width: `${Math.round(probability * 100)}%` }}
        />
      </div>
      <span className={"rounded-full px-2 py-0.5 text-[11px] font-medium ring-1 ring-inset " + tone.badgeClassName}>
        {(probability * 100).toFixed(1)}% &middot; {tone.label}
      </span>
    </div>
  );
}

function PendingCard({ applicant, index, onQuickApprove, onManualReview, saving }) {
  return (
    <GlowCard index={index} className="p-5">
      <div className="mb-4 flex items-center justify-between">
        <span className="font-mono text-sm tabular-nums text-zinc-100">{applicant.Applicant_ID}</span>
        <span className="text-xs text-zinc-500">{applicant.Employment_Type}</span>
      </div>

      <div className="mb-4">
        <div className="mb-1.5 text-[10px] uppercase tracking-wide text-zinc-500">Risk Meter</div>
        <RiskMeter probability={applicant.Default_Probability} />
      </div>

      {!applicant.Consistency_Flag && (
        <div className="mb-3 flex items-start gap-2 rounded-lg bg-amber-500/10 px-3 py-2 text-[12px] text-amber-400 ring-1 ring-inset ring-amber-500/20">
          <AlertTriangle size={13} strokeWidth={2} className="mt-0.5 shrink-0" />
          <span>Some details couldn't be automatically verified — review before deciding.</span>
        </div>
      )}

      <div className="mb-5">
        <div className="mb-1.5 text-[10px] uppercase tracking-wide text-zinc-500">Key Indicators</div>
        <div className="flex flex-wrap gap-1.5">
          {applicant.Top_Risk_Drivers.length === 0 ? (
            <span className="text-[12px] text-zinc-500">No notable risk factors found.</span>
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
      </div>

      <div className="flex items-center gap-2.5">
        <button
          onClick={() => onQuickApprove(applicant.Applicant_ID)}
          disabled={saving}
          className="flex flex-1 items-center justify-center gap-1.5 rounded-lg bg-emerald-500/10 px-3 py-2 text-[13px] font-medium text-emerald-400 ring-1 ring-inset ring-emerald-500/20 transition-colors hover:bg-emerald-500/20 disabled:cursor-not-allowed disabled:opacity-60"
        >
          <CheckCircle2 size={14} strokeWidth={2} />
          {saving ? "Approving..." : "Quick Approve"}
        </button>
        <button
          onClick={() => onManualReview(applicant.Applicant_ID)}
          className="flex flex-1 items-center justify-center gap-1.5 rounded-lg border border-zinc-800 px-3 py-2 text-[13px] font-medium text-zinc-300 transition-colors hover:border-accent hover:text-accent"
        >
          <FileSearch size={14} strokeWidth={2} />
          Manual Review
        </button>
      </div>
    </GlowCard>
  );
}

export default function PendingReviews() {
  const applicants = useLoanStore((state) => state.applicants);
  const approveApplicant = useLoanStore((state) => state.approveApplicant);
  const rejectApplicant = useLoanStore((state) => state.rejectApplicant);

  const [reviewingId, setReviewingId] = useState(null);
  const [approvingId, setApprovingId] = useState(null);

  const pending = applicants.filter((a) => a.Review_Status === "pending");
  const reviewingApplicant = applicants.find((a) => a.Applicant_ID === reviewingId) ?? null;

  // Store update happens immediately so the UI never waits on the network;
  // the Excel export runs after and its result is ignored by the caller -
  // fire-and-forget, matching saveApplicantToExcel's own contract.
  async function handleQuickApprove(applicantId) {
    const applicant = applicants.find((a) => a.Applicant_ID === applicantId);
    approveApplicant(applicantId);
    setApprovingId(applicantId);
    if (applicant) await saveApplicantToExcel(applicant, "approved");
    setApprovingId(null);
  }

  async function handleApprove(applicantId) {
    const applicant = applicants.find((a) => a.Applicant_ID === applicantId);
    approveApplicant(applicantId);
    setReviewingId(null);
    if (applicant) await saveApplicantToExcel(applicant, "approved");
  }

  async function handleReject(applicantId) {
    const applicant = applicants.find((a) => a.Applicant_ID === applicantId);
    rejectApplicant(applicantId);
    setReviewingId(null);
    if (applicant) await saveApplicantToExcel(applicant, "rejected");
  }

  return (
    <div>
      <PageHeader
        eyebrow="Human-in-the-Loop"
        title="Pending Reviews"
        description="Every scored applicant lands here for an explicit Quick Approve, Manual Review, or Reject decision."
        right={<span className="text-xs tabular-nums text-zinc-500">{pending.length} awaiting decision</span>}
      />

      {pending.length === 0 ? (
        <div className="flex flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-zinc-800 py-32 text-center">
          <ClipboardCheck size={22} strokeWidth={1.5} className="text-zinc-500" />
          <p className="text-sm text-zinc-400">No applicants are currently flagged or pending review.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {pending.map((applicant, index) => (
            <PendingCard
              key={applicant.Applicant_ID}
              applicant={applicant}
              index={index}
              onQuickApprove={handleQuickApprove}
              onManualReview={setReviewingId}
              saving={approvingId === applicant.Applicant_ID}
            />
          ))}
        </div>
      )}

      {reviewingApplicant && (
        <ManualReviewModal
          applicant={reviewingApplicant}
          files={{
            salarySlip: reviewingApplicant.salarySlipUrl,
            bankStatement: reviewingApplicant.bankStatementUrl,
          }}
          onApprove={() => handleApprove(reviewingApplicant.Applicant_ID)}
          onReject={() => handleReject(reviewingApplicant.Applicant_ID)}
          onClose={() => setReviewingId(null)}
        />
      )}
    </div>
  );
}
