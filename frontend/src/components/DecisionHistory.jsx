import { useMemo, useRef, useState } from "react";
import { ArrowUpRight, History, Search } from "lucide-react";
import { motion, useReducedMotion } from "motion/react";
import { useLoanStore } from "../store/useLoanStore";
import StatusPill from "./StatusPill";
import PageHeader from "./PageHeader";

const DECISION_OPTIONS = ["All Decisions", "Approved", "Rejected"];
const EMPLOYMENT_OPTIONS = ["All Employment", "Salaried", "Self-Employed", "Business Owner"];

const inr = new Intl.NumberFormat("en-IN", {
  style: "currency",
  currency: "INR",
  maximumFractionDigits: 0,
});

const listVariants = {
  hidden: {},
  show: {
    transition: { staggerChildren: 0.035, delayChildren: 0.05 },
  },
};

const rowVariants = {
  hidden: { opacity: 0, y: 10 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: [0.16, 1, 0.3, 1] } },
};

function DecisionBadge({ status }) {
  const isApproved = status === "approved";
  return (
    <span
      className={
        "inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-medium tabular-nums " +
        (isApproved
          ? "bg-emerald-500/10 text-emerald-400 ring-1 ring-inset ring-emerald-500/20"
          : "bg-red-500/10 text-red-400 ring-1 ring-inset ring-red-500/20")
      }
    >
      {isApproved ? "Approved" : "Rejected"}
    </span>
  );
}

function DecisionRow({ applicant, onSelect, reduce }) {
  const ref = useRef(null);
  const [hovered, setHovered] = useState(false);

  function handlePointerMove(e) {
    const el = ref.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    el.style.setProperty("--x", `${e.clientX - rect.left}px`);
    el.style.setProperty("--y", `${e.clientY - rect.top}px`);
  }

  return (
    <motion.tr
      ref={ref}
      variants={rowVariants}
      onMouseMove={handlePointerMove}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      onClick={() => onSelect(applicant.Applicant_ID)}
      whileHover={reduce ? undefined : { scale: 1.012 }}
      transition={{ type: "spring", stiffness: 380, damping: 28 }}
      className="group relative z-0 cursor-pointer hover:z-10"
      style={{
        background: hovered
          ? "radial-gradient(280px circle at var(--x, 50%) var(--y, 50%), rgba(59,130,246,0.09), transparent 70%)"
          : "transparent",
      }}
    >
      <td className="rounded-l-lg px-4 py-3.5 font-mono tabular-nums text-zinc-300 transition-colors group-hover:text-zinc-100">
        {applicant.Applicant_ID}
      </td>
      <td className="px-4 py-3.5 text-zinc-400">{applicant.Employment_Type}</td>
      <td className="px-4 py-3.5 text-right font-mono tabular-nums text-zinc-300">
        {inr.format(applicant.Requested_Loan_Amount)}
      </td>
      <td className="px-4 py-3.5">
        <StatusPill verdict={applicant.Classification_Verdict} />
      </td>
      <td className="px-4 py-3.5">
        <DecisionBadge status={applicant.Review_Status} />
      </td>
      <td className="rounded-r-lg px-3 py-3.5">
        <button
          onClick={(e) => {
            e.stopPropagation();
            onSelect(applicant.Applicant_ID);
          }}
          className="flex items-center gap-1 whitespace-nowrap rounded-md px-2 py-1 text-xs font-medium text-zinc-500 opacity-0 transition-all group-hover:opacity-100 hover:bg-accent/10 hover:text-accent"
        >
          Review Insights
          <ArrowUpRight size={12} strokeWidth={2} />
        </button>
      </td>
    </motion.tr>
  );
}

export default function DecisionHistory() {
  const applicants = useLoanStore((state) => state.applicants);
  const selectApplicant = useLoanStore((state) => state.selectApplicant);
  const reduce = useReducedMotion();

  const [query, setQuery] = useState("");
  const [decisionFilter, setDecisionFilter] = useState(DECISION_OPTIONS[0]);
  const [employmentFilter, setEmploymentFilter] = useState(EMPLOYMENT_OPTIONS[0]);

  const decided = useMemo(
    () => applicants.filter((a) => a.Review_Status === "approved" || a.Review_Status === "rejected"),
    [applicants]
  );

  const filtered = useMemo(() => {
    return decided.filter((a) => {
      if (query.trim() && !a.Applicant_ID.toLowerCase().includes(query.trim().toLowerCase())) return false;
      if (decisionFilter === "Approved" && a.Review_Status !== "approved") return false;
      if (decisionFilter === "Rejected" && a.Review_Status !== "rejected") return false;
      if (employmentFilter !== "All Employment" && a.Employment_Type !== employmentFilter) return false;
      return true;
    });
  }, [decided, query, decisionFilter, employmentFilter]);

  return (
    <div className="w-full">
      <PageHeader
        eyebrow="Audit Trail"
        title="Decision History"
        description="Approved and rejected applications from Pending Reviews, this session only — nothing here is persisted, so this list clears on sign-out or page refresh."
        right={
          <span className="rounded-full border border-zinc-800 bg-zinc-900/60 px-3 py-1.5 font-mono text-xs tabular-nums text-zinc-300">
            {filtered.length} of {decided.length} records
          </span>
        }
      />

      <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-center">
        <div className="relative flex-1">
          <Search size={14} strokeWidth={2} className="absolute left-3 top-1/2 -translate-y-1/2 text-zinc-500" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search by Applicant ID..."
            className="w-full rounded-lg border border-zinc-800 bg-zinc-950/60 py-2 pl-9 pr-3 text-sm text-zinc-100 outline-none transition-colors placeholder:text-zinc-600 focus:border-accent"
          />
        </div>
        <select
          value={decisionFilter}
          onChange={(e) => setDecisionFilter(e.target.value)}
          className="rounded-lg border border-zinc-800 bg-zinc-950/60 px-3 py-2 text-sm text-zinc-300 outline-none transition-colors focus:border-accent"
        >
          {DECISION_OPTIONS.map((opt) => (
            <option key={opt} value={opt} className="bg-zinc-900">
              {opt}
            </option>
          ))}
        </select>
        <select
          value={employmentFilter}
          onChange={(e) => setEmploymentFilter(e.target.value)}
          className="rounded-lg border border-zinc-800 bg-zinc-950/60 px-3 py-2 text-sm text-zinc-300 outline-none transition-colors focus:border-accent"
        >
          {EMPLOYMENT_OPTIONS.map((opt) => (
            <option key={opt} value={opt} className="bg-zinc-900">
              {opt}
            </option>
          ))}
        </select>
      </div>

      {decided.length === 0 ? (
        <div className="flex flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-zinc-800 py-32 text-center">
          <History size={22} strokeWidth={1.5} className="text-zinc-500" />
          <p className="text-sm text-zinc-400">
            No decisions yet this session — approve or reject an applicant from Pending Reviews to see it here.
          </p>
        </div>
      ) : (
        <div className="overflow-x-auto rounded-xl border border-zinc-800/80 p-1">
          <table className="w-full border-collapse text-left text-sm">
            <thead>
              <tr className="text-[11px] uppercase tracking-wide text-zinc-500">
                <th className="px-4 py-3 font-medium">Applicant ID</th>
                <th className="px-4 py-3 font-medium">Employment</th>
                <th className="px-4 py-3 text-right font-medium">Loan Amount</th>
                <th className="px-4 py-3 font-medium">Model Verdict</th>
                <th className="px-4 py-3 font-medium">Decision</th>
                <th className="w-6 px-3 py-3" />
              </tr>
            </thead>
            <motion.tbody
              key={`${query}-${decisionFilter}-${employmentFilter}`}
              variants={reduce ? undefined : listVariants}
              initial={reduce ? undefined : "hidden"}
              animate={reduce ? undefined : "show"}
              className="divide-y divide-zinc-800/50"
            >
              {filtered.map((applicant) => (
                <DecisionRow
                  key={applicant.Applicant_ID}
                  applicant={applicant}
                  onSelect={selectApplicant}
                  reduce={reduce}
                />
              ))}
            </motion.tbody>
          </table>
          {filtered.length === 0 && (
            <p className="px-4 py-8 text-center text-sm text-zinc-500">No decisions match the current filters.</p>
          )}
        </div>
      )}
    </div>
  );
}
