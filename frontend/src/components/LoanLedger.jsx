import { useRef, useState } from "react";
import { ChevronRight } from "lucide-react";
import { motion, useReducedMotion } from "motion/react";
import { useLoanStore } from "../store/useLoanStore";
import StatusPill from "./StatusPill";

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

function LedgerRow({ applicant, onSelect, reduce }) {
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
        {inr.format(applicant.Monthly_Net_Income)}
      </td>
      <td className="px-4 py-3.5 text-right font-mono tabular-nums text-zinc-300">
        {inr.format(applicant.Requested_Loan_Amount)}
      </td>
      <td className="px-4 py-3.5">
        <StatusPill verdict={applicant.Classification_Verdict} />
      </td>
      <td className="rounded-r-lg px-3 py-3.5 text-zinc-600 transition-colors group-hover:text-accent">
        <ChevronRight size={14} strokeWidth={2} />
      </td>
    </motion.tr>
  );
}

export default function LoanLedger() {
  const applicants = useLoanStore((state) => state.applicants);
  const selectApplicant = useLoanStore((state) => state.selectApplicant);
  const reduce = useReducedMotion();

  return (
    <div className="w-full">
      <div className="mb-6 flex items-baseline justify-between px-1">
        <h1 className="text-base font-semibold text-zinc-100">Loan Application Ledger</h1>
        <span className="text-xs tabular-nums text-zinc-500">{applicants.length} records</span>
      </div>

      <div className="overflow-x-auto rounded-xl border border-zinc-800/80 p-1">
        <table className="w-full border-collapse text-left text-sm">
          <thead>
            <tr className="text-[11px] uppercase tracking-wide text-zinc-500">
              <th className="px-4 py-3 font-medium">Applicant ID</th>
              <th className="px-4 py-3 font-medium">Employment</th>
              <th className="px-4 py-3 text-right font-medium">Income</th>
              <th className="px-4 py-3 text-right font-medium">Loan Amount</th>
              <th className="px-4 py-3 font-medium">Status</th>
              <th className="w-6 px-3 py-3" />
            </tr>
          </thead>
          <motion.tbody
            variants={reduce ? undefined : listVariants}
            initial={reduce ? undefined : "hidden"}
            animate={reduce ? undefined : "show"}
            className="divide-y divide-zinc-800/50"
          >
            {applicants.map((applicant) => (
              <LedgerRow
                key={applicant.Applicant_ID}
                applicant={applicant}
                onSelect={selectApplicant}
                reduce={reduce}
              />
            ))}
          </motion.tbody>
        </table>
      </div>
    </div>
  );
}
