import { useEffect, useState } from "react";
import { AlertTriangle, CheckCircle2, Download, FileSpreadsheet, FileText, X, XCircle } from "lucide-react";
import { motion, AnimatePresence } from "motion/react";
import * as XLSX from "xlsx";

const inr = new Intl.NumberFormat("en-IN", {
  style: "currency",
  currency: "INR",
  maximumFractionDigits: 0,
});

const VERIFICATION_STATUS = {
  matched: { icon: CheckCircle2, className: "text-emerald-400" },
  mismatch: { icon: AlertTriangle, className: "text-red-400" },
};

// Same fields CustomerInsights.jsx's cross-verification log covers,
// re-derived here since that component doesn't export the builder.
function buildVerificationLog(applicant) {
  return [
    {
      field: "Applicant Name",
      source: "Salary Slip",
      value: applicant.Extracted_Full_Name ?? "Not available",
      status: applicant.Name_Match === false ? "mismatch" : "matched",
    },
    {
      field: "Monthly Net Income",
      source: "Salary Slip",
      value: inr.format(applicant.Monthly_Net_Income),
      status: "matched",
    },
    {
      field: "Avg. Monthly Balance",
      source: "Bank Statement",
      value: inr.format(applicant.Average_Monthly_Bank_Balance),
      status: applicant.Bank_Consistency_Verified === false ? "mismatch" : "matched",
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
      value: applicant.Number_of_Bounced_Transactions_Last_6M ?? "Not available",
      status: applicant.Number_of_Bounced_Transactions_Last_6M >= 2 ? "mismatch" : "matched",
    },
  ];
}

// Browsers have no native XLSX viewer the way they do a PDF plugin, so an
// <object> embed just shows blank/broken for a spreadsheet - this parses
// the workbook client-side (SheetJS) and renders the first sheet as a
// plain table instead, so the bank statement is actually readable here
// rather than download-only.
function SpreadsheetPreview({ fileUrl }) {
  const [rows, setRows] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setRows(null);
    setError(null);

    fetch(fileUrl)
      .then((res) => res.arrayBuffer())
      .then((buffer) => {
        if (cancelled) return;
        const workbook = XLSX.read(buffer, { type: "array", cellDates: true });
        const sheet = workbook.Sheets[workbook.SheetNames[0]];
        const data = XLSX.utils.sheet_to_json(sheet, {
          header: 1,
          raw: false,
          defval: "",
          dateNF: "yyyy-mm-dd",
        });
        setRows(data);
      })
      .catch((err) => {
        if (!cancelled) setError(err?.message || "Could not read the spreadsheet.");
      });

    return () => {
      cancelled = true;
    };
  }, [fileUrl]);

  if (error) {
    return <p className="p-4 text-center text-[12px] text-red-400">{error}</p>;
  }
  if (!rows) {
    return <p className="p-4 text-center text-[12px] text-zinc-500">Loading spreadsheet…</p>;
  }

  return (
    <div className="max-h-64 w-full overflow-auto rounded border border-zinc-800/60">
      <table className="w-full border-collapse text-[11px]">
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="border-b border-zinc-800/60 last:border-0 even:bg-zinc-900/30">
              {row.map((cell, j) => (
                <td key={j} className="whitespace-nowrap px-2.5 py-1.5 text-zinc-300">
                  {cell === null || cell === undefined ? "" : String(cell)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DocumentPane({ label, icon: Icon, fileUrl, mimeType, kind = "pdf" }) {
  return (
    <div className="flex flex-col rounded-lg border border-zinc-800 bg-zinc-950/40">
      <div className="flex items-center justify-between gap-2 border-b border-zinc-800/60 px-3.5 py-2.5 text-[11px] font-medium uppercase tracking-wide text-zinc-400">
        <span className="flex items-center gap-2">
          <Icon size={13} strokeWidth={2} />
          {label}
        </span>
        {fileUrl && (
          <a href={fileUrl} download className="flex items-center gap-1 text-zinc-500 transition-colors hover:text-accent">
            <Download size={12} strokeWidth={2} />
            Download
          </a>
        )}
      </div>
      <div className="flex flex-1 items-center justify-center p-2">
        {!fileUrl ? (
          <p className="p-8 text-center text-[12px] text-zinc-500">
            Original document not retained for this session - re-upload to inspect the source file.
          </p>
        ) : kind === "spreadsheet" ? (
          <SpreadsheetPreview fileUrl={fileUrl} />
        ) : (
          <object data={fileUrl} type={mimeType} className="h-64 w-full rounded">
            <p className="p-4 text-center text-[12px] text-zinc-500">Preview unavailable for this file type.</p>
          </object>
        )}
      </div>
    </div>
  );
}

export default function ManualReviewModal({ applicant, files, onApprove, onReject, onClose }) {
  useEffect(() => {
    function handleKeyDown(e) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  const verificationLog = buildVerificationLog(applicant);

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        onClick={onClose}
        className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
      >
        <motion.div
          initial={{ opacity: 0, y: 16, scale: 0.98 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 10, scale: 0.98 }}
          transition={{ type: "spring", stiffness: 320, damping: 30 }}
          onClick={(e) => e.stopPropagation()}
          className="flex max-h-[88vh] w-full max-w-5xl flex-col overflow-hidden rounded-xl border border-zinc-800 bg-midnight-card shadow-2xl"
        >
          <div className="flex shrink-0 items-center justify-between border-b border-zinc-800/80 px-6 py-4">
            <div>
              <h2 className="font-display text-sm font-semibold text-zinc-100">Document Inspection</h2>
              <p className="mt-0.5 font-mono text-xs text-zinc-500">{applicant.Applicant_ID}</p>
            </div>
            <button
              onClick={onClose}
              className="rounded-lg p-1.5 text-zinc-400 transition-colors hover:bg-zinc-800/60 hover:text-white"
            >
              <X size={16} strokeWidth={2} />
            </button>
          </div>

          <div className="grid flex-1 grid-cols-1 gap-5 overflow-y-auto p-6 lg:grid-cols-2">
            {/* Left: document viewer */}
            <div className="space-y-3">
              <DocumentPane
                label="Salary Slip"
                icon={FileText}
                fileUrl={files?.salarySlip}
                mimeType="application/pdf"
              />
              <DocumentPane
                label="Bank Statement"
                icon={FileSpreadsheet}
                fileUrl={files?.bankStatement}
                kind="spreadsheet"
              />
            </div>

            {/* Right: verification details */}
            <div className="rounded-lg border border-zinc-800 bg-zinc-950/40 p-4">
              <div className="mb-3 text-[11px] font-bold uppercase tracking-wide text-zinc-300">
                Verification Details
              </div>
              <div className="space-y-3">
                {verificationLog.map((row) => {
                  const { icon: Icon, className } = VERIFICATION_STATUS[row.status];
                  return (
                    <div
                      key={row.field}
                      className={
                        "flex items-center gap-2.5 rounded-md px-2.5 py-2 text-[12px] " +
                        (row.status === "mismatch" ? "bg-red-500/5" : "")
                      }
                    >
                      <Icon size={13} strokeWidth={2} className={"shrink-0 " + className} />
                      <span className="w-40 shrink-0 text-zinc-300">{row.field}</span>
                      <span className="w-28 shrink-0 text-zinc-500">{row.source}</span>
                      <span
                        className={
                          "truncate font-mono tabular-nums " +
                          (row.status === "mismatch" ? "text-red-400" : "text-zinc-300")
                        }
                      >
                        {row.value}
                      </span>
                    </div>
                  );
                })}
              </div>

              {applicant.Top_Risk_Drivers.length > 0 && (
                <div className="mt-4 border-t border-zinc-800/60 pt-4">
                  <div className="mb-2 text-[10px] uppercase tracking-wide text-zinc-500">Key Indicators</div>
                  <div className="flex flex-wrap gap-1.5">
                    {applicant.Top_Risk_Drivers.map((driver) => (
                      <span
                        key={driver}
                        className="rounded-full border border-zinc-800 px-2.5 py-1 text-[11px] text-zinc-400"
                      >
                        {driver}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="flex shrink-0 items-center justify-end gap-3 border-t border-zinc-800/80 px-6 py-4">
            <button
              onClick={onReject}
              className="flex items-center gap-1.5 rounded-lg bg-red-500/10 px-4 py-2 text-[13px] font-medium text-red-400 ring-1 ring-inset ring-red-500/20 transition-colors hover:bg-red-500/20"
            >
              <XCircle size={14} strokeWidth={2} />
              Reject Application
            </button>
            <button
              onClick={onApprove}
              className="flex items-center gap-1.5 rounded-lg bg-emerald-500/10 px-4 py-2 text-[13px] font-medium text-emerald-400 ring-1 ring-inset ring-emerald-500/20 transition-colors hover:bg-emerald-500/20"
            >
              <CheckCircle2 size={14} strokeWidth={2} />
              Approve Application
            </button>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}
