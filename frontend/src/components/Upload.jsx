import { useRef, useState } from "react";
import { motion } from "motion/react";
import {
  AlertTriangle,
  Briefcase,
  Calendar,
  CheckCircle2,
  Clock,
  FileCheck,
  FileSpreadsheet,
  FileText,
  Hash,
  IndianRupee,
  Loader2,
  Play,
  ShieldAlert,
  UploadCloud,
  User,
  X,
} from "lucide-react";
import GlowCard from "./GlowCard";
import PageHeader from "./PageHeader";
import { useLoanStore } from "../store/useLoanStore";

const TENURE_OPTIONS_MONTHS = [12, 24, 36, 48, 60, 72, 84];
const EMPLOYMENT_STATUS_OPTIONS = ["Salaried", "Self-Employed", "Business Owner"];

const EXTRACTION_FIELD_LABELS = {
  Full_Name: "Full Name",
  Monthly_Net_Income: "Monthly Net Income",
  Gross_Income: "Gross Income",
  Total_Deductions: "Total Deductions",
  Total_Existing_EMIs: "Existing EMIs",
  Average_Monthly_Bank_Balance: "Avg. Monthly Balance",
  Number_of_Bounced_Transactions_Last_6M: "Bounced Transactions (6M)",
};

// Birthday-aware whole-years calculation, mirroring
// record_builder.calculate_age() - purely a live UI preview; the backend
// recomputes this itself from the same date_of_birth value rather than
// trusting anything computed client-side.
function previewAge(dateOfBirthValue) {
  if (!dateOfBirthValue) return null;
  const dob = new Date(dateOfBirthValue);
  if (Number.isNaN(dob.getTime())) return null;

  const today = new Date();
  let years = today.getFullYear() - dob.getFullYear();
  const hadBirthdayThisYear =
    today.getMonth() > dob.getMonth() || (today.getMonth() === dob.getMonth() && today.getDate() >= dob.getDate());
  if (!hadBirthdayThisYear) years -= 1;
  return years;
}

const METHOD_LABELS = {
  regex: { text: "via Regex", className: "bg-zinc-800 text-zinc-400" },
  gemini: { text: "via Gemini (AI fallback)", className: "bg-purple-500/10 text-purple-300 ring-1 ring-inset ring-purple-500/25" },
};

// Gemini-internal metadata (which model tier actually served the request,
// and whether the Pro tier had to fall back to Flash on a quota error) -
// surfaced as a badge, not as a row in the generic extracted-fields list.
const GEMINI_METADATA_KEYS = ["model_used", "fallback_triggered"];

function friendlyModelName(modelName) {
  if (!modelName) return modelName;
  return modelName
    .split("-")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function ExtractionResultCard({ result }) {
  const methodBadge = result.method ? METHOD_LABELS[result.method] : null;
  const fallbackTriggered = result.data?.fallback_triggered === true;
  const modelUsed = result.data?.model_used;
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-950/40 p-3.5">
      <div className="mb-2.5 flex items-center justify-between">
        <span className="flex items-center gap-2 text-[11px] font-medium uppercase tracking-wide text-zinc-500">
          Extraction Preview
          {methodBadge && (
            <span className={"rounded-full px-2 py-0.5 text-[10px] font-medium normal-case tracking-normal " + methodBadge.className}>
              {methodBadge.text}
            </span>
          )}
          {fallbackTriggered && (
            <span className="rounded-full bg-amber-500/10 px-2 py-0.5 text-[10px] font-medium normal-case tracking-normal text-amber-300 ring-1 ring-inset ring-amber-500/25">
              Processed via {friendlyModelName(modelUsed)} [Fallback]
            </span>
          )}
        </span>
        <span className="font-mono text-[11px] tabular-nums text-zinc-600">
          {result.elapsed_seconds.toFixed(2)}s
        </span>
      </div>
      {result.error ? (
        <div className="flex items-start gap-1.5 text-[12px] text-amber-400">
          <AlertTriangle size={13} strokeWidth={2} className="mt-0.5 shrink-0" />
          <span>{result.error}</span>
        </div>
      ) : (
        <div className="space-y-1.5">
          {Object.entries(result.data ?? {})
            .filter(([key, value]) => !Array.isArray(value) && !GEMINI_METADATA_KEYS.includes(key))
            .map(([key, value]) => (
              <div key={key} className="flex items-center justify-between gap-3 text-[12px]">
                <span className="text-zinc-500">{EXTRACTION_FIELD_LABELS[key] ?? key}</span>
                <span className="truncate font-mono tabular-nums text-zinc-200">{String(value)}</span>
              </div>
            ))}
        </div>
      )}
    </div>
  );
}

function FileSlot({ label, accept, file, onFile, onClear, icon: Icon, tone }) {
  const inputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragOver(false);
        const dropped = e.dataTransfer.files?.[0];
        if (dropped) onFile(dropped);
      }}
      className={
        "rounded-lg border p-3.5 transition-colors " +
        (dragOver ? "border-accent bg-accent/5" : "border-zinc-800 bg-zinc-950/40")
      }
    >
      <div className="mb-2 flex items-center justify-between">
        <div className="flex items-center gap-2 text-[11px] font-medium uppercase tracking-wide text-zinc-500">
          <Icon size={13} strokeWidth={2} className={tone} />
          {label}
        </div>
        {file && (
          <button
            onClick={() => {
              onClear();
              if (inputRef.current) inputRef.current.value = "";
            }}
            className="text-zinc-600 transition-colors hover:text-red-400"
          >
            <X size={13} strokeWidth={2} />
          </button>
        )}
      </div>

      {file ? (
        <p className="truncate text-[13px] text-zinc-200">{file.name}</p>
      ) : (
        <button
          onClick={() => inputRef.current?.click()}
          className="text-[13px] text-zinc-600 transition-colors hover:text-accent"
        >
          Click to browse, or drop a file directly on this box
        </button>
      )}

      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="hidden"
        onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])}
      />
    </div>
  );
}

function Toggle({ checked, onChange, label, description }) {
  return (
    <div className="flex items-center justify-between gap-4 rounded-lg border border-zinc-800 bg-zinc-950/40 px-4 py-3">
      <div>
        <div className="text-sm font-medium text-zinc-200">{label}</div>
        {description && <div className="mt-0.5 text-xs text-zinc-500">{description}</div>}
      </div>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={
          "relative h-6 w-11 shrink-0 rounded-full transition-colors " +
          (checked ? "bg-accent" : "bg-zinc-700")
        }
      >
        <motion.span
          className="absolute left-0.5 top-0.5 h-5 w-5 rounded-full bg-white shadow"
          animate={{ x: checked ? 20 : 0 }}
          transition={{ type: "spring", stiffness: 500, damping: 30 }}
        />
      </button>
    </div>
  );
}

export default function Upload() {
  const addApplicant = useLoanStore((state) => state.addApplicant);
  const setCurrentTab = useLoanStore((state) => state.setCurrentTab);

  const [applicantId, setApplicantId] = useState("");
  const [applicantName, setApplicantName] = useState("");
  const [dateOfBirth, setDateOfBirth] = useState("");
  const [loanAmount, setLoanAmount] = useState("");
  const [tenureMonths, setTenureMonths] = useState("");
  const [employmentStatus, setEmploymentStatus] = useState("");
  const [isFirstLoan, setIsFirstLoan] = useState(false);
  const [salarySlip, setSalarySlip] = useState(null);
  const [bankStatement, setBankStatement] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [status, setStatus] = useState("idle"); // idle | running | done | failed
  const [extractionResult, setExtractionResult] = useState(null);
  const [submitError, setSubmitError] = useState(null);

  const [scoringStatus, setScoringStatus] = useState("idle"); // idle | running | done | failed
  const [scoringError, setScoringError] = useState(null);
  const [scoredApplicant, setScoredApplicant] = useState(null);

  const canSubmit =
    applicantId.trim().length > 0 &&
    applicantName.trim().length > 0 &&
    dateOfBirth.trim().length > 0 &&
    loanAmount.trim().length > 0 &&
    tenureMonths.trim().length > 0 &&
    salarySlip !== null &&
    bankStatement !== null &&
    status !== "running";

  // Salary slip accepts PDF/JPG (Gemini can read a photographed/scanned
  // slip) and bank statement accepts XLSX/XLS/PDF, so a bare PDF drop on
  // the shared zone can't tell which one it is - it defaults a dropped PDF
  // to the salary slip. Drop directly on either box below to be unambiguous.
  function assignDroppedFile(file) {
    if (/\.(xlsx|xls)$/i.test(file.name)) setBankStatement(file);
    else if (/\.(pdf|jpe?g)$/i.test(file.name)) setSalarySlip(file);
  }

  function handleDrop(e) {
    e.preventDefault();
    setIsDragging(false);
    Array.from(e.dataTransfer.files).forEach(assignDroppedFile);
  }

  async function handleSubmit() {
    if (!canSubmit) return;

    setStatus("running");
    setSubmitError(null);
    setExtractionResult(null);

    const formData = new FormData();
    formData.append("salary_slip", salarySlip);
    formData.append("bank_statement", bankStatement);

    try {
      const response = await fetch("/api/extraction/preview", {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        throw new Error(`Extraction API returned ${response.status}`);
      }
      const result = await response.json();
      setExtractionResult(result);
      setStatus("done");
    } catch (err) {
      setSubmitError(
        err instanceof TypeError
          ? "Could not reach the extraction API. Is agentic_api running (uvicorn loan_processing.agentic_api.main:app --app-dir src --reload)?"
          : err.message
      );
      setStatus("failed");
    }
  }

  // Runs after extraction so the preview can be inspected first: scores the
  // same documents against the production risk model (via /scoring/assess,
  // not the extraction-only /extraction/preview call above) and pushes the
  // result into the Ledger.
  async function handleAddToLedger() {
    setScoringStatus("running");
    setScoringError(null);
    setScoredApplicant(null);

    const formData = new FormData();
    formData.append("salary_slip", salarySlip);
    formData.append("bank_statement", bankStatement);
    formData.append("applicant_id", applicantId.trim());
    formData.append("applicant_name", applicantName.trim());
    formData.append("date_of_birth", dateOfBirth);
    formData.append("employment_status", employmentStatus);
    formData.append("requested_loan_amount", loanAmount);
    formData.append("is_first_loan", isFirstLoan ? "true" : "false");
    formData.append("requested_tenure_months", tenureMonths);

    try {
      const response = await fetch("/api/scoring/assess", {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.detail || `Scoring API returned ${response.status}`);
      }
      const applicant = await response.json();
      // Object URLs, not the files themselves, are what gets retained - so
      // the Manual Review drawer can show the real source documents for as
      // long as the session stays active (see useLoanStore.js's
      // clearSessionDocuments, called on logout).
      addApplicant({
        ...applicant,
        salarySlipUrl: URL.createObjectURL(salarySlip),
        bankStatementUrl: URL.createObjectURL(bankStatement),
      });
      setScoredApplicant(applicant);
      setScoringStatus("done");
      // Brief pause so the confirmation is actually visible before the tab
      // switch unmounts this view.
      setTimeout(() => setCurrentTab("ledger"), 1200);
    } catch (err) {
      setScoringError(
        err instanceof TypeError
          ? "Could not reach the scoring API. Is agentic_api running (uvicorn loan_processing.agentic_api.main:app --app-dir src --reload)?"
          : err.message
      );
      setScoringStatus("failed");
    }
  }

  return (
    <div>
      <PageHeader
        eyebrow="Intake"
        title="New Application"
        description="Submit applicant details and source documents to run the assessment pipeline."
      />

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        {/* Left: manual input form */}
        <GlowCard index={0} className="p-7">
          <div className="mb-5 flex items-center gap-2 border-b border-zinc-800/60 pb-3">
            <User size={14} strokeWidth={2} className="text-accent" />
            <span className="text-[11px] font-bold uppercase tracking-wide text-zinc-400">Applicant Details</span>
          </div>
          <div className="space-y-5">
            <div>
              <label className="mb-1.5 flex items-center gap-1.5 text-xs text-zinc-400">
                <Hash size={13} strokeWidth={2} />
                Applicant ID
              </label>
              <input
                type="text"
                value={applicantId}
                onChange={(e) => setApplicantId(e.target.value)}
                placeholder="e.g. APP-10432"
                className="w-full rounded-md border border-zinc-800 bg-zinc-950/60 px-3 py-2 text-sm text-zinc-100 outline-none transition-colors placeholder:text-zinc-600 focus:border-accent"
              />
              <p className="mt-1.5 text-[11px] text-zinc-600">
                Real-world documents don't carry a machine-readable ID — assign one to identify this record
                in the Ledger.
              </p>
            </div>
            <div>
              <label className="mb-1.5 flex items-center gap-1.5 text-xs text-zinc-400">
                <User size={13} strokeWidth={2} />
                Applicant Name
              </label>
              <input
                type="text"
                value={applicantName}
                onChange={(e) => setApplicantName(e.target.value)}
                placeholder="e.g. Priya Sharma"
                className="w-full rounded-md border border-zinc-800 bg-zinc-950/60 px-3 py-2 text-sm text-zinc-100 outline-none transition-colors placeholder:text-zinc-600 focus:border-accent"
              />
              <p className="mt-1.5 text-[11px] text-zinc-600">
                Must match the name on the uploaded salary slip exactly (case/spacing-insensitive) — a
                mismatch blocks submission until corrected.
              </p>
            </div>
            <div>
              <label className="mb-1.5 flex items-center gap-1.5 text-xs text-zinc-400">
                <Calendar size={13} strokeWidth={2} />
                Date of Birth
              </label>
              <input
                type="date"
                value={dateOfBirth}
                max={new Date().toISOString().slice(0, 10)}
                onChange={(e) => setDateOfBirth(e.target.value)}
                className="w-full rounded-md border border-zinc-800 bg-zinc-950/60 px-3 py-2 text-sm text-zinc-100 outline-none transition-colors focus:border-accent [color-scheme:dark]"
              />
              <p className="mt-1.5 text-[11px] text-zinc-600">
                {previewAge(dateOfBirth) !== null
                  ? `Age used for assessment: ${previewAge(dateOfBirth)} (calculated as of today, not read from the documents).`
                  : "Age is calculated from this as of today's date, not extracted from the documents."}
              </p>
            </div>
            <div>
              <label className="mb-1.5 flex items-center gap-1.5 text-xs text-zinc-400">
                <IndianRupee size={13} strokeWidth={2} />
                Loan Amount Requested
              </label>
              <input
                type="number"
                min="0"
                value={loanAmount}
                onChange={(e) => setLoanAmount(e.target.value)}
                placeholder="e.g. 1500000"
                className="w-full rounded-md border border-zinc-800 bg-zinc-950/60 px-3 py-2 text-sm tabular-nums text-zinc-100 outline-none transition-colors placeholder:text-zinc-600 focus:border-accent"
              />
            </div>
            <div>
              <label className="mb-1.5 flex items-center gap-1.5 text-xs text-zinc-400">
                <Clock size={13} strokeWidth={2} />
                Requested Tenure
              </label>
              <select
                required
                value={tenureMonths}
                onChange={(e) => setTenureMonths(e.target.value)}
                className="w-full rounded-md border border-zinc-800 bg-zinc-950/60 px-3 py-2 text-sm text-zinc-100 outline-none transition-colors focus:border-accent"
              >
                <option value="" disabled className="bg-zinc-900">
                  Select tenure...
                </option>
                {TENURE_OPTIONS_MONTHS.map((months) => (
                  <option key={months} value={months} className="bg-zinc-900">
                    {months} months
                  </option>
                ))}
              </select>
              <p className="mt-1.5 text-[11px] text-zinc-600">
                Drives one of the model's strongest signals (loan-affordability vs. income) - always required,
                never left to a default.
              </p>
            </div>
            <div>
              <label className="mb-1.5 flex items-center gap-1.5 text-xs text-zinc-400">
                <Briefcase size={13} strokeWidth={2} />
                Employment Status
              </label>
              <select
                required
                value={employmentStatus}
                onChange={(e) => setEmploymentStatus(e.target.value)}
                className="w-full rounded-md border border-zinc-800 bg-zinc-950/60 px-3 py-2 text-sm text-zinc-100 outline-none transition-colors focus:border-accent"
              >
                <option value="" disabled className="bg-zinc-900">
                  Select employment status...
                </option>
                {EMPLOYMENT_STATUS_OPTIONS.map((option) => (
                  <option key={option} value={option} className="bg-zinc-900">
                    {option}
                  </option>
                ))}
              </select>
              <p className="mt-1.5 text-[11px] text-zinc-600">
                Self-reported, not read from the salary slip - the applicant's own selection here is
                what feeds the risk model.
              </p>
            </div>

            <Toggle
              checked={isFirstLoan}
              onChange={setIsFirstLoan}
              label="Is this your first loan?"
              description="Flags the applicant as new-to-credit (NTC) - no repayment track record yet."
            />
            {isFirstLoan && (
              <span className="inline-flex items-center gap-1.5 rounded-full bg-amber-500/10 px-2.5 py-1 text-[11px] font-medium text-amber-400 ring-1 ring-inset ring-amber-500/20">
                <ShieldAlert size={11} strokeWidth={2} />
                New to Credit (NTC) Mode Active
              </span>
            )}
          </div>
        </GlowCard>

        {/* Right: drag-and-drop zone */}
        <GlowCard index={1} className="p-7">
          <div className="mb-5 flex items-center gap-2 border-b border-zinc-800/60 pb-3">
            <FileCheck size={14} strokeWidth={2} className="text-accent" />
            <span className="text-[11px] font-bold uppercase tracking-wide text-zinc-400">Source Documents</span>
          </div>
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            className={
              "flex flex-col items-center justify-center gap-4 rounded-lg border-2 border-dashed p-8 text-center transition-colors " +
              (isDragging ? "border-accent bg-accent/5" : "border-zinc-800")
            }
          >
            <UploadCloud size={40} strokeWidth={1.5} className="text-zinc-600" />
            <p className="text-[13px] text-zinc-500">
              Drop a file here, or on the matching box below - drop directly on a box if both files are the
              same type (e.g. two PDFs)
            </p>

            <div className="grid w-full grid-cols-1 gap-3 sm:grid-cols-2">
              <FileSlot
                label="Salary Slip (PDF/JPG)"
                accept=".pdf,.jpg,.jpeg"
                file={salarySlip}
                onFile={setSalarySlip}
                onClear={() => setSalarySlip(null)}
                icon={FileText}
                tone="text-red-400"
              />
              <FileSlot
                label="Bank Statement (XLSX/PDF)"
                accept=".xlsx,.xls,.pdf"
                file={bankStatement}
                onFile={setBankStatement}
                onClear={() => setBankStatement(null)}
                icon={FileSpreadsheet}
                tone="text-emerald-400"
              />
            </div>
          </div>
        </GlowCard>
      </div>

      <div className="mt-6 flex flex-col items-center gap-3">
        <button
          onClick={handleSubmit}
          disabled={!canSubmit}
          className={
            "flex items-center gap-2.5 rounded-xl px-6 py-3.5 text-sm font-semibold tracking-wide shadow-[0_4px_15px_rgba(59,130,246,0.25)] transition-all active:scale-[0.98] " +
            (canSubmit
              ? "bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 text-white hover:opacity-95"
              : "cursor-not-allowed bg-zinc-900 text-zinc-600 shadow-none")
          }
        >
          {status === "running" ? (
            <Loader2 size={16} strokeWidth={2} className="animate-spin" />
          ) : (
            <Play size={14} strokeWidth={2} className="fill-current" />
          )}
          {status === "running" ? "Running Extraction..." : "Run Document Extraction"}
        </button>
        {!salarySlip || !bankStatement ? (
          <p className="text-[11px] text-zinc-600">Both a salary slip and a bank statement are required.</p>
        ) : null}

        {status === "running" && (
          <div className="w-full max-w-2xl rounded-lg border border-zinc-800 bg-black/60 p-4 font-mono text-xs leading-relaxed text-emerald-400/90">
            <p>&gt; Uploading salary slip and bank statement...</p>
            <p>&gt; Awaiting agentic_api extraction service (regex, Gemini fallback if needed)...</p>
            <p className="inline-flex items-center gap-1">
              &gt; <span className="h-3 w-1.5 animate-pulse bg-emerald-400/90" />
            </p>
          </div>
        )}

        {status === "failed" && submitError && (
          <div className="flex w-full max-w-2xl items-start gap-2 rounded-md bg-red-500/10 px-3.5 py-2.5 text-[12px] text-red-400 ring-1 ring-inset ring-red-500/20">
            <AlertTriangle size={14} strokeWidth={2} className="mt-0.5 shrink-0" />
            <span>{submitError}</span>
          </div>
        )}

        {status === "done" && extractionResult && (
          <div className="w-full max-w-2xl">
            <div className="mb-3 flex items-center gap-2 rounded-md bg-emerald-500/10 px-3.5 py-2.5 text-[12px] text-emerald-400 ring-1 ring-inset ring-emerald-500/20">
              <CheckCircle2 size={14} strokeWidth={2} className="shrink-0" />
              <span>
                Extraction complete for {applicantName}. Review the recovered fields below, then score the
                applicant against the production risk model to add them to the Ledger.
              </span>
            </div>
            <ExtractionResultCard result={extractionResult} />

            <div className="mt-4 flex flex-col items-center gap-3">
              <button
                onClick={handleAddToLedger}
                disabled={scoringStatus === "running"}
                className={
                  "flex items-center gap-2.5 rounded-xl px-6 py-3.5 text-sm font-semibold tracking-wide shadow-[0_4px_15px_rgba(16,185,129,0.25)] transition-all active:scale-[0.98] " +
                  (scoringStatus === "running"
                    ? "cursor-not-allowed bg-zinc-900 text-zinc-600 shadow-none"
                    : "bg-gradient-to-r from-emerald-600 via-teal-600 to-cyan-600 text-white hover:opacity-95")
                }
              >
                {scoringStatus === "running" ? (
                  <Loader2 size={16} strokeWidth={2} className="animate-spin" />
                ) : (
                  <CheckCircle2 size={14} strokeWidth={2} />
                )}
                {scoringStatus === "running" ? "Scoring Applicant..." : "Score & Add to Ledger"}
              </button>

              {scoringStatus === "failed" && scoringError && (
                <div className="flex w-full items-start gap-2 rounded-md bg-red-500/10 px-3.5 py-2.5 text-[12px] text-red-400 ring-1 ring-inset ring-red-500/20">
                  <AlertTriangle size={14} strokeWidth={2} className="mt-0.5 shrink-0" />
                  <span>{scoringError}</span>
                </div>
              )}

              {scoringStatus === "done" && scoredApplicant && (
                <div className="flex w-full flex-col gap-1 rounded-md bg-emerald-500/10 px-3.5 py-2.5 text-[12px] text-emerald-400 ring-1 ring-inset ring-emerald-500/20">
                  <span>
                    Added <span className="font-mono">{scoredApplicant.Applicant_ID}</span> to the Ledger.
                    Default probability: {(scoredApplicant.Default_Probability * 100).toFixed(2)}%.
                  </span>
                  {scoredApplicant.Missing_Fields?.length > 0 && (
                    <span className="text-amber-400">
                      Flagged for manual review — missing: {scoredApplicant.Missing_Fields.join(", ")}.
                    </span>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
