import { useRef, useState } from "react";
import { motion } from "motion/react";
import {
  AlertTriangle,
  CheckCircle2,
  FileSpreadsheet,
  FileText,
  IndianRupee,
  Loader2,
  ShieldAlert,
  UploadCloud,
  User,
  X,
} from "lucide-react";
import GlowCard from "./GlowCard";

const EXTRACTION_FIELD_LABELS = {
  Applicant_ID: "Applicant ID",
  Age: "Age",
  Employment_Type: "Employment Type",
  Monthly_Net_Income: "Monthly Net Income",
  Total_Existing_EMIs: "Existing EMIs",
  Average_Monthly_Bank_Balance: "Avg. Monthly Balance",
  Number_of_Bounced_Transactions_Last_6M: "Bounced Transactions (6M)",
};

function ExtractionResultCard({ title, result }) {
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-950/40 p-3.5">
      <div className="mb-2.5 flex items-center justify-between">
        <span className="text-[11px] font-medium uppercase tracking-wide text-zinc-500">{title}</span>
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
          {Object.entries(result.data ?? {}).map(([key, value]) => (
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
  const [applicantName, setApplicantName] = useState("");
  const [loanAmount, setLoanAmount] = useState("");
  const [isFirstLoan, setIsFirstLoan] = useState(false);
  const [salarySlip, setSalarySlip] = useState(null);
  const [bankStatement, setBankStatement] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [status, setStatus] = useState("idle"); // idle | running | done | failed
  const [benchmarkResult, setBenchmarkResult] = useState(null);
  const [submitError, setSubmitError] = useState(null);

  const canSubmit =
    applicantName.trim().length > 0 &&
    loanAmount.trim().length > 0 &&
    salarySlip !== null &&
    bankStatement !== null &&
    status !== "running";

  // Both slots now accept PDF, so a bare drop on the shared zone can't tell
  // a PDF salary slip from a PDF bank statement - it defaults .pdf/.jpg to
  // the salary slip. Drop directly on either box below to be unambiguous.
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
    setBenchmarkResult(null);

    const formData = new FormData();
    formData.append("salary_slip", salarySlip);
    formData.append("bank_statement", bankStatement);

    try {
      const response = await fetch("/api/extraction/benchmark", {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        throw new Error(`Extraction API returned ${response.status}`);
      }
      const result = await response.json();
      setBenchmarkResult(result);
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

  return (
    <div>
      <div className="mb-6 px-1">
        <h1 className="text-base font-semibold text-zinc-100">New Application</h1>
        <p className="mt-1 text-sm text-zinc-500">
          Submit applicant details and source documents to run the assessment pipeline.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        {/* Left: manual input form */}
        <GlowCard index={0} className="p-7">
          <div className="mb-5 text-[11px] font-medium uppercase tracking-wide text-zinc-500">
            Applicant Details
          </div>
          <div className="space-y-5">
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
          <div className="mb-5 text-[11px] font-medium uppercase tracking-wide text-zinc-500">
            Source Documents
          </div>
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            className={
              "flex flex-col items-center justify-center gap-4 rounded-lg border-2 border-dashed p-6 text-center transition-colors " +
              (isDragging ? "border-accent bg-accent/5" : "border-zinc-800")
            }
          >
            <UploadCloud size={26} strokeWidth={1.5} className="text-zinc-600" />
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
            "flex items-center gap-2 rounded-lg px-6 py-3 text-sm font-semibold tracking-wide transition-colors " +
            (canSubmit
              ? "bg-accent text-white hover:bg-accent/90"
              : "cursor-not-allowed bg-zinc-900 text-zinc-600")
          }
        >
          {status === "running" && <Loader2 size={16} strokeWidth={2} className="animate-spin" />}
          {status === "running" ? "Running Extraction..." : "Run Document Extraction"}
        </button>
        {!salarySlip || !bankStatement ? (
          <p className="text-[11px] text-zinc-600">Both a salary slip and a bank statement are required.</p>
        ) : null}

        {status === "failed" && submitError && (
          <div className="flex w-full max-w-2xl items-start gap-2 rounded-md bg-red-500/10 px-3.5 py-2.5 text-[12px] text-red-400 ring-1 ring-inset ring-red-500/20">
            <AlertTriangle size={14} strokeWidth={2} className="mt-0.5 shrink-0" />
            <span>{submitError}</span>
          </div>
        )}

        {status === "done" && benchmarkResult && (
          <div className="w-full max-w-2xl">
            <div className="mb-3 flex items-center gap-2 rounded-md bg-emerald-500/10 px-3.5 py-2.5 text-[12px] text-emerald-400 ring-1 ring-inset ring-emerald-500/20">
              <CheckCircle2 size={14} strokeWidth={2} className="shrink-0" />
              <span>
                Extraction complete for {applicantName}. Regex parsers and the Gemini extraction agent ran
                side-by-side against the uploaded documents — this is the extraction-benchmark result only;
                risk scoring is not yet wired up in this demo.
              </span>
            </div>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              <ExtractionResultCard title="Regex Extraction" result={benchmarkResult.regex} />
              <ExtractionResultCard title="Gemini Extraction" result={benchmarkResult.gemini} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
