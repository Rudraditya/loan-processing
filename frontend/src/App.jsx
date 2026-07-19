import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { useLoanStore } from "./store/useLoanStore";
import LoanLedger from "./components/LoanLedger";
import CustomerInsights from "./components/CustomerInsights";
import ArchitecturePipeline from "./components/ArchitecturePipeline";
import Upload from "./components/Upload";

const TABS = [
  { id: "upload", label: "New Application" },
  { id: "ledger", label: "Ledger" },
  { id: "insights", label: "Insights" },
  { id: "pipeline", label: "Pipeline" },
];

const PAGES = {
  ledger: LoanLedger,
  insights: CustomerInsights,
  pipeline: ArchitecturePipeline,
  upload: Upload,
};

export default function App() {
  const currentTab = useLoanStore((state) => state.currentTab);
  const setCurrentTab = useLoanStore((state) => state.setCurrentTab);
  const applicants = useLoanStore((state) => state.applicants);
  const reduce = useReducedMotion();
  const ActivePage = PAGES[currentTab];

  const pendingReviews = applicants.filter((a) => !a.Consistency_Flag).length;

  return (
    <div className="min-h-[100dvh] bg-zinc-950 text-zinc-100">
      <div className="mx-auto max-w-[1680px] px-8 py-6">
        <div className="mb-4 flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent/15 text-sm font-bold text-accent">
            L
          </div>
          <span className="text-sm font-semibold tracking-tight text-zinc-300">Loan Assessment Agent</span>
        </div>

        <div className="sticky top-4 z-50 mb-8 flex w-fit items-center gap-1 rounded-full bg-white/90 p-1.5 shadow-lg backdrop-blur-md mx-auto">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setCurrentTab(tab.id)}
              className={
                "rounded-full px-4 py-2 text-sm font-medium text-gray-700 transition-colors hover:bg-gray-200/50 " +
                (currentTab === tab.id ? "bg-gray-200/70 text-gray-900" : "")
              }
            >
              {tab.label}
            </button>
          ))}

          <button className="ml-1 flex items-center gap-1.5 rounded-full bg-black px-4 py-1.5 text-sm font-medium text-white">
            <span className="relative flex h-1.5 w-1.5">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
              <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-emerald-500" />
            </span>
            {pendingReviews > 0 ? `${pendingReviews} Pending Review${pendingReviews === 1 ? "" : "s"}` : "System Online"}
          </button>
        </div>

        <AnimatePresence mode="wait">
          <motion.div
            key={currentTab}
            initial={reduce ? false : { opacity: 0, y: 14, scale: 0.99 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={reduce ? undefined : { opacity: 0, y: -10, scale: 0.99 }}
            transition={{ type: "spring", stiffness: 300, damping: 28, mass: 0.7 }}
          >
            <ActivePage />
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}
