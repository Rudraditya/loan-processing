import { useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { useLoanStore } from "./store/useLoanStore";
import Sidebar from "./components/Sidebar";
import Header from "./components/Header";
import Login from "./components/Login";
import DashboardOverview from "./components/DashboardOverview";
import LoanLedger from "./components/LoanLedger";
import CustomerInsights from "./components/CustomerInsights";
import ArchitecturePipeline from "./components/ArchitecturePipeline";
import Upload from "./components/Upload";
import PendingReviews from "./components/PendingReviews";
import DecisionHistory from "./components/DecisionHistory";

const PAGES = {
  dashboard: DashboardOverview,
  ledger: LoanLedger,
  insights: CustomerInsights,
  pipeline: ArchitecturePipeline,
  upload: Upload,
  "pending-reviews": PendingReviews,
  decisions: DecisionHistory,
};

export default function App() {
  const [username, setUsername] = useState(null);
  const currentTab = useLoanStore((state) => state.currentTab);
  const setCurrentTab = useLoanStore((state) => state.setCurrentTab);
  const applicants = useLoanStore((state) => state.applicants);
  const clearSessionDocuments = useLoanStore((state) => state.clearSessionDocuments);
  const reduce = useReducedMotion();
  const ActivePage = PAGES[currentTab];

  const pendingReviews = applicants.filter((a) => a.Review_Status === "pending").length;

  function handleLogout() {
    // Uploaded documents are only retained as blob: object URLs for the
    // duration of the session - revoke them (and strip the now-dead URLs
    // from each Ledger row) the moment the user signs out.
    clearSessionDocuments();
    setUsername(null);
  }

  if (!username) {
    return <Login onLogin={setUsername} />;
  }

  return (
    <div className="flex h-[100dvh] bg-midnight text-zinc-100">
      <Sidebar
        currentTab={currentTab}
        setCurrentTab={setCurrentTab}
        username={username}
        onLogout={handleLogout}
        pendingReviews={pendingReviews}
      />
      <div className="flex min-w-0 flex-1 flex-col">
        <Header pendingReviews={pendingReviews} onPendingReviewsClick={() => setCurrentTab("pending-reviews")} />
        <main className="flex-1 overflow-y-auto px-8 py-6">
          <div className="mx-auto max-w-[1680px]">
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
        </main>
      </div>
    </div>
  );
}
