import { create } from "zustand";

// Every newly-scored applicant lands in Pending Reviews awaiting an explicit
// human decision - a verified, low-risk verdict is a strong recommendation
// (still shown via the risk meter and, in CustomerInsights.jsx,
// getRecommendation()'s "APPROVE - LOW RISK" label), not an auto-decision.
// This used to auto-approve consistent, low-risk applicants straight past
// Pending Reviews - a human could never accept/reject those, only ones the
// model or consistency check happened to flag. Every applicant now gets a
// real Quick Approve / Manual Review / Reject action regardless of what the
// risk meter shows; the underlying flagging logic that decides whether a
// warning banner appears (Consistency_Flag) or what the model recommends
// (Classification_Verdict) is unchanged.
function deriveReviewStatus() {
  return "pending";
}

// Uploaded documents are kept as blob: object URLs for the lifetime of the
// browser session (see Upload.jsx's handleAddToLedger, which creates them),
// so a reviewer can inspect the original file later via the Manual Review
// drawer - not persisted anywhere, and revoked here whenever a record is
// replaced or the session ends, so the underlying blob data doesn't leak.
function revokeApplicantDocumentUrls(applicant) {
  if (applicant?.salarySlipUrl) URL.revokeObjectURL(applicant.salarySlipUrl);
  if (applicant?.bankStatementUrl) URL.revokeObjectURL(applicant.bankStatementUrl);
}

export const useLoanStore = create((set) => ({
  currentTab: "dashboard",
  activeApplicantId: null,
  applicants: [],

  setCurrentTab: (tab) => set({ currentTab: tab }),

  selectApplicant: (applicantId) =>
    set({ activeApplicantId: applicantId, currentTab: "insights" }),

  // Adds a real, scored applicant (from POST /api/scoring/assess) to the
  // front of the ledger. Replaces any existing row with the same
  // Applicant_ID rather than duplicating it, since re-submitting the same
  // applicant's documents is expected to update their record in place - the
  // replaced row's document URLs are revoked so they don't leak.
  addApplicant: (applicant) =>
    set((state) => {
      const previous = state.applicants.find((a) => a.Applicant_ID === applicant.Applicant_ID);
      if (previous) revokeApplicantDocumentUrls(previous);
      return {
        applicants: [
          { ...applicant, Review_Status: deriveReviewStatus(applicant) },
          ...state.applicants.filter((a) => a.Applicant_ID !== applicant.Applicant_ID),
        ],
      };
    }),

  // Revokes every retained document's object URL and clears them from the
  // ledger rows - called on logout so uploaded documents don't outlive the
  // session, while leaving the rest of each scored record (and the Excel
  // log, which is server-side) intact.
  clearSessionDocuments: () =>
    set((state) => ({
      applicants: state.applicants.map((a) => {
        revokeApplicantDocumentUrls(a);
        const { salarySlipUrl, bankStatementUrl, ...rest } = a;
        return rest;
      }),
    })),

  approveApplicant: (applicantId) =>
    set((state) => ({
      applicants: state.applicants.map((a) =>
        a.Applicant_ID === applicantId
          ? { ...a, Review_Status: "approved", Classification_Verdict: 0 }
          : a
      ),
    })),

  rejectApplicant: (applicantId) =>
    set((state) => ({
      applicants: state.applicants.map((a) =>
        a.Applicant_ID === applicantId ? { ...a, Review_Status: "rejected" } : a
      ),
    })),
}));
