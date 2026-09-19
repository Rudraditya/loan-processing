// Fire-and-forget call to the Excel persistence backend (agentic_api's
// POST /save-applicant, proxied through Vite at /api/save-applicant - see
// vite.config.js). Never throws: a failed export shouldn't block the review
// decision that triggered it, so callers just get a boolean back and can
// surface it as a non-blocking toast/warning if they want to.
//
// No `Name` field exists on real applicant records (only Applicant_ID - see
// LoanLedger.jsx), so it isn't fabricated here either.
export async function saveApplicantToExcel(applicant, finalStatus) {
  try {
    const response = await fetch("/api/save-applicant", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        Applicant_ID: applicant.Applicant_ID,
        Employment_Type: applicant.Employment_Type,
        Monthly_Net_Income: applicant.Monthly_Net_Income,
        Requested_Loan_Amount: applicant.Requested_Loan_Amount,
        Default_Probability: applicant.Default_Probability,
        Classification_Verdict: applicant.Classification_Verdict,
        Final_Status: finalStatus,
      }),
    });
    return response.ok;
  } catch {
    return false;
  }
}
