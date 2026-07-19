import { create } from "zustand";

// Deterministic PRNG (mulberry32) so the mock ledger is stable across reloads,
// matching the seed=42 convention used by the Python data_generation pipeline.
function mulberry32(seed) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const rng = mulberry32(42);
const rand = (min, max) => min + rng() * (max - min);
const randInt = (min, max) => Math.floor(rand(min, max + 1));
const pick = (arr) => arr[randInt(0, arr.length - 1)];
const round = (n, step) => Math.round(n / step) * step;

const RISK_DRIVER_POOL = {
  highDti: "High Debt-to-Income Ratio",
  bounced: "Frequent Bounced Transactions",
  lowBalance: "Thin Bank Balance Buffer",
  highLti: "High Loan-to-Income Ratio",
  longTenure: "Extended Repayment Tenure Requested",
};

function buildApplicant(index) {
  const id = `APP-${String(index + 1).padStart(3, "0")}`;
  const employmentType = rng() < 0.68 ? "Salaried" : "Self-Employed";
  const age = randInt(23, 58);

  const incomeBase = employmentType === "Salaried" ? rand(28000, 210000) : rand(35000, 260000);
  const monthlyNetIncome = round(incomeBase, 500);

  const emiBurdenRatio = rand(0.05, 0.55);
  const totalExistingEmis = round(monthlyNetIncome * emiBurdenRatio, 250);

  const bouncedTx = rng() < 0.72 ? 0 : randInt(1, 5);

  const requestedLoanAmount = round(monthlyNetIncome * rand(6, 26), 5000);
  const requestedTenureMonths = pick([12, 24, 36, 48, 60, 72, 84]);

  const balanceRatio = rand(0.8, 3.6) * (1 - emiBurdenRatio * 0.4);
  const averageMonthlyBankBalance = round(monthlyNetIncome * Math.max(balanceRatio, 0.3), 250);

  const dti = totalExistingEmis / monthlyNetIncome;
  const lti = requestedLoanAmount / (monthlyNetIncome * 12);

  let riskScore =
    Math.min(dti, 1) * 0.47 +
    Math.min(bouncedTx / 5, 1) * 0.30 +
    Math.min(lti / 4, 1) * 0.23;
  riskScore = Math.max(0.02, Math.min(0.97, riskScore + rand(-0.06, 0.06)));

  const defaultProbability = Math.round(riskScore * 1000) / 1000;
  const classificationVerdict = defaultProbability >= 0.5 ? 1 : 0;

  const drivers = [];
  if (dti > 0.35) drivers.push(RISK_DRIVER_POOL.highDti);
  if (bouncedTx >= 2) drivers.push(RISK_DRIVER_POOL.bounced);
  if (averageMonthlyBankBalance < monthlyNetIncome * 0.7) drivers.push(RISK_DRIVER_POOL.lowBalance);
  if (lti > 3) drivers.push(RISK_DRIVER_POOL.highLti);
  if (requestedTenureMonths >= 72 && drivers.length < 2) drivers.push(RISK_DRIVER_POOL.longTenure);
  const topRiskDrivers = drivers.slice(0, 3);

  const extractionConfidence = Math.round(rand(89.5, 99.4) * 10) / 10;
  const consistencyFlag = rng() > 0.12;
  const processingLatency = Math.round(rand(1.1, 4.9) * 100) / 100;

  return {
    Applicant_ID: id,
    Age: age,
    Employment_Type: employmentType,
    Monthly_Net_Income: monthlyNetIncome,
    Total_Existing_EMIs: totalExistingEmis,
    Requested_Loan_Amount: requestedLoanAmount,
    Requested_Tenure_Months: requestedTenureMonths,
    Average_Monthly_Bank_Balance: averageMonthlyBankBalance,
    Number_of_Bounced_Transactions_Last_6M: bouncedTx,
    Extraction_Confidence: extractionConfidence,
    Consistency_Flag: consistencyFlag,
    Processing_Latency: processingLatency,
    Default_Probability: defaultProbability,
    Classification_Verdict: classificationVerdict,
    Top_Risk_Drivers: topRiskDrivers,
  };
}

function generateApplicants(count) {
  return Array.from({ length: count }, (_, i) => buildApplicant(i));
}

export const useLoanStore = create((set) => ({
  currentTab: "ledger",
  activeApplicantId: null,
  applicants: generateApplicants(20),

  setCurrentTab: (tab) => set({ currentTab: tab }),

  selectApplicant: (applicantId) =>
    set({ activeApplicantId: applicantId, currentTab: "insights" }),
}));
