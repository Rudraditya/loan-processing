import { useMemo } from "react";
import ReactFlow, { Background, Controls, Handle, Position } from "reactflow";
import "reactflow/dist/style.css";
import {
  Cpu,
  FileSpreadsheet,
  FileText,
  FileUp,
  Gauge,
  MousePointerClick,
  ScanText,
  SlidersHorizontal,
  Users,
  Wand2,
  Zap,
} from "lucide-react";
import { useLoanStore } from "../store/useLoanStore";
import StatusPill from "./StatusPill";

const EDGE_COLOR = "#22d3ee"; // bright cyan, high-contrast against zinc-950

function segmentFor(applicant) {
  const highIncome = applicant.Monthly_Net_Income > 150000;
  if (applicant.Employment_Type === "Self-Employed") {
    return highIncome ? "High-Income Self-Employed" : "Emerging Self-Employed";
  }
  return highIncome ? "Premium Salaried" : "Mass-Market Salaried";
}

function PipelineNode({ data }) {
  const Icon = data.icon;
  return (
    <div className="w-80 rounded-xl border border-zinc-700 bg-zinc-900/95 p-6 shadow-xl shadow-black/40 backdrop-blur-sm">
      <Handle type="target" position={Position.Left} className="!h-3 !w-3 !border-zinc-600 !bg-cyan-400" />
      <Handle type="source" position={Position.Right} className="!h-3 !w-3 !border-zinc-600 !bg-cyan-400" />
      <div className="flex items-center gap-2.5 text-white">
        <Icon size={22} strokeWidth={2} className="shrink-0 text-accent" />
        <span className="text-xl font-bold leading-tight">{data.title}</span>
      </div>
      {data.subtitle && <div className="mt-2 text-base leading-snug text-gray-300">{data.subtitle}</div>}
      {data.body}
    </div>
  );
}

const nodeTypes = { pipeline: PipelineNode };

function buildGraph(applicant) {
  const col = (i) => i * 400;

  const stat = (label, value) => (
    <div className="mt-3 flex items-baseline justify-between">
      <span className="text-xs font-medium uppercase tracking-wide text-gray-400">{label}</span>
      <span className="font-mono text-xl font-bold tabular-nums text-white">{value}</span>
    </div>
  );

  const nodes = [
    {
      id: "upload",
      type: "pipeline",
      position: { x: col(0), y: 300 },
      data: { icon: FileUp, title: "Document Upload", subtitle: "Salary slip + bank statement" },
    },
    {
      id: "parsePdf",
      type: "pipeline",
      position: { x: col(1), y: 60 },
      data: { icon: FileText, title: "Parse PDF (Regex)", subtitle: "pdfplumber + label anchors" },
    },
    {
      id: "parseExcel",
      type: "pipeline",
      position: { x: col(1), y: 560 },
      data: { icon: FileSpreadsheet, title: "Parse Excel (Pandas)", subtitle: "day-weighted balance calc" },
    },
    {
      id: "extraction",
      type: "pipeline",
      position: { x: col(2), y: 300 },
      data: {
        icon: ScanText,
        title: "Extraction Layer",
        body: (
          <>
            {stat("Confidence", `${applicant.Extraction_Confidence}%`)}
            {stat("Latency", `${applicant.Processing_Latency}s`)}
          </>
        ),
      },
    },
    {
      id: "mlLayer",
      type: "pipeline",
      position: { x: col(3), y: 300 },
      data: { icon: Cpu, title: "ML Financial Layer", subtitle: "11-feature vector assembled" },
    },
    {
      id: "feature",
      type: "pipeline",
      position: { x: col(4), y: 20 },
      data: { icon: SlidersHorizontal, title: "Feature Engineering", subtitle: "2 engineered ratios + one-hot" },
    },
    {
      id: "imputation",
      type: "pipeline",
      position: { x: col(4), y: 300 },
      data: { icon: Wand2, title: "Data Imputation", subtitle: "NaN-tolerant (native to XGBoost)" },
    },
    {
      id: "xgboost",
      type: "pipeline",
      position: { x: col(4), y: 580 },
      data: { icon: Zap, title: "XGBoost Inference", subtitle: "predict_proba()" },
    },
    {
      id: "riskOutput",
      type: "pipeline",
      position: { x: col(5), y: 140 },
      data: {
        icon: Gauge,
        title: "Risk Meter Output",
        body: (
          <div className="mt-3 flex items-center gap-3">
            <span className="font-mono text-2xl font-bold tabular-nums text-white">
              {(applicant.Default_Probability * 100).toFixed(1)}%
            </span>
            <StatusPill verdict={applicant.Classification_Verdict} />
          </div>
        ),
      },
    },
    {
      id: "clustering",
      type: "pipeline",
      position: { x: col(5), y: 460 },
      data: {
        icon: Users,
        title: "Customer Clustering",
        body: <div className="mt-3 text-base font-medium text-gray-200">{segmentFor(applicant)}</div>,
      },
    },
  ];

  const edgeIds = [
    ["upload", "parsePdf"],
    ["upload", "parseExcel"],
    ["parsePdf", "extraction"],
    ["parseExcel", "extraction"],
    ["extraction", "mlLayer"],
    ["mlLayer", "feature"],
    ["mlLayer", "imputation"],
    ["mlLayer", "xgboost"],
    ["feature", "riskOutput"],
    ["feature", "clustering"],
    ["imputation", "riskOutput"],
    ["imputation", "clustering"],
    ["xgboost", "riskOutput"],
    ["xgboost", "clustering"],
  ];

  const edges = edgeIds.map(([source, target]) => ({
    id: `${source}-${target}`,
    source,
    target,
    type: "smoothstep",
    animated: true,
    style: { stroke: EDGE_COLOR, strokeWidth: 2.5 },
  }));

  return { nodes, edges };
}

export default function ArchitecturePipeline() {
  const activeApplicantId = useLoanStore((state) => state.activeApplicantId);
  const applicants = useLoanStore((state) => state.applicants);
  const setCurrentTab = useLoanStore((state) => state.setCurrentTab);

  const applicant = applicants.find((a) => a.Applicant_ID === activeApplicantId) ?? null;

  const { nodes, edges } = useMemo(() => (applicant ? buildGraph(applicant) : { nodes: [], edges: [] }), [
    applicant,
  ]);

  if (!applicant) {
    return (
      <div className="flex flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-zinc-800 py-32 text-center">
        <MousePointerClick size={22} strokeWidth={1.5} className="text-zinc-600" />
        <p className="text-sm text-zinc-500">Select an applicant from the Ledger to trace their pipeline run.</p>
        <button
          onClick={() => setCurrentTab("ledger")}
          className="mt-1 rounded-md border border-zinc-800 px-3.5 py-1.5 text-[13px] text-zinc-300 transition-colors hover:border-accent hover:text-accent"
        >
          Go to Ledger
        </button>
      </div>
    );
  }

  return (
    <div>
      <div className="mb-4 flex items-baseline justify-between px-1">
        <h1 className="text-base font-semibold text-zinc-100">Data Flow Pipeline</h1>
        <span className="font-mono text-xs tabular-nums text-zinc-500">{applicant.Applicant_ID}</span>
      </div>

      <p className="mb-3 px-1 text-xs text-zinc-500">
        Drag to pan, scroll to zoom, or use the controls to fit the full diagram.
      </p>

      <div className="h-[640px] w-full overflow-hidden rounded-xl border border-zinc-800/80 bg-zinc-950/60">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          fitView
          fitViewOptions={{ padding: 0.12 }}
          minZoom={0.4}
          maxZoom={1.75}
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable={false}
          panOnDrag
          zoomOnScroll
          zoomOnPinch
          proOptions={{ hideAttribution: false }}
        >
          <Background color="#3f3f46" gap={26} size={1} />
          <Controls showInteractive={false} />
        </ReactFlow>
      </div>
    </div>
  );
}
