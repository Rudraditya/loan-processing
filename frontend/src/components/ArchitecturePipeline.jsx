import { useEffect, useMemo, useRef, useState } from "react";
import ReactFlow, { Background, Controls, Handle, Position } from "reactflow";
import "reactflow/dist/style.css";
import {
  Cpu,
  FileSpreadsheet,
  FileText,
  FileUp,
  Gauge,
  Loader2,
  Play,
  ScanText,
  SlidersHorizontal,
  Users,
  Wand2,
  Zap,
} from "lucide-react";
import PageHeader from "./PageHeader";

const EDGE_COLOR = "#22d3ee"; // bright cyan, high-contrast against zinc-950

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
    </div>
  );
}

const nodeTypes = { pipeline: PipelineNode };

// Static illustration of app.py's own extraction -> scoring pipeline (not a
// per-applicant trace, and not the separate agentic_api extraction/scoring
// path - see CLAUDE.md's "four co-existing subsystems" for that distinction).
function buildGraph() {
  const col = (i) => i * 400;

  const nodes = [
    {
      id: "upload",
      type: "pipeline",
      position: { x: col(0), y: 300 },
      data: { icon: FileUp, title: "Document Upload", subtitle: "Salary slip (PDF/JPG) + bank statement (XLSX/PDF)" },
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
        subtitle: "Combines both documents into one record; cross-checks the computed balance against the statement's own header",
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
      data: { icon: SlidersHorizontal, title: "Feature Engineering", subtitle: "2 engineered ratios + one-hot encoding" },
    },
    {
      id: "imputation",
      type: "pipeline",
      position: { x: col(4), y: 300 },
      data: { icon: Wand2, title: "Data Imputation", subtitle: "Fields unavailable from documents filled with training-set medians" },
    },
    {
      id: "model",
      type: "pipeline",
      position: { x: col(4), y: 580 },
      data: { icon: Zap, title: "Logistic Regression Inference", subtitle: "predict_proba() on scaled, imputed features" },
    },
    {
      id: "riskOutput",
      type: "pipeline",
      position: { x: col(5), y: 140 },
      data: { icon: Gauge, title: "Risk Meter Output", subtitle: "Default probability + Approve / Decline verdict" },
    },
    {
      id: "clustering",
      type: "pipeline",
      position: { x: col(5), y: 460 },
      data: { icon: Users, title: "Customer Segmentation", subtitle: "Employment type x income tier" },
    },
  ];

  // The wave number is this edge's topological depth in the pipeline - used
  // only to sequence the "Simulate Flow" animation, not real timing data.
  const edgeIds = [
    ["upload", "parsePdf", 0],
    ["upload", "parseExcel", 0],
    ["parsePdf", "extraction", 1],
    ["parseExcel", "extraction", 1],
    ["extraction", "mlLayer", 2],
    ["mlLayer", "feature", 3],
    ["mlLayer", "imputation", 3],
    ["mlLayer", "model", 3],
    ["feature", "riskOutput", 4],
    ["feature", "clustering", 4],
    ["imputation", "riskOutput", 4],
    ["imputation", "clustering", 4],
    ["model", "riskOutput", 4],
    ["model", "clustering", 4],
  ];

  const edges = edgeIds.map(([source, target, wave]) => ({
    id: `${source}-${target}`,
    source,
    target,
    type: "smoothstep",
    animated: true,
    style: { stroke: EDGE_COLOR, strokeWidth: 2.5 },
    data: { wave },
  }));

  return { nodes, edges };
}

const { nodes: PIPELINE_NODES, edges: PIPELINE_EDGES } = buildGraph();
const MAX_WAVE = Math.max(...PIPELINE_EDGES.map((e) => e.data.wave));
const WAVE_STEP_MS = 450;

export default function ArchitecturePipeline() {
  const [activeWave, setActiveWave] = useState(-1);
  const [simulating, setSimulating] = useState(false);
  const timeoutsRef = useRef([]);

  useEffect(() => () => timeoutsRef.current.forEach(clearTimeout), []);

  function runSimulation() {
    timeoutsRef.current.forEach(clearTimeout);
    timeoutsRef.current = [];
    setSimulating(true);
    for (let wave = 0; wave <= MAX_WAVE; wave++) {
      timeoutsRef.current.push(setTimeout(() => setActiveWave(wave), wave * WAVE_STEP_MS));
    }
    timeoutsRef.current.push(
      setTimeout(() => {
        setActiveWave(-1);
        setSimulating(false);
      }, (MAX_WAVE + 1) * WAVE_STEP_MS + 400)
    );
  }

  const displayEdges = useMemo(
    () =>
      PIPELINE_EDGES.map((edge) => {
        const isLit = edge.data.wave === activeWave;
        return {
          ...edge,
          style: {
            stroke: isLit ? "#67e8f9" : EDGE_COLOR,
            strokeWidth: isLit ? 4.5 : 2.5,
          },
        };
      }),
    [activeWave]
  );

  return (
    <div>
      <PageHeader
        eyebrow="Architecture Overview"
        title="Data Flow Pipeline"
        description="How a submitted application moves through extraction and scoring. Drag to pan, scroll to zoom, or use the controls to fit the full diagram."
        right={
          <button
            onClick={runSimulation}
            disabled={simulating}
            className="flex items-center gap-1.5 rounded-lg border border-zinc-800 bg-zinc-900/60 px-3 py-1.5 text-xs font-medium text-zinc-300 transition-colors hover:border-accent hover:text-accent disabled:cursor-not-allowed disabled:opacity-60"
          >
            {simulating ? (
              <Loader2 size={12} strokeWidth={2} className="animate-spin" />
            ) : (
              <Play size={12} strokeWidth={2} className="fill-current" />
            )}
            {simulating ? "Simulating..." : "Simulate Flow"}
          </button>
        }
      />

      <div className="h-[640px] w-full overflow-hidden rounded-xl border border-zinc-800/80 bg-zinc-950/60">
        <ReactFlow
          nodes={PIPELINE_NODES}
          edges={displayEdges}
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
