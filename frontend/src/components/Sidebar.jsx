import { ClipboardCheck, Compass, FolderClosed, Gauge, History, LayoutGrid, LogOut, Plus, Workflow } from "lucide-react";

const NAV_ITEMS = [
  { id: "dashboard", label: "Dashboard", icon: LayoutGrid },
  { id: "ledger", label: "Ledger", icon: FolderClosed },
  { id: "pending-reviews", label: "Pending Reviews", icon: ClipboardCheck, badge: "pendingReviews" },
  { id: "decisions", label: "Decision History", icon: History },
  { id: "insights", label: "Insights", icon: Gauge },
  { id: "pipeline", label: "Pipeline", icon: Workflow },
];

export default function Sidebar({ currentTab, setCurrentTab, username, onLogout, pendingReviews = 0 }) {
  return (
    <aside className="flex h-[100dvh] w-64 shrink-0 flex-col border-r border-zinc-800/80 bg-midnight-card px-4 py-6">
      <div className="mb-8 flex items-center gap-2.5 px-2">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent/15 text-accent">
          <Compass size={16} strokeWidth={2.5} />
        </div>
        <span className="font-display text-sm font-semibold tracking-tight text-white">Loan Assessment Agent</span>
      </div>

      <button
        onClick={() => setCurrentTab("upload")}
        className={
          "mb-6 flex items-center justify-center gap-2 rounded-xl px-4 py-3 text-sm font-semibold text-white shadow-[0_4px_15px_rgba(59,130,246,0.25)] transition-all active:scale-[0.98] " +
          (currentTab === "upload"
            ? "bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500"
            : "bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 hover:opacity-95")
        }
      >
        <Plus size={16} strokeWidth={2.5} />
        New Application
      </button>

      <nav className="flex-1 space-y-1">
        {NAV_ITEMS.map((item) => {
          const isActive = currentTab === item.id;
          const Icon = item.icon;
          const badgeCount = item.badge === "pendingReviews" ? pendingReviews : 0;
          return (
            <button
              key={item.id}
              onClick={() => setCurrentTab(item.id)}
              className={
                "group relative flex w-full items-center gap-3 rounded-lg py-2.5 pl-4 pr-3 text-sm transition-colors " +
                (isActive ? "bg-zinc-800/60 text-white" : "text-zinc-200 hover:bg-zinc-800/30 hover:text-white")
              }
            >
              <span
                className={
                  "absolute left-0 top-1/2 h-5 w-[3px] -translate-y-1/2 rounded-full transition-colors " +
                  (isActive ? "bg-accent" : "bg-transparent")
                }
              />
              <Icon size={16} strokeWidth={2} className={isActive ? "text-accent" : "text-zinc-400"} />
              <span className="flex-1 text-left">{item.label}</span>
              {badgeCount > 0 && (
                <span className="rounded-full bg-accent/20 px-1.5 py-0.5 font-mono text-[10px] font-semibold tabular-nums text-accent">
                  {badgeCount}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      <div className="border-t border-zinc-800/60 pt-4">
        <div className="flex items-center gap-2.5 rounded-lg border border-zinc-800/60 bg-zinc-900/40 px-3 py-2.5">
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-accent/15 text-xs font-bold uppercase text-accent">
            {username?.[0] ?? "?"}
          </div>
          <div className="min-w-0 flex-1">
            <div className="truncate text-xs font-medium text-white">{username}</div>
            <div className="text-[10px] text-zinc-300">Signed in</div>
          </div>
          <button
            onClick={onLogout}
            className="shrink-0 text-zinc-300 transition-colors hover:text-red-400"
            title="Sign out"
          >
            <LogOut size={15} strokeWidth={2} />
          </button>
        </div>
      </div>
    </aside>
  );
}
