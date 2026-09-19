import { Bell } from "lucide-react";

export default function Header({ pendingReviews, onPendingReviewsClick }) {
  return (
    <header className="sticky top-0 z-40 flex h-16 shrink-0 items-center justify-end border-b border-zinc-800/80 bg-midnight-card/90 px-8 backdrop-blur-md">
      <div className="flex items-center gap-4">
        <button
          onClick={onPendingReviewsClick}
          disabled={pendingReviews === 0}
          className="flex items-center gap-2 rounded-full border border-zinc-800 bg-zinc-900/60 px-3 py-1.5 font-mono text-[11px] text-zinc-100 transition-colors enabled:hover:border-accent enabled:hover:text-accent disabled:cursor-default"
        >
          <span className="relative flex h-1.5 w-1.5">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
            <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-emerald-500" />
          </span>
          {pendingReviews > 0 ? `${pendingReviews} Pending Review${pendingReviews === 1 ? "" : "s"}` : "System Online"}
        </button>

        <button
          onClick={onPendingReviewsClick}
          disabled={pendingReviews === 0}
          className="relative rounded-lg border border-zinc-800/60 p-2 text-zinc-200 transition-colors enabled:hover:bg-zinc-900/60 enabled:hover:text-white disabled:cursor-default"
        >
          <Bell size={15} strokeWidth={2} />
          {pendingReviews > 0 && <span className="absolute right-1.5 top-1.5 h-1.5 w-1.5 rounded-full bg-accent" />}
        </button>
      </div>
    </header>
  );
}
