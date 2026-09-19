import { useRevealText } from "../lib/anime";

// Shared page-header treatment (eyebrow tag + kinetic-text title +
// description) used at the top of every page - the big-bold-headline
// typographic rhythm borrowed from animejs.com's own site, standing in for
// the small `<h1 className="text-base font-semibold">` each page used to
// hand-roll individually.
export default function PageHeader({ eyebrow, title, description, right, className = "" }) {
  const titleRef = useRevealText([title]);

  return (
    <div className={"mb-8 flex flex-col gap-4 px-1 sm:flex-row sm:items-end sm:justify-between " + className}>
      <div className="min-w-0">
        {eyebrow && (
          <span className="mb-2 inline-block font-mono text-[11px] font-medium uppercase tracking-[0.2em] text-accent/80">
            {eyebrow}
          </span>
        )}
        <h1
          ref={titleRef}
          className="font-display text-3xl font-bold leading-[1.05] tracking-tight text-white sm:text-4xl"
        >
          {title}
        </h1>
        {description && <p className="mt-2.5 max-w-xl text-sm leading-relaxed text-zinc-400">{description}</p>}
      </div>
      {right && <div className="shrink-0">{right}</div>}
    </div>
  );
}
