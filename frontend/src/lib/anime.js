import { useEffect, useRef } from "react";
import { animate, createScope, stagger, splitText, utils } from "animejs";

export function prefersReducedMotion() {
  return typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

// Splits a heading's text into characters and animates them in on mount -
// the kinetic-typography treatment animejs.com uses for its own page
// titles (see splitText() in their docs). Re-runs whenever `deps` change so
// a heading that swaps content (e.g. CustomerInsights switching applicant)
// replays the reveal instead of splitting stale text.
export function useRevealText(deps = []) {
  const ref = useRef(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    if (prefersReducedMotion()) {
      utils.set(el, { opacity: 1 });
      return;
    }

    const split = splitText(el, { chars: true });
    const anim = animate(split.chars, {
      opacity: [0, 1],
      y: [18, 0],
      rotateZ: [() => utils.random(-6, 6), 0],
      delay: stagger(16),
      duration: 650,
      ease: "outExpo",
    });

    return () => {
      anim.pause();
      if (typeof split.revert === "function") split.revert();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return ref;
}

// Fades/slides in every direct child carrying data-reveal, staggered - a
// drop-in replacement for the ad-hoc Framer `listVariants`/`rowVariants`
// pattern repeated across pages, scoped to `root` so it never touches
// elements outside this container (e.g. the Sidebar) and cleanly reverts on
// unmount/deps change via createScope().
export function useStaggerReveal(deps = []) {
  const root = useRef(null);
  const scope = useRef(null);

  useEffect(() => {
    if (!root.current) return;

    scope.current = createScope({ root }).add(() => {
      const targets = root.current.querySelectorAll("[data-reveal]");
      if (!targets.length) return;
      if (prefersReducedMotion()) {
        utils.set(targets, { opacity: 1 });
        return;
      }
      utils.set(targets, { opacity: 0, y: 14 });
      animate(targets, {
        opacity: [0, 1],
        y: [14, 0],
        delay: stagger(45),
        duration: 500,
        ease: "outQuart",
      });
    });

    return () => scope.current?.revert();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return root;
}

// Animates a numeric text node from 0 -> value (e.g. stat tiles, meters).
// Mirrors the pattern CustomerInsights.jsx's RiskGauge used to do with
// Framer's useMotionValue, just driven by anime.js instead.
export function useCountUp(value, { decimals = 0, duration = 1200, prefix = "", suffix = "" } = {}) {
  const ref = useRef(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const safeValue = Number.isFinite(value) ? value : 0;

    if (prefersReducedMotion()) {
      el.textContent = `${prefix}${safeValue.toFixed(decimals)}${suffix}`;
      return;
    }

    const counter = { v: 0 };
    const anim = animate(counter, {
      v: safeValue,
      duration,
      ease: "outExpo",
      onUpdate: () => {
        el.textContent = `${prefix}${counter.v.toFixed(decimals)}${suffix}`;
      },
    });

    return () => anim.pause();
  }, [value, decimals, duration, prefix, suffix]);

  return ref;
}
