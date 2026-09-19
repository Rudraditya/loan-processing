import { useEffect, useRef } from "react";
import { animate } from "animejs";
import { prefersReducedMotion } from "../lib/anime";

// Decorative animated ring motif echoing animejs.com's hero circle (dashed
// orbit + glowing gradient arc, slowly counter-rotating). Purely
// aria-hidden decoration - never carries information a screen reader needs.
export default function OrbitRing({ size = 420, className = "" }) {
  const outerRef = useRef(null);
  const innerRef = useRef(null);
  const dotsRef = useRef(null);

  useEffect(() => {
    if (prefersReducedMotion()) return;
    const anims = [
      animate(outerRef.current, { rotate: "1turn", duration: 42000, loop: true, ease: "linear" }),
      animate(innerRef.current, { rotate: "-1turn", duration: 28000, loop: true, ease: "linear" }),
      animate(dotsRef.current, { opacity: [0.25, 0.9, 0.25], duration: 2800, loop: true, ease: "inOutSine" }),
    ];
    return () => anims.forEach((a) => a.pause());
  }, []);

  const r1 = size / 2 - 6;
  const r2 = size / 2 - 34;
  const c = size / 2;

  return (
    <svg
      viewBox={`0 0 ${size} ${size}`}
      width={size}
      height={size}
      className={className}
      style={{ overflow: "visible" }}
      aria-hidden="true"
    >
      <defs>
        <linearGradient id="orbit-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#3b82f6" />
          <stop offset="50%" stopColor="#8b5cf6" />
          <stop offset="100%" stopColor="#f472b6" />
        </linearGradient>
      </defs>

      <circle cx={c} cy={c} r={r1 + 14} fill="none" stroke="#27272a" strokeWidth="1" opacity="0.4" />

      <g ref={outerRef} style={{ transformOrigin: "50% 50%" }}>
        <circle
          cx={c}
          cy={c}
          r={r1}
          fill="none"
          stroke="url(#orbit-gradient)"
          strokeWidth="1.5"
          strokeDasharray="1 7"
          opacity="0.7"
        />
      </g>

      <g ref={innerRef} style={{ transformOrigin: "50% 50%" }}>
        <circle
          cx={c}
          cy={c}
          r={r2}
          fill="none"
          stroke="url(#orbit-gradient)"
          strokeWidth="2"
          strokeDasharray={`${r2 * 0.85} ${2 * Math.PI * r2}`}
          strokeLinecap="round"
          opacity="0.5"
        />
      </g>

      <g ref={dotsRef}>
        <circle cx={c} cy={c - r1} r="3" fill="#8b5cf6" />
        <circle cx={c + r1} cy={c} r="2.5" fill="#3b82f6" />
        <circle cx={c} cy={c + r1} r="2" fill="#f472b6" />
        <circle cx={c - r1} cy={c} r="2.5" fill="#8b5cf6" />
      </g>
    </svg>
  );
}
