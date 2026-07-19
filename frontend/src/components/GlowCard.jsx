import { useRef } from "react";
import { motion, useMotionValue, useSpring, useReducedMotion } from "motion/react";

const MAGNETIC_STRENGTH = 10;

export default function GlowCard({ children, className = "", index = 0, style, ...props }) {
  const ref = useRef(null);
  const reduce = useReducedMotion();

  const x = useMotionValue(0);
  const y = useMotionValue(0);
  const springX = useSpring(x, { stiffness: 200, damping: 20, mass: 0.4 });
  const springY = useSpring(y, { stiffness: 200, damping: 20, mass: 0.4 });

  function handlePointerMove(e) {
    const el = ref.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const relX = e.clientX - rect.left;
    const relY = e.clientY - rect.top;
    el.style.setProperty("--x", `${relX}px`);
    el.style.setProperty("--y", `${relY}px`);
    if (reduce) return;
    x.set((relX / rect.width - 0.5) * MAGNETIC_STRENGTH);
    y.set((relY / rect.height - 0.5) * MAGNETIC_STRENGTH);
  }

  function handlePointerLeave() {
    x.set(0);
    y.set(0);
  }

  return (
    <motion.div
      ref={ref}
      onMouseMove={handlePointerMove}
      onMouseLeave={handlePointerLeave}
      initial={reduce ? false : { opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.55, delay: index * 0.09, ease: [0.16, 1, 0.3, 1] }}
      whileHover={reduce ? undefined : { y: -3 }}
      style={{ x: springX, y: springY, ...style }}
      className={
        "group relative overflow-hidden rounded-xl border border-zinc-800/80 bg-zinc-900/30 " +
        "transition-colors duration-300 hover:border-accent/40 " +
        className
      }
      {...props}
    >
      <div
        className="pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-300 group-hover:opacity-100"
        style={{
          background:
            "radial-gradient(420px circle at var(--x, 50%) var(--y, 50%), rgba(59,130,246,0.12), transparent 65%)",
        }}
      />
      <div className="relative">{children}</div>
    </motion.div>
  );
}
