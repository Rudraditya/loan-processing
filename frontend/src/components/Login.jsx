import { useEffect, useRef, useState } from "react";
import { animate, stagger, utils } from "animejs";
import { Eye, EyeOff, Lock, ShieldCheck, User } from "lucide-react";
import { prefersReducedMotion, useRevealText } from "../lib/anime";
import OrbitRing from "./OrbitRing";

// Client-side-only placeholder gate - the check happens in the bundled JS, so
// it's trivially visible/bypassable via devtools. Fine as a "for now" stopgap
// per explicit instruction, not a substitute for real auth.
const VALID_USERNAME = "admin";
const VALID_PASSWORD = "admin@123";

export default function Login({ onLogin }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const headlineRef = useRevealText([]);
  const cardRef = useRef(null);

  useEffect(() => {
    if (!cardRef.current) return;
    if (prefersReducedMotion()) {
      utils.set(cardRef.current, { opacity: 1 });
      return;
    }
    animate(cardRef.current, {
      opacity: [0, 1],
      y: [24, 0],
      duration: 700,
      ease: "outExpo",
    });
    animate(cardRef.current.querySelectorAll("[data-field]"), {
      opacity: [0, 1],
      x: [-12, 0],
      delay: stagger(70, { start: 250 }),
      duration: 500,
      ease: "outQuart",
    });
  }, []);

  function handleSubmit(e) {
    e.preventDefault();
    if (!username || !password) {
      setError("Please fill in all fields");
      return;
    }
    setError("");
    setIsLoading(true);

    setTimeout(() => {
      setIsLoading(false);
      if (username === VALID_USERNAME && password === VALID_PASSWORD) {
        onLogin(username);
      } else {
        setError("Invalid username or password");
      }
    }, 500);
  }

  return (
    <div className="relative flex min-h-[100dvh] items-center overflow-hidden bg-midnight text-zinc-100">
      <div className="pointer-events-none absolute left-[8%] top-1/2 hidden -translate-y-1/2 lg:block">
        <OrbitRing size={520} />
      </div>
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_top,rgba(59,130,246,0.08),transparent_55%)]" />

      <div className="relative z-10 mx-auto grid w-full max-w-[1400px] grid-cols-1 items-center gap-16 px-8 py-16 lg:grid-cols-2">
        {/* Left: hero headline, animejs.com-style */}
        <div className="hidden lg:block">
          <span className="mb-5 inline-flex items-center gap-2 rounded-full border border-zinc-800 bg-zinc-900/60 px-3 py-1.5 font-mono text-[11px] text-zinc-300">
            <span className="relative flex h-1.5 w-1.5">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
              <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-emerald-500" />
            </span>
            portal.status: online
          </span>
          <h1
            ref={headlineRef}
            className="font-display text-6xl font-extrabold leading-[0.98] tracking-tight text-white xl:text-7xl"
          >
            All-in-one
            <br />
            loan assessment
            <br />
            engine.
          </h1>
          <p className="mt-6 max-w-md text-base leading-relaxed text-zinc-400">
            A fast and explainable credit-risk console — document extraction, model scoring, and
            human-in-the-loop review in one place.
          </p>
        </div>

        {/* Right: sign-in card */}
        <div className="mx-auto w-full max-w-[420px] lg:mx-0 lg:ml-auto">
          <div className="relative z-10 mb-8 flex items-center gap-4 lg:hidden">
            <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-accent/15 text-2xl font-bold text-accent">
              L
            </div>
            <span className="font-display text-3xl font-extrabold tracking-tight text-white">
              Loan Assessment Agent
            </span>
          </div>

          <div
            ref={cardRef}
            className="relative z-10 w-full rounded-2xl border border-zinc-800/80 bg-zinc-900/40 p-9 opacity-0 shadow-[0_20px_50px_rgba(0,0,0,0.5)] backdrop-blur-xl"
          >
            <div className="mb-7">
              <h2 className="font-display text-2xl font-bold tracking-tight text-white">Sign In</h2>
              <p className="mt-1.5 text-sm text-zinc-300">Access the loan assessment dashboard</p>
            </div>

            <form onSubmit={handleSubmit} className="space-y-5">
              {error && (
                <div
                  data-field
                  className="rounded-lg border border-red-800/50 bg-red-950/40 px-3.5 py-2.5 text-xs text-red-400"
                >
                  {error}
                </div>
              )}

              <div data-field className="space-y-1.5">
                <label className="flex items-center gap-1.5 text-xs font-medium uppercase tracking-wide text-zinc-200">
                  <User size={13} strokeWidth={2} />
                  Username
                </label>
                <input
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  placeholder="admin"
                  autoComplete="username"
                  className="w-full rounded-lg border border-zinc-800 bg-zinc-950/60 px-3.5 py-2.5 text-sm text-zinc-100 outline-none transition-colors placeholder:text-zinc-600 focus:border-accent"
                />
              </div>

              <div data-field className="space-y-1.5">
                <label className="flex items-center gap-1.5 text-xs font-medium uppercase tracking-wide text-zinc-200">
                  <Lock size={13} strokeWidth={2} />
                  Password
                </label>
                <div className="relative">
                  <input
                    type={showPassword ? "text" : "password"}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="••••••••"
                    autoComplete="current-password"
                    className="w-full rounded-lg border border-zinc-800 bg-zinc-950/60 px-3.5 py-2.5 pr-11 text-sm text-zinc-100 outline-none transition-colors placeholder:text-zinc-600 focus:border-accent"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword((v) => !v)}
                    className="absolute right-3.5 top-1/2 -translate-y-1/2 text-zinc-300 transition-colors hover:text-white"
                  >
                    {showPassword ? <EyeOff size={15} strokeWidth={2} /> : <Eye size={15} strokeWidth={2} />}
                  </button>
                </div>
              </div>

              <button
                data-field
                type="submit"
                disabled={isLoading}
                className="flex w-full items-center justify-center gap-2 rounded-lg bg-accent px-4 py-3 text-sm font-semibold text-white transition-colors hover:bg-accent/90 disabled:cursor-not-allowed disabled:opacity-70"
              >
                {isLoading ? (
                  <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                ) : (
                  <>
                    <ShieldCheck size={16} strokeWidth={2} />
                    Sign In
                  </>
                )}
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}
