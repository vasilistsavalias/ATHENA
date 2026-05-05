"use client";

import { ReactNode } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";

import { WIZARD_STEPS, WizardStep, isStepUnlocked, stepIndex } from "@/lib/flow";
import { useI18n } from "@/lib/i18n/context";

type WizardShellProps = {
  step: WizardStep;
  title: string;
  description?: string;
  participantId?: string;
  progressLabel?: string;
  progressValue?: number;
  children: ReactNode;
  footer?: ReactNode;
  backDisabled?: boolean;
};

function stepLabel(
  id: WizardStep,
  dictionary: ReturnType<typeof useI18n>["dictionary"],
): string {
  const key =
    id === "profile"
      ? "profile"
      : id === "block-a-feedback"
        ? "blockAFeedback"
        : id === "block-b-feedback"
          ? "blockBFeedback"
          : id === "block-c-feedback"
            ? "blockCFeedback"
          : id === "block-c"
            ? "blockC"
            : id === "block-a"
              ? "blockA"
              : id === "block-b"
              ? "blockB"
              : (id as "welcome" | "consent" | "complete");
  return dictionary.steps[key];
}

export function WizardShell({
  step,
  title,
  description,
  participantId,
  progressLabel,
  progressValue,
  children,
  footer,
  backDisabled,
}: WizardShellProps) {
  const router = useRouter();
  const { locale, setLocale, dictionary } = useI18n();
  const currentIndex = stepIndex(step);
  const totalSteps = WIZARD_STEPS.length;

  return (
    <div className="flex overflow-hidden" style={{ height: "100dvh", flexDirection: "column" }}>
      {/* ── Top rail ── */}
      <header
        className="flex shrink-0 items-center justify-between border-b px-4 py-2.5 backdrop-blur-xl sm:px-6"
        style={{ borderColor: "rgba(255,255,255,0.07)", background: "rgba(10,15,31,0.78)" }}
      >
        {/* Brand + back button + participant */}
        <div className="flex items-center gap-2.5">
          {/* Back button — hidden on first step, disabled during submissions */}
          {currentIndex > 0 && (
            <motion.button
              type="button"
              onClick={() => {
                if (currentIndex > 0) {
                  const prevStep = WIZARD_STEPS[currentIndex - 1];
                  router.push(prevStep.route);
                }
              }}
              disabled={backDisabled}
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              className="flex h-7 w-7 items-center justify-center rounded-lg transition-colors disabled:cursor-not-allowed disabled:opacity-30"
              style={{ color: "rgba(199,210,254,0.45)" }}
              title="Back"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M19 12H5M12 19l-7-7 7-7" />
              </svg>
            </motion.button>
          )}
          <span className="athena-chip rounded-full px-3 py-1 text-[10px] font-bold tracking-widest">
            ATHENA
          </span>
          {participantId && (
            <span
              className="hidden rounded-full border px-2.5 py-0.5 text-[10px] font-medium sm:inline-block"
              style={{
                borderColor: "rgba(165,180,252,0.18)",
                background: "rgba(165,180,252,0.06)",
                color: "rgba(199,210,254,0.5)",
              }}
            >
              {participantId}
            </span>
          )}
        </div>

        {/* Step dots */}
        <div className="flex items-center gap-1.5">
          {WIZARD_STEPS.map((entry, i) => {
            const isActive = entry.id === step;
            const isPast = i < currentIndex;
            const isUnlocked = isStepUnlocked(step, entry.id);
            return (
              <motion.div
                key={entry.id}
                title={stepLabel(entry.id, dictionary)}
                initial={false}
                animate={{
                  width: isActive ? 22 : 6,
                  backgroundColor: isActive
                    ? "rgba(103,214,255,0.92)"
                    : isPast
                      ? "rgba(103,214,255,0.38)"
                      : isUnlocked
                        ? "rgba(255,255,255,0.18)"
                        : "rgba(255,255,255,0.08)",
                }}
                transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
                style={{ height: 6, borderRadius: 99 }}
              />
            );
          })}
        </div>

        {/* Step counter + lang toggle */}
        <div className="flex items-center gap-2.5">
          <span className="hidden text-[10px] font-medium sm:block" style={{ color: "rgba(199,210,254,0.38)" }}>
            {currentIndex + 1}/{totalSteps} · {stepLabel(step, dictionary)}
          </span>
          <div
            className="flex rounded-xl p-0.5"
            style={{ border: "1px solid rgba(255,255,255,0.1)", background: "rgba(255,255,255,0.04)" }}
          >
            {(["en", "el"] as const).map((lang) => (
              <button
                key={lang}
                type="button"
                onClick={() => setLocale(lang)}
                className={`rounded-lg px-2.5 py-1 text-[10px] font-bold uppercase tracking-wide transition-all duration-200 ${
                  locale === lang
                    ? "bg-gradient-to-r from-cyan-300 to-indigo-400 text-slate-950 shadow-sm"
                    : "text-indigo-100/45 hover:text-indigo-100/80"
                }`}
              >
                {lang}
              </button>
            ))}
          </div>
        </div>
      </header>

      {/* ── Progress bar ── */}
      {typeof progressValue === "number" && (
        <div className="relative h-0.5 w-full shrink-0" style={{ background: "rgba(255,255,255,0.05)" }}>
          <motion.div
            className="absolute inset-y-0 left-0"
            style={{ background: "linear-gradient(90deg, #67d6ff, #818cf8)" }}
            initial={false}
            animate={{ width: `${Math.max(0, Math.min(100, progressValue))}%` }}
            transition={{ duration: 0.65, ease: [0.22, 1, 0.36, 1] }}
          />
        </div>
      )}

      {/* ── Scrollable content ── */}
      <main className="flex min-h-0 flex-1 flex-col overflow-y-auto">
        <div className="mx-auto w-full max-w-5xl px-5 py-8 sm:px-8">
          {/* Heading */}
          <motion.div
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.48, ease: [0.22, 1, 0.36, 1] }}
            className="mb-7 space-y-1.5"
          >
            <h1 className="text-3xl font-bold tracking-tight sm:text-4xl">{title}</h1>
            {description && (
              <p className="text-sm" style={{ color: "rgba(179,190,223,0.65)" }}>
                {description}
              </p>
            )}
            {progressLabel && (
              <p className="text-[11px] font-medium" style={{ color: "rgba(179,190,223,0.38)" }}>
                {progressLabel}
              </p>
            )}
          </motion.div>

          {/* Page content */}
          <AnimatePresence mode="wait" initial={false}>
            <motion.div
              key={step}
              initial={{ opacity: 0, x: 48 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -32, scale: 0.99 }}
              transition={{ duration: 0.42, ease: [0.22, 1, 0.36, 1] }}
              className="space-y-5"
            >
              {children}
            </motion.div>
          </AnimatePresence>
        </div>
      </main>

      {/* ── Footer ── */}
      {footer && (
        <div
          className="shrink-0 border-t px-6 py-4 backdrop-blur-xl"
          style={{ borderColor: "rgba(255,255,255,0.07)", background: "rgba(10,15,31,0.82)" }}
        >
          <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-end gap-2">
            {footer}
          </div>
        </div>
      )}
    </div>
  );
}
