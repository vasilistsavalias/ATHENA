"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence, type Variants } from "framer-motion";

import { ApiError, CampaignInfo, getActiveCampaign, getProgress, getSession, invite, ProgressInfo } from "@/lib/api";
import { nextRouteFromProgress } from "@/lib/flow";
import { prewarmBackend } from "@/lib/prewarm";
import { useI18n } from "@/lib/i18n/context";

/* ─── animation variants ─── */
const sectionVariants: Variants = {
  hidden: { opacity: 0, y: 40 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.7, ease: [0.22, 1, 0.36, 1] as [number, number, number, number] } },
};

const stagger: Variants = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.12 } },
};

const item: Variants = {
  hidden: { opacity: 0, y: 32 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.65, ease: [0.22, 1, 0.36, 1] as [number, number, number, number] } },
};

function ExampleImagePreview({
  heightClass,
  src,
}: {
  heightClass: string;
  src: string;
}) {
  return (
    <div
      className={`relative overflow-hidden rounded-lg border ${heightClass}`}
      style={{
        borderColor: "rgba(255,255,255,0.08)",
        background: "linear-gradient(180deg, rgba(12, 18, 36, 0.9), rgba(9, 14, 28, 0.95))",
      }}
    >
      <img src={src} alt="" className="h-full w-full object-contain" />
      <div
        className="absolute inset-x-0 bottom-0 h-12"
        style={{ background: "linear-gradient(to top, rgba(7, 11, 24, 0.55), transparent)" }}
      />
    </div>
  );
}

/* ─── Fake Block A example card ─── */
function BlockAExampleCard({ dict }: { dict: ReturnType<typeof useI18n>["dictionary"] }) {
  const scales = [
    { label: dict.landing.exampleAuthenticity, value: 4 },
    { label: dict.landing.examplePlausibility, value: 3 },
    { label: dict.landing.exampleConfidence, value: 5 },
  ];
  return (
    <div className="overflow-hidden rounded-2xl border" style={{ borderColor: "rgba(141,159,234,0.22)", background: "linear-gradient(160deg, rgba(21,31,58,0.82), rgba(17,25,47,0.85))" }}>
      <div className="p-3 pb-0">
        <ExampleImagePreview heightClass="h-36 sm:h-44" src="/landing/block-a/image.png" />
      </div>
      {/* Fake scales */}
      <div className="space-y-3 p-4">
        {scales.map((s) => (
          <div key={s.label}>
            <p className="mb-1.5 text-[11px] font-medium" style={{ color: "rgba(179,190,223,0.6)" }}>{s.label}</p>
            <div className="flex gap-1.5">
              {[1, 2, 3, 4, 5].map((n) => (
                <div
                  key={n}
                  className="flex h-8 w-8 items-center justify-center rounded-lg text-xs font-bold"
                  style={
                    n === s.value
                      ? { background: "linear-gradient(135deg, #67d6ff, #818cf8)", color: "#0b1020", boxShadow: "0 2px 12px -3px rgba(103,214,255,0.5)" }
                      : { border: "1px solid rgba(255,255,255,0.12)", color: "rgba(179,190,223,0.5)" }
                  }
                >
                  {n}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ─── Fake Block B example card ─── */
function BlockBExampleCard({ dict }: { dict: ReturnType<typeof useI18n>["dictionary"] }) {
  return (
    <div className="overflow-hidden rounded-2xl border" style={{ borderColor: "rgba(141,159,234,0.22)", background: "linear-gradient(160deg, rgba(21,31,58,0.82), rgba(17,25,47,0.85))" }}>
      <p className="px-4 pt-4 text-center text-[11px] font-medium" style={{ color: "rgba(179,190,223,0.5)" }}>
        {dict.landing.exampleWhichBetter}
      </p>
      {/* Two panels side by side */}
      <div className="grid grid-cols-2 gap-3 p-4">
        {[dict.landing.exampleOptionA, dict.landing.exampleOptionB].map((label, i) => (
          <div
            key={label}
            className="flex flex-col items-center gap-2 rounded-xl border p-3 transition-colors"
            style={
              i === 0
                ? { borderColor: "rgba(103,214,255,0.55)", background: "rgba(103,214,255,0.08)", boxShadow: "0 0 20px -6px rgba(103,214,255,0.25)" }
                : { borderColor: "rgba(255,255,255,0.1)", background: "rgba(255,255,255,0.03)" }
            }
          >
            <ExampleImagePreview
              heightClass="h-20 w-full sm:h-28"
              src={i === 0 ? "/landing/block-b/option-a.png" : "/landing/block-b/option-b.png"}
            />
            <span className="text-[11px] font-semibold" style={{ color: i === 0 ? "rgba(103,214,255,0.9)" : "rgba(179,190,223,0.5)" }}>{label}</span>
            {i === 0 && (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#67d6ff" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M20 6 9 17l-5-5" />
              </svg>
            )}
          </div>
        ))}
      </div>
      {/* Confidence row */}
      <div className="border-t px-4 pb-4 pt-3" style={{ borderColor: "rgba(255,255,255,0.06)" }}>
        <p className="mb-1.5 text-[11px] font-medium" style={{ color: "rgba(179,190,223,0.5)" }}>{dict.landing.exampleConfidence}</p>
        <div className="flex gap-1.5">
          {[1, 2, 3, 4, 5].map((n) => (
            <div
              key={n}
              className="flex h-7 w-7 items-center justify-center rounded-lg text-[10px] font-bold"
              style={
                n === 4
                  ? { background: "linear-gradient(135deg, #67d6ff, #818cf8)", color: "#0b1020" }
                  : { border: "1px solid rgba(255,255,255,0.12)", color: "rgba(179,190,223,0.45)" }
              }
            >
              {n}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function BlockCExampleCard({ locale }: { locale: "el" | "en" }) {
  const labels = ["A", "B", "C", "D"];
  return (
    <div
      className="overflow-hidden rounded-2xl border"
      style={{ borderColor: "rgba(141,159,234,0.22)", background: "linear-gradient(160deg, rgba(21,31,58,0.82), rgba(17,25,47,0.85))" }}
    >
      <p className="px-4 pt-4 text-center text-[11px] font-medium" style={{ color: "rgba(179,190,223,0.5)" }}>
        {locale === "el" ? "Πιο κοντά στην πραγματικότητα;" : "Closest to reality?"}
      </p>
      <div className="grid grid-cols-2 gap-3 p-4">
        {labels.map((label, idx) => (
          <div
            key={label}
            className="flex flex-col items-center gap-2 rounded-xl border p-3"
            style={
              idx === 2
                ? { borderColor: "rgba(103,214,255,0.55)", background: "rgba(103,214,255,0.08)" }
                : { borderColor: "rgba(255,255,255,0.1)", background: "rgba(255,255,255,0.03)" }
            }
          >
            <ExampleImagePreview
              heightClass="h-16 w-full sm:h-20"
              src={[
                "/landing/block-c/option-a.png",
                "/landing/block-c/option-b.png",
                "/landing/block-c/option-c.png",
                "/landing/block-c/option-d.png",
              ][idx]}
            />
            <span className="text-[11px] font-semibold" style={{ color: idx === 2 ? "rgba(103,214,255,0.9)" : "rgba(179,190,223,0.5)" }}>
              {locale === "el" ? `Επιλογή ${label}` : `Option ${label}`}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ─── Checkmark icon ─── */
function CheckIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" className="shrink-0">
      <circle cx="12" cy="12" r="10" fill="rgba(103,214,255,0.15)" stroke="rgba(103,214,255,0.5)" strokeWidth="1.5" />
      <path d="M8 12.5l2.5 2.5 5-5" stroke="#67d6ff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

/* ════════════════════════════════════════════════════════════
   Main Landing Page
   ════════════════════════════════════════════════════════════ */
export default function LandingPage() {
  const router = useRouter();
  const { locale, setLocale, dictionary } = useI18n();
  const dataUseNotice = dictionary.landing.dataUseNotice ?? (
    locale === "el"
      ? "Οι βαθμολογίες, οι επιλογές, οι χρόνοι απάντησης και τα γραπτά σχόλια αποθηκεύονται ανώνυμα και χρησιμοποιούνται για στατιστική ανάλυση, γραφήματα και την τεκμηρίωση της διπλωματικής."
      : "All ratings, choices, timings, and written comments are stored in anonymised form and will be used for statistical analysis, figures, and reporting in the thesis."
  );

  /* invite form state */
  const [inviteCode, setInviteCode] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [shakeKey, setShakeKey] = useState(0);
  const [backendCold, setBackendCold] = useState(false);

  /* session resume state */
  const [hasSession, setHasSession] = useState(false);
  const [progress, setProgress] = useState<ProgressInfo | null>(null);
  const [activeCampaign, setActiveCampaign] = useState<CampaignInfo | null>(null);

  const inviteRef = useRef<HTMLInputElement>(null);

  /* ── Prewarm backend + check existing session on mount ── */
  useEffect(() => {
    prewarmBackend();

    async function bootstrap() {
      let loadedCampaign: CampaignInfo | null = null;
      try {
        loadedCampaign = await getActiveCampaign();
        setActiveCampaign(loadedCampaign);
      } catch {
        /* campaign may not be active yet */
      }
      try {
        const sessionPayload = await getSession();
        const p = await getProgress();
        if (loadedCampaign && sessionPayload.campaign.id !== loadedCampaign.id) {
          setHasSession(false);
          setProgress(null);
          return;
        }
        setProgress(p);
        setHasSession(true);
      } catch {
        /* not authenticated - fine */
      }
    }
    void bootstrap();
  }, []);

  const nextRoute = hasSession && progress ? nextRouteFromProgress(progress) : null;
  const hasBlockA = (progress ? Math.max(progress.block_a_total, activeCampaign?.block_a_target_count ?? 0) : (activeCampaign?.block_a_target_count ?? 0)) > 0;
  const hasBlockC = (progress ? Math.max(progress.block_c_total, activeCampaign?.block_c_target_count ?? 0) : (activeCampaign?.block_c_target_count ?? 0)) > 0;
  const whatIntro = hasBlockA
    ? hasBlockC
      ? locale === "el"
        ? "\u0397 \u03c3\u03c5\u03bd\u03b5\u03b4\u03c1\u03af\u03b1 \u03ad\u03c7\u03b5\u03b9 \u03c4\u03c1\u03af\u03b1 \u03c3\u03cd\u03bd\u03c4\u03bf\u03bc\u03b1 \u03bc\u03ad\u03c1\u03b7. \u0394\u03b5\u03af\u03c4\u03b5 \u03c3\u03c5\u03bd\u03bf\u03c0\u03c4\u03b9\u03ba\u03ac \u03c4\u03b9 \u03b8\u03b1 \u03c3\u03c5\u03bd\u03b1\u03bd\u03c4\u03ae\u03c3\u03b5\u03c4\u03b5:"
        : "The session has three parts. Here is what you will see:"
      : locale === "el"
        ? "\u0397 \u03c3\u03c5\u03bd\u03b5\u03b4\u03c1\u03af\u03b1 \u03ad\u03c7\u03b5\u03b9 \u03b4\u03cd\u03bf \u03c3\u03cd\u03bd\u03c4\u03bf\u03bc\u03b1 \u03bc\u03ad\u03c1\u03b7. \u0394\u03b5\u03af\u03c4\u03b5 \u03c3\u03c5\u03bd\u03bf\u03c0\u03c4\u03b9\u03ba\u03ac \u03c4\u03b9 \u03b8\u03b1 \u03c3\u03c5\u03bd\u03b1\u03bd\u03c4\u03ae\u03c3\u03b5\u03c4\u03b5:"
        : "The session has two parts. Here is what you will see:"
    : locale === "el"
      ? "\u0397 \u03c3\u03c5\u03bd\u03b5\u03b4\u03c1\u03af\u03b1 \u03b1\u03c0\u03bf\u03c4\u03b5\u03bb\u03b5\u03af\u03c4\u03b1\u03b9 \u03b1\u03c0\u03cc \u03bc\u03af\u03b1 \u03c3\u03cd\u03bd\u03c4\u03bf\u03bc\u03b7 \u03c3\u03c5\u03b3\u03ba\u03c1\u03b9\u03c4\u03b9\u03ba\u03ae \u03b5\u03bd\u03cc\u03c4\u03b7\u03c4\u03b1. \u0394\u03b5\u03af\u03c4\u03b5 \u03c3\u03c5\u03bd\u03bf\u03c0\u03c4\u03b9\u03ba\u03ac \u03c4\u03b9 \u03b8\u03b1 \u03c3\u03c5\u03bd\u03b1\u03bd\u03c4\u03ae\u03c3\u03b5\u03c4\u03b5:"
      : "The session consists of one short comparison task. Here is what you will see:";
  const blockBTitle = hasBlockA
    ? dictionary.landing.whatBlockBTitle
    : locale === "el"
      ? "Συγκριτική αξιολόγηση"
      : "Comparison task";
  const blockCTitle = locale === "el" ? "Μέρος 3 - Σύγκριση Πολλαπλών Μοντέλων" : "Part 3 - Multi-model";
  const blockCDesc = locale === "el"
    ? "Θα δείτε 4 υποψήφιες εικόνες από διαφορετικά μοντέλα. Επιλέξτε ποια είναι πιο κοντά στην αρχαιολογική πραγματικότητα."
    : "You will see 4 candidate images from different models. Select the one closest to archaeological reality.";
  const blockCExample = locale === "el" ? "ΠΑΡΑΔΕΙΓΜΑ ΟΘΟΝΗΣ:" : "SCREEN EXAMPLE:";
  const profileNotice = hasBlockA
    ? dictionary.landing.profileStepNotice
    : locale === "el"
      ? "Πριν ξεκινήσει η σύγκριση, θα δείτε ένα σύντομο βήμα προφίλ σχετικά με την ειδίκευσή σας."
      : "Before the comparison task starts, there is one short background step about your specialization.";
  const humanInLoopNotice =
    locale === "el"
      ? "Το ATHENA είναι υποστηρικτικό εργαλείο (human-in-the-loop): βοηθά τον αρχαιολόγο στην αποκατάσταση και δεν αντικαθιστά τον ειδικό."
      : "ATHENA is an assistive human-in-the-loop tool: it supports archaeologists in restoration work and does not replace experts.";
  const realWorldNotice =
    locale === "el"
      ? "Στην πραγματική αποκατάσταση δεν υπάρχει διαθέσιμο «αρχικό». Η βαθμονόμηση με κρυφές ταυτότητες χρησιμοποιείται μόνο για ερευνητική αξιολόγηση."
      : "In real restoration work, the original image is unavailable. Hidden-identity calibration is used only for research evaluation.";

  /* ── Handle invite submission ── */
  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    setBackendCold(false);

    /* Give the user a "waking up" toast if the request takes >5s */
    const coldTimer = setTimeout(() => setBackendCold(true), 5000);

    const isServerError = (err: unknown) =>
      !(err instanceof ApiError) || err.status >= 500;

    try {
      let payload;
      try {
        payload = await invite(inviteCode);
      } catch (firstErr) {
        /* Cold-start: if it's a 5xx / network error, wait and retry up to 2x */
        if (isServerError(firstErr)) {
          clearTimeout(coldTimer);
          setBackendCold(true);
          // Wait up to 60s total across two retries (20s + 40s)
          for (const waitMs of [20_000, 40_000]) {
            await new Promise((r) => setTimeout(r, waitMs));
            try {
              payload = await invite(inviteCode);
              break;
            } catch (retryErr) {
              if (!isServerError(retryErr)) throw retryErr;
              // still a server error — loop again or fall through
            }
          }
          if (!payload) throw firstErr;
        } else {
          throw firstErr;
        }
      }
      clearTimeout(coldTimer);
      if (!payload.progress.profile_completed) {
        router.replace("/consent");
      } else {
        router.replace(nextRouteFromProgress(payload.progress));
      }
    } catch (err) {
      clearTimeout(coldTimer);
      setBackendCold(false);
      const msg =
        err instanceof ApiError
          ? err.message || dictionary.landing.inviteInvalid
          : dictionary.landing.inviteInvalid;
      setError(msg);
      setShakeKey((k) => k + 1);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="relative min-h-dvh">
      {/* ── Animated background orbs (fixed behind everything) ── */}
      <div className="pointer-events-none fixed inset-0 z-0">
        <motion.div
          className="absolute -left-[20%] -top-[30%] h-[90vh] w-[90vh] rounded-full"
          style={{ background: "radial-gradient(circle, rgba(124,124,255,0.24) 0%, transparent 68%)" }}
          animate={{ scale: [1, 1.08, 1], x: [0, 22, 0], y: [0, -14, 0] }}
          transition={{ duration: 13, repeat: Infinity, ease: "easeInOut" }}
        />
        <motion.div
          className="absolute -right-[18%] -top-[20%] h-[75vh] w-[75vh] rounded-full"
          style={{ background: "radial-gradient(circle, rgba(103,214,255,0.18) 0%, transparent 68%)" }}
          animate={{ scale: [1, 1.06, 1], x: [0, -20, 0], y: [0, 14, 0] }}
          transition={{ duration: 16, repeat: Infinity, ease: "easeInOut", delay: 2.5 }}
        />
        <motion.div
          className="absolute bottom-[-25%] left-[25%] h-[65vh] w-[65vh] rounded-full"
          style={{ background: "radial-gradient(circle, rgba(40,79,173,0.22) 0%, transparent 68%)" }}
          animate={{ scale: [1, 1.1, 1], y: [0, -22, 0] }}
          transition={{ duration: 19, repeat: Infinity, ease: "easeInOut", delay: 1 }}
        />
      </div>

      {/* ── Language toggle — floating top right ── */}
      <div className="fixed right-4 top-4 z-50">
        <div
          className="flex rounded-xl p-0.5 backdrop-blur-xl"
          style={{ border: "1px solid rgba(255,255,255,0.12)", background: "rgba(10,15,31,0.65)" }}
        >
          {(["el", "en"] as const).map((lang) => (
            <button
              key={lang}
              type="button"
              onClick={() => setLocale(lang)}
              className={`rounded-lg px-3 py-1.5 text-[11px] font-bold uppercase tracking-wide transition-all duration-200 ${
                locale === lang
                  ? "bg-gradient-to-r from-cyan-300 to-indigo-400 text-slate-950 shadow-sm"
                  : "text-indigo-100/45 hover:text-indigo-100/80"
              }`}
            >
              {lang === "el" ? dictionary.landing.langEl : dictionary.landing.langEn}
            </button>
          ))}
        </div>
      </div>

      {/* ══════════════════════════════════════════
          SECTION 1 — HERO (full viewport)
         ══════════════════════════════════════════ */}
      <section className="relative z-10 flex h-dvh flex-col items-center justify-center px-6 text-center">
        <motion.div
          variants={stagger}
          initial="hidden"
          animate="visible"
          className="flex max-w-xl flex-col items-center gap-6"
        >
          <motion.div variants={item}>
            <h1
              className="text-7xl font-bold tracking-[0.14em] sm:text-8xl"
              style={{
                background: "linear-gradient(135deg, #eaf0ff 0%, #67d6ff 45%, #a78bfa 100%)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
                backgroundClip: "text",
              }}
            >
              ATHENA
            </h1>
            <p className="mt-2 text-[11px] font-medium tracking-[0.22em] uppercase sm:text-xs" style={{ color: "rgba(179,190,223,0.5)" }}>
              {dictionary.landing.acronym}
            </p>
          </motion.div>

          <motion.p
            variants={item}
            className="max-w-md text-base leading-relaxed sm:text-lg"
            style={{ color: "rgba(179,190,223,0.65)" }}
          >
            {dictionary.landing.heroSubtitle}
          </motion.p>

          {/* Resume button — only shown for returning users */}
          {hasSession && nextRoute && (
            <motion.div variants={item}>
              <motion.button
                onClick={() => router.push(nextRoute)}
                whileHover={{ scale: 1.04, y: -2 }}
                whileTap={{ scale: 0.96 }}
                className="rounded-2xl px-10 py-4 text-sm font-bold tracking-wide text-slate-950 transition-all duration-300"
                style={{
                  background: "linear-gradient(135deg, #4ade80, #22d3ee)",
                  boxShadow: "0 8px 36px -8px rgba(74,222,128,0.5), 0 2px 14px -4px rgba(34,211,238,0.4)",
                }}
              >
                {dictionary.landing.inviteResume}
              </motion.button>
            </motion.div>
          )}
        </motion.div>

        {/* Scroll indicator */}
        {!hasSession && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1.5 }}
            className="absolute bottom-8"
          >
            <motion.div
              animate={{ y: [0, 8, 0] }}
              transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
              className="flex flex-col items-center gap-1"
            >
              <div className="h-8 w-[1px]" style={{ background: "linear-gradient(to bottom, transparent, rgba(103,214,255,0.4))" }} />
              <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                <path d="M2 4l4 4 4-4" stroke="rgba(103,214,255,0.5)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </motion.div>
          </motion.div>
        )}
      </section>

      {/* ══════════════════════════════════════════
          SECTION 2 — ABOUT
         ══════════════════════════════════════════ */}
      <motion.section
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, margin: "-80px" }}
        variants={sectionVariants}
        className="relative z-10 mx-auto max-w-2xl px-6 py-20"
      >
        <h2 className="mb-5 text-2xl font-bold tracking-tight sm:text-3xl">{dictionary.landing.aboutHeading}</h2>
        <p className="text-base leading-[1.85] sm:text-lg" style={{ color: "rgba(199,210,254,0.72)" }}>
          {dictionary.landing.aboutBody}
        </p>
      </motion.section>

      {/* ══════════════════════════════════════════
          SECTION 3 — WHAT YOU'LL DO (with examples)
         ══════════════════════════════════════════ */}
      <motion.section
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, margin: "-80px" }}
        variants={stagger}
        className="relative z-10 mx-auto max-w-4xl px-6 pb-20"
      >
        <motion.h2 variants={item} className="mb-3 text-2xl font-bold tracking-tight sm:text-3xl">
          {dictionary.landing.whatHeading}
        </motion.h2>
        <motion.p variants={item} className="mb-10 text-base" style={{ color: "rgba(199,210,254,0.6)" }}>
          {whatIntro}
        </motion.p>
        <div className={`grid gap-10 ${hasBlockC ? "lg:grid-cols-3" : "lg:grid-cols-2"}`}>
          {hasBlockA ? (
            <motion.div variants={item} className="space-y-4">
              <div className="flex items-center gap-3">
                <span
                  className="inline-flex h-8 w-8 items-center justify-center rounded-full text-xs font-black"
                  style={{ background: "linear-gradient(135deg, #67d6ff, #818cf8)", color: "#0b1020" }}
                >
                  1
                </span>
                <h3 className="text-lg font-bold">{dictionary.landing.whatBlockATitle}</h3>
              </div>
              <p className="text-sm leading-relaxed" style={{ color: "rgba(199,210,254,0.65)" }}>
                {dictionary.landing.whatBlockADesc}
              </p>
              <p className="text-[11px] font-semibold uppercase tracking-wider" style={{ color: "rgba(103,214,255,0.5)" }}>
                {dictionary.landing.whatBlockAExample}
              </p>
              <BlockAExampleCard dict={dictionary} />
            </motion.div>
          ) : null}

          <motion.div variants={item} className="space-y-4">
            <div className="flex items-center gap-3">
              <span
                className="inline-flex h-8 w-8 items-center justify-center rounded-full text-xs font-black"
                style={{ background: "linear-gradient(135deg, #67d6ff, #818cf8)", color: "#0b1020" }}
              >
                {hasBlockA ? 2 : 1}
              </span>
              <h3 className="text-lg font-bold">{blockBTitle}</h3>
            </div>
            <p className="text-sm leading-relaxed" style={{ color: "rgba(199,210,254,0.65)" }}>
              {dictionary.landing.whatBlockBDesc}
            </p>
            <p className="text-[11px] font-semibold uppercase tracking-wider" style={{ color: "rgba(103,214,255,0.5)" }}>
              {dictionary.landing.whatBlockBExample}
            </p>
            <BlockBExampleCard dict={dictionary} />
          </motion.div>

          {hasBlockC ? (
            <motion.div variants={item} className="space-y-4">
              <div className="flex items-center gap-3">
                <span
                  className="inline-flex h-8 w-8 items-center justify-center rounded-full text-xs font-black"
                  style={{ background: "linear-gradient(135deg, #67d6ff, #818cf8)", color: "#0b1020" }}
                >
                  {hasBlockA ? 3 : 2}
                </span>
                <h3 className="text-lg font-bold">{blockCTitle}</h3>
              </div>
              <p className="text-sm leading-relaxed" style={{ color: "rgba(199,210,254,0.65)" }}>
                {blockCDesc}
              </p>
              <p className="text-[11px] font-semibold uppercase tracking-wider" style={{ color: "rgba(103,214,255,0.5)" }}>
                {blockCExample}
              </p>
              <BlockCExampleCard locale={locale} />
            </motion.div>
          ) : null}
        </div>

        {/* Time estimate */}
        <motion.div
          variants={item}
          className="mt-10 flex justify-center"
        >
          <span
            className="inline-block rounded-full px-6 py-2.5 text-sm font-semibold"
            style={{ border: "1px solid rgba(103,214,255,0.25)", background: "rgba(103,214,255,0.06)", color: "rgba(103,214,255,0.85)" }}
          >
            {dictionary.landing.timeEstimate}
          </span>
        </motion.div>
      </motion.section>

      {/* ══════════════════════════════════════════
          SECTION 4 — SAFETY / TRUST
         ══════════════════════════════════════════ */}
      <motion.section
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, margin: "-80px" }}
        variants={stagger}
        className="relative z-10 mx-auto max-w-2xl px-6 pb-20"
      >
        <motion.h2 variants={item} className="mb-6 text-2xl font-bold tracking-tight sm:text-3xl">
          {dictionary.landing.safetyHeading}
        </motion.h2>
        <div className="space-y-4">
          {dictionary.landing.safetyBullets.map((bullet) => (
            <motion.div key={bullet} variants={item} className="flex items-start gap-3">
              <CheckIcon />
              <p className="text-sm leading-relaxed sm:text-base" style={{ color: "rgba(199,210,254,0.72)" }}>
                {bullet}
              </p>
            </motion.div>
          ))}
          <motion.p variants={item} className="text-sm sm:text-base" style={{ color: "rgba(103,214,255,0.86)" }}>
            {dictionary.landing.anonymityNotice}
          </motion.p>
          <motion.p variants={item} className="text-sm sm:text-base" style={{ color: "rgba(199,210,254,0.68)" }}>
            {profileNotice}
          </motion.p>
          <motion.p variants={item} className="text-sm sm:text-base" style={{ color: "rgba(199,210,254,0.72)" }}>
            {humanInLoopNotice}
          </motion.p>
          <motion.p variants={item} className="text-sm sm:text-base" style={{ color: "rgba(199,210,254,0.72)" }}>
            {realWorldNotice}
          </motion.p>
          <motion.p variants={item} className="text-sm sm:text-base" style={{ color: "rgba(199,210,254,0.68)" }}>
            {dataUseNotice}
          </motion.p>
        </div>
      </motion.section>

      {/* ══════════════════════════════════════════
          SECTION 5 — INVITE CODE FORM
         ══════════════════════════════════════════ */}
      <motion.section
        id="invite-section"
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, margin: "-60px" }}
        variants={stagger}
        className="relative z-10 mx-auto flex max-w-sm flex-col items-center gap-6 px-6 pb-32 pt-4"
      >
        <motion.h2 variants={item} className="text-center text-xl font-bold tracking-tight sm:text-2xl">
          {dictionary.landing.inviteHeading}
        </motion.h2>

        <motion.form variants={item} onSubmit={handleSubmit} className="w-full space-y-3">
          <motion.div
            key={shakeKey}
            animate={shakeKey > 0 ? { x: [-11, 11, -8, 8, -5, 5, 0] } : {}}
            transition={{ duration: 0.48, ease: "easeOut" }}
          >
            <input
              ref={inviteRef}
              type="text"
              value={inviteCode}
              onChange={(e) => setInviteCode(e.target.value)}
              placeholder={dictionary.landing.invitePlaceholder}
              required
              autoComplete="off"
              spellCheck={false}
              className="w-full rounded-2xl px-5 py-4 text-center text-lg font-semibold tracking-[0.14em] text-white outline-none transition-all duration-300 placeholder:tracking-normal placeholder:text-sm"
              style={{
                border: "1px solid rgba(255,255,255,0.1)",
                background: "rgba(255,255,255,0.05)",
              }}
              onFocus={(e) => {
                e.currentTarget.style.border = "1px solid rgba(103,214,255,0.55)";
                e.currentTarget.style.background = "rgba(255,255,255,0.08)";
                e.currentTarget.style.boxShadow = "0 0 0 3px rgba(103,214,255,0.14), 0 0 28px rgba(103,214,255,0.1)";
              }}
              onBlur={(e) => {
                e.currentTarget.style.border = "1px solid rgba(255,255,255,0.1)";
                e.currentTarget.style.background = "rgba(255,255,255,0.05)";
                e.currentTarget.style.boxShadow = "none";
              }}
            />
          </motion.div>

          <AnimatePresence>
            {error && (
              <motion.p
                initial={{ opacity: 0, y: -6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="text-center text-xs"
                style={{ color: "rgba(248,113,113,0.85)" }}
              >
                {error}
              </motion.p>
            )}
          </AnimatePresence>

          <AnimatePresence>
            {backendCold && (
              <motion.p
                initial={{ opacity: 0, y: -4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="text-center text-xs"
                style={{ color: "rgba(103,214,255,0.7)" }}
              >
                {dictionary.common.backendWaking}
              </motion.p>
            )}
          </AnimatePresence>

          <motion.button
            type="submit"
            disabled={loading || !inviteCode.trim()}
            whileHover={{ scale: 1.03, y: -2 }}
            whileTap={{ scale: 0.97 }}
            className="w-full rounded-2xl py-4 text-sm font-bold tracking-wide text-slate-950 transition-opacity duration-300 disabled:cursor-not-allowed disabled:opacity-40"
            style={{
              background: "linear-gradient(135deg, #67d6ff 0%, #818cf8 100%)",
              boxShadow: "0 8px 36px -8px rgba(103,214,255,0.5), 0 2px 14px -4px rgba(124,124,255,0.4)",
            }}
          >
            {loading ? "···" : dictionary.landing.inviteSubmit}
          </motion.button>
        </motion.form>

        <motion.p
          variants={item}
          className="text-center text-xs"
          style={{ color: "rgba(179,190,223,0.35)" }}
        >
          {dictionary.landing.inviteResumeHint}
        </motion.p>
      </motion.section>
    </div>
  );
}
