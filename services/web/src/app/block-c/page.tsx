"use client";
/* eslint-disable @next/next/no-img-element */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { AnimatePresence, motion } from "framer-motion";

import {
  ApiError,
  BlockCItem,
  getBlockCNext,
  getProgress,
  getSession,
  ProgressInfo,
  resolveAssetUrl,
  SessionInfo,
  submitBlockC,
} from "@/lib/api";
import { useI18n } from "@/lib/i18n/context";
import { formatI18n } from "@/lib/i18n/format";
import { formatEta, getAverageResponseMs, recordResponseTime } from "@/lib/progress";
import { WizardShell } from "@/components/wizard-shell";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";

const BLOCK_C_TIME_KEY = "arch_eval_block_c_times";
type BlockCChoice = "A" | "B" | "C" | "D";

function ConfidenceButton({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <motion.button
      type="button"
      whileHover={{ scale: 1.08, y: -1 }}
      whileTap={{ scale: 0.92 }}
      className={`min-h-11 rounded-xl border px-4 py-2 text-sm font-bold transition-colors ${
        active
          ? "border-cyan-300 bg-cyan-300 text-slate-950"
          : "border-white/15 bg-white/5 text-indigo-100/75 hover:border-cyan-300/50"
      }`}
      onClick={onClick}
    >
      {label}
    </motion.button>
  );
}

function routeAfterBlockC(progress: ProgressInfo): "/block-c-feedback" | "/complete" {
  if (!progress.block_c_feedback_completed) {
    return "/block-c-feedback";
  }
  return "/complete";
}

export default function BlockCPage() {
  const router = useRouter();
  const { dictionary, locale } = useI18n();
  const startedAtRef = useRef<number>(Date.now());

  const [session, setSession] = useState<SessionInfo | null>(null);
  const [progress, setProgress] = useState<ProgressInfo | null>(null);
  const [item, setItem] = useState<BlockCItem | null>(null);
  const [choice, setChoice] = useState<BlockCChoice | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [comment, setComment] = useState("");
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [itemKey, setItemKey] = useState(0);

  const progressPercent = useMemo(() => {
    if (!progress || progress.block_c_total === 0) return 0;
    return (progress.block_c_completed / progress.block_c_total) * 100;
  }, [progress]);

  const remaining = progress ? Math.max(0, progress.block_c_total - progress.block_c_completed) : 0;
  const avgMs = getAverageResponseMs(BLOCK_C_TIME_KEY);
  const eta = formatEta((remaining * avgMs) / 1000);

  const resetForm = () => {
    setChoice(null);
    setConfidence(null);
    setComment("");
    startedAtRef.current = Date.now();
  };

  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [sessionPayload, progressPayload] = await Promise.all([getSession(), getProgress()]);
      setSession(sessionPayload);
      setProgress(progressPayload);
      if (progressPayload.is_complete) {
        router.replace("/complete");
        return;
      }
      if (!progressPayload.profile_completed) {
        router.replace("/profile");
        return;
      }
      if (progressPayload.block_a_total > 0 && progressPayload.block_a_completed < progressPayload.block_a_total) {
        router.replace("/block-a");
        return;
      }
      if (progressPayload.block_a_total > 0 && !progressPayload.block_a_feedback_completed) {
        router.replace("/block-a-feedback");
        return;
      }
      if (progressPayload.block_b_total > 0 && progressPayload.block_b_completed < progressPayload.block_b_total) {
        router.replace("/block-b");
        return;
      }
      if (progressPayload.block_b_total > 0 && !progressPayload.block_b_feedback_completed) {
        router.replace("/block-b-feedback");
        return;
      }
      if (progressPayload.block_c_total <= 0) {
        router.replace("/complete");
        return;
      }
      if (progressPayload.block_c_completed >= progressPayload.block_c_total) {
        router.replace(routeAfterBlockC(progressPayload));
        return;
      }

      const next = await getBlockCNext();
      if (next.done || !next.item) {
        router.replace("/block-c-feedback");
        return;
      }
      setItem(next.item);
      setItemKey((value) => value + 1);
      resetForm();
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        router.replace("/");
        return;
      }
      setError(err instanceof Error ? err.message : "Failed to load Part 3.");
    } finally {
      setLoading(false);
    }
  }, [router]);

  useEffect(() => {
    void loadData();
  }, [loadData]);

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && ["INPUT", "TEXTAREA"].includes(target.tagName)) return;
      const key = event.key.toLowerCase();
      if (["a", "b", "c", "d"].includes(key)) {
        setChoice(key.toUpperCase() as BlockCChoice);
      }
      if (/^[1-5]$/.test(key)) {
        setConfidence(Number(key));
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  const handleSubmit = async () => {
    if (!item || !choice || confidence === null) {
      setError(dictionary.blockC.chooseError);
      return;
    }
    const trimmedComment = comment.trim();
    if (!trimmedComment) {
      setError(dictionary.blockC.commentRequiredError);
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      const responseTime = Date.now() - startedAtRef.current;
      const nextPayload = await submitBlockC({
        assignment_id: item.assignment_id,
        choice,
        confidence,
        comment: trimmedComment,
        response_time_ms: responseTime,
      });
      recordResponseTime(BLOCK_C_TIME_KEY, responseTime);
      const updatedProgress = await getProgress();
      setProgress(updatedProgress);
      if (nextPayload.done || !nextPayload.item) {
        router.replace("/block-c-feedback");
        return;
      }
      setItem(nextPayload.item);
      setItemKey((value) => value + 1);
      resetForm();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to submit Part 3 response.");
    } finally {
      setSubmitting(false);
    }
  };

  const candidates: Array<{ value: BlockCChoice; label: string; url: string }> = item
    ? [
        { value: "A", label: dictionary.blockC.candidateA, url: item.option_a_url },
        { value: "B", label: dictionary.blockC.candidateB, url: item.option_b_url },
        { value: "C", label: dictionary.blockC.candidateC, url: item.option_c_url },
        { value: "D", label: dictionary.blockC.candidateD, url: item.option_d_url },
      ]
    : [];

  const blockCDescription =
    locale === "el"
      ? "Θα δείτε τέσσερις αποκαταστάσεις του ίδιου αντικειμένου που έχουν παραχθεί από μοντέλα. Και οι τέσσερις εικόνες είναι συνθετικές και καμία δεν είναι πραγματική. Στόχος σας είναι να επιλέξετε την καλύτερη αποκατάσταση, δηλαδή αυτή που φαίνεται πιο αρχαιολογικά πειστική και πιο κατάλληλη να χρησιμοποιηθεί ως κύρια έξοδος μοντέλου."
      : "You will see four model-generated restorations of the same object. All four images are synthetic; none is real. Your task is to choose the best restoration, that is, the one that appears most archaeologically plausible and most suitable to be used as the primary model output.";
  const blockCQuestion =
    locale === "el"
      ? "Ποια παραγόμενη εικόνα είναι η καλύτερη συνολικά;"
      : "Which generated image is the best overall candidate?";
  const blockCAllFakeNotice =
    locale === "el"
      ? "Και οι τέσσερις εικόνες είναι παραγόμενες από μοντέλα. Δεν υπάρχει πραγματική εικόνα σε αυτό το μέρος."
      : "All four images in this part are model-generated. There is no real image in this section.";
  const blockCPrimaryModelNotice =
    locale === "el"
      ? "Η επιλογή σας θα μας δείξει ποιο μοντέλο πρέπει να χρησιμοποιείται κυρίως στις επόμενες αποκαταστάσεις."
      : "Your choice will help determine which model should be used primarily in future restorations.";

  return (
    <WizardShell
      step="block-c"
      title={dictionary.blockC.title}
      description={blockCDescription}
      participantId={session?.participant_id}
      progressLabel={
        progress
          ? `${progress.block_c_completed}/${progress.block_c_total} ${dictionary.blockC.progressLabel} · ${formatI18n(dictionary.common.etaPrefix, {
              value: eta,
            })}`
          : undefined
      }
      progressValue={progressPercent}
      footer={
        <Button onClick={handleSubmit} disabled={loading || submitting || !choice || confidence === null || !comment.trim()}>
          {submitting ? dictionary.blockC.submitting : dictionary.blockC.submit}
        </Button>
      }
    >
      {loading ? <p className="text-sm text-indigo-100/70">{dictionary.blockC.loading}</p> : null}
      {error ? <p className="text-sm text-red-500">{error}</p> : null}
      {!loading && item ? (
        <AnimatePresence mode="wait" initial={false}>
          <motion.div
            key={itemKey}
            initial={{ opacity: 0, x: 42 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -30 }}
            transition={{ duration: 0.32, ease: [0.22, 1, 0.36, 1] }}
            className="space-y-5"
          >
            <div className="space-y-2">
              <p className="text-xs text-indigo-100/55">
                {dictionary.blockC.itemPrefix} {item.item_order} / {item.total_items}
              </p>
              <p className="text-sm font-medium text-indigo-50">{blockCQuestion}</p>
              <p className="rounded-lg border border-cyan-300/20 bg-cyan-300/5 px-3 py-2 text-xs text-indigo-100/75">
                {blockCAllFakeNotice}
              </p>
              <p className="rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-xs text-indigo-100/65">
                {blockCPrimaryModelNotice}
              </p>
            </div>

            <div className="grid grid-cols-2 gap-3">
              {candidates.map((candidate) => {
                const selected = choice === candidate.value;
                return (
                  <motion.button
                    key={candidate.value}
                    type="button"
                    aria-pressed={selected}
                    onClick={() => setChoice(candidate.value)}
                    onMouseUp={(event) => event.currentTarget.blur()}
                    whileHover={{ scale: 1.015 }}
                    whileTap={{ scale: 0.985 }}
                    className="relative cursor-pointer overflow-hidden rounded-2xl border text-left outline-none transition-all duration-300 focus:outline-none focus-visible:outline-none focus-visible:ring-0"
                    style={{
                      borderColor: selected ? "rgba(103,214,255,0.75)" : "rgba(255,255,255,0.08)",
                      background: selected ? "rgba(103,214,255,0.06)" : "rgba(10,15,31,0.5)",
                      boxShadow: selected
                        ? "0 0 0 2px rgba(103,214,255,0.35), 0 8px 32px -8px rgba(103,214,255,0.25)"
                        : "none",
                      touchAction: "manipulation",
                    }}
                  >
                    <img
                      src={resolveAssetUrl(candidate.url)}
                      alt={candidate.label}
                      draggable={false}
                      className="pointer-events-none h-[34vh] w-full select-none object-contain"
                    />
                    <div
                      className="pointer-events-none absolute right-2 top-2 flex h-7 w-7 items-center justify-center rounded-full transition-all duration-150"
                      style={{
                        background: "rgba(103,214,255,0.9)",
                        opacity: selected ? 1 : 0,
                        transform: selected ? "scale(1)" : "scale(0.7)",
                      }}
                    >
                      <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                        <path d="M2 7L6 11L12 3" stroke="#0a0f1f" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                    </div>
                    <div
                      className="pointer-events-none px-3 py-2 text-center text-sm font-bold"
                      style={{ color: selected ? "rgba(103,214,255,0.9)" : "rgba(220,228,255,0.82)" }}
                    >
                      {candidate.value} · {candidate.label}
                    </div>
                  </motion.button>
                );
              })}
            </div>

            <div className="space-y-2">
              <Label>{dictionary.blockC.confidence}</Label>
              <div className="flex gap-2">
                {[1, 2, 3, 4, 5].map((value) => (
                  <ConfidenceButton key={value} label={`${value}`} active={confidence === value} onClick={() => setConfidence(value)} />
                ))}
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="block-c-comment">{dictionary.blockC.commentLabel}</Label>
              <Textarea
                id="block-c-comment"
                value={comment}
                onChange={(event) => setComment(event.target.value)}
                placeholder={dictionary.blockC.commentPlaceholder}
                maxLength={2000}
                required
              />
              <p className="text-xs text-indigo-100/45">{dictionary.blockC.commentHint}</p>
              <p className="text-xs text-indigo-100/45">
                {dictionary.blockC.commentExample1}
                <br />
                {dictionary.blockC.commentExample2}
              </p>
            </div>
          </motion.div>
        </AnimatePresence>
      ) : null}
    </WizardShell>
  );
}
