"use client";
/* eslint-disable @next/next/no-img-element */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { AnimatePresence, motion } from "framer-motion";

import {
  ApiError,
  BlockBItem,
  getBlockBNext,
  getProgress,
  getSession,
  ProgressInfo,
  resolveAssetUrl,
  SessionInfo,
  submitBlockB,
} from "@/lib/api";
import { mapBlockBChoiceKey } from "@/lib/flow";
import { useI18n } from "@/lib/i18n/context";
import { formatI18n } from "@/lib/i18n/format";
import { formatEta, getAverageResponseMs, recordResponseTime } from "@/lib/progress";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { WizardShell } from "@/components/wizard-shell";

const BLOCK_B_TIME_KEY = "arch_eval_block_b_times";

type PairChoice = "A" | "B" | "Tie" | "Unsure";

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
      whileHover={{ scale: 1.1, y: -1 }}
      whileTap={{ scale: 0.9 }}
      animate={active ? { scale: [1, 1.18, 1], transition: { duration: 0.25, ease: [0.22, 1, 0.36, 1] } } : {}}
      className={`min-h-11 rounded-xl border px-4 py-2 text-sm font-bold transition-colors ${
        active
          ? "border-cyan-300 bg-cyan-300 text-slate-950"
          : "border-white/15 bg-white/5 text-indigo-100/75 hover:border-cyan-300/50"
      }`}
      style={active ? { boxShadow: "0 4px 18px -4px rgba(103,214,255,0.5)" } : {}}
      onClick={onClick}
    >
      {label}
    </motion.button>
  );
}

function routeAfterBlockB(progress: ProgressInfo): "/block-b-feedback" | "/block-c" | "/block-c-feedback" | "/complete" {
  if (!progress.block_b_feedback_completed) {
    return "/block-b-feedback";
  }
  if (progress.block_c_total > 0 && progress.block_c_completed < progress.block_c_total) {
    return "/block-c";
  }
  if (progress.block_c_total > 0 && !progress.block_c_feedback_completed) {
    return "/block-c-feedback";
  }
  return "/complete";
}

export default function BlockBPage() {
  const router = useRouter();
  const { dictionary, locale } = useI18n();
  const startedAtRef = useRef<number>(Date.now());

  const [session, setSession] = useState<SessionInfo | null>(null);
  const [progress, setProgress] = useState<ProgressInfo | null>(null);
  const [item, setItem] = useState<BlockBItem | null>(null);
  const [choice, setChoice] = useState<PairChoice | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [comment, setComment] = useState("");
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [itemKey, setItemKey] = useState(0);

  const progressPercent = useMemo(() => {
    if (!progress || progress.block_b_total === 0) {
      return 0;
    }
    return (progress.block_b_completed / progress.block_b_total) * 100;
  }, [progress]);

  const remaining = progress ? Math.max(0, progress.block_b_total - progress.block_b_completed) : 0;
  const avgMs = getAverageResponseMs(BLOCK_B_TIME_KEY);
  const eta = formatEta((remaining * avgMs) / 1000);

  function resetForm() {
    setChoice(null);
    setConfidence(null);
    setComment("");
    startedAtRef.current = Date.now();
  }

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
      if (progressPayload.block_a_total > 0 && !progressPayload.block_a_feedback_completed) {
        router.replace("/block-a-feedback");
        return;
      }
      if (progressPayload.block_a_total > 0 && progressPayload.block_a_completed < progressPayload.block_a_total) {
        router.replace("/block-a");
        return;
      }
      if (progressPayload.block_b_completed >= progressPayload.block_b_total) {
        router.replace(routeAfterBlockB(progressPayload));
        return;
      }
      const next = await getBlockBNext();
      if (next.done || !next.item) {
        router.replace("/block-b-feedback");
        return;
      }
      setItem(next.item);
      setItemKey((k) => k + 1);
      resetForm();
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        router.replace("/");
        return;
      }
      if (err instanceof ApiError && err.status === 409) {
        router.replace("/block-a");
        return;
      }
      setError(err instanceof Error ? err.message : "Failed to load Block B.");
    } finally {
      setLoading(false);
    }
  }, [router]);

  useEffect(() => {
    void loadData();
  }, [loadData]);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null;
      if (target && ["INPUT", "TEXTAREA"].includes(target.tagName)) {
        return;
      }
      const mapped = mapBlockBChoiceKey(event.key);
      if (mapped) {
        setChoice(mapped);
      } else if (/^[1-5]$/.test(event.key)) {
        setConfidence(Number(event.key));
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  async function handleSubmit() {
    if (!item || !choice || confidence === null) {
      setError(dictionary.blockB.chooseError);
      return;
    }
    const trimmedComment = comment.trim();
    if (!trimmedComment) {
      setError(dictionary.blockB.commentRequiredError);
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      const responseTime = Date.now() - startedAtRef.current;
      const nextPayload = await submitBlockB({
        assignment_id: item.assignment_id,
        choice,
        confidence,
        comment: trimmedComment,
        response_time_ms: responseTime,
      });
      recordResponseTime(BLOCK_B_TIME_KEY, responseTime);
      const updatedProgress = await getProgress();
      setProgress(updatedProgress);
      if (nextPayload.done || !nextPayload.item) {
        router.replace("/block-b-feedback");
        return;
      }
      setItem(nextPayload.item);
      setItemKey((k) => k + 1);
      resetForm();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to submit response.");
    } finally {
      setSubmitting(false);
    }
  }

  const assistiveNotice =
    locale === "el"
      ? "Το ATHENA είναι υποστηρικτικό εργαλείο human-in-the-loop: στόχος είναι να ενισχύει την κρίση του αρχαιολόγου, όχι να τον αντικαθιστά."
      : "ATHENA is an assistive human-in-the-loop tool: it is meant to support archaeologists, not replace expert judgment.";

  const blockBDescription =
    locale === "el"
      ? "Θα δείτε δύο εικόνες του ίδιου αντικειμένου. Η μία είναι η πραγματική εικόνα και η άλλη είναι αποκατάσταση που παράχθηκε από μοντέλο. Στην πραγματική αποκατάσταση το αρχικό δεν θα υπήρχε, αλλά εδώ εμφανίζεται επειδή η φθορά δημιουργήθηκε συνθετικά. Στόχος σας είναι να αποφασίσετε ποια εικόνα είναι η πραγματική και ποια η παραγόμενη, καθώς και να κρίνετε πόσο κοντά φτάνει το μοντέλο στο πραγματικό αποτέλεσμα."
      : "You will see two images of the same object. One is the real image and one is a model-generated restoration. In real restoration practice, the original would not exist; here it is shown only because the damage was created synthetically. Your task is to decide which image is real and which is generated, and to judge how close the generated result comes to the real one.";
  const blockBQuestion =
    locale === "el"
      ? "Ποια εικόνα είναι η πραγματική; Η άλλη είναι η παραγόμενη από μοντέλο."
      : "Which image is the real one? The other is model-generated.";
  const blockBSyntheticNotice =
    locale === "el"
      ? "Στην πραγματική αποκατάσταση το αρχικό δεν θα υπήρχε. Εδώ το βλέπετε μόνο επειδή η φθορά ήταν συνθετική, ώστε να ελέγξουμε πόσο κοντά φτάνει το μοντέλο στο πραγματικό."
      : "In real restoration practice, the original would not exist. It is shown here only because the damage was synthetic, so we can measure how close the model comes to the real image.";

  return (
    <WizardShell
      step="block-b"
      title={dictionary.blockB.title}
      description={blockBDescription}
      participantId={session?.participant_id}
      progressLabel={
        progress
          ? `${progress.block_b_completed}/${progress.block_b_total} ${dictionary.blockB.progressLabel} · ${formatI18n(
              dictionary.common.etaPrefix,
              { value: eta }
            )}`
          : undefined
      }
      progressValue={progressPercent}
      footer={
        <Button onClick={handleSubmit} disabled={submitting || !choice || confidence === null || !comment.trim()}>
          {submitting ? dictionary.blockB.submitting : dictionary.blockB.submit}
        </Button>
      }
    >
      {loading ? <p className="text-sm text-indigo-100/70">{dictionary.blockB.loading}</p> : null}
      {error ? <p className="text-sm text-red-600">{error}</p> : null}
      {!loading && item ? (
        <AnimatePresence mode="wait" initial={false}>
          <motion.div
            key={itemKey}
            initial={{ opacity: 0, x: 52 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -36 }}
            transition={{ duration: 0.38, ease: [0.22, 1, 0.36, 1] }}
            className="space-y-5"
          >
            <div className="space-y-2">
              <p className="text-xs" style={{ color: "rgba(179,190,223,0.5)" }}>
                {dictionary.blockB.itemPrefix} {item.item_order} / {item.total_items}
              </p>
              <div className="flex flex-wrap gap-2 text-[11px]">
                {item.is_practice ? (
                  <span className="rounded-full border px-2 py-0.5" style={{ borderColor: "rgba(103,214,255,0.35)", color: "rgba(103,214,255,0.9)" }}>
                    {dictionary.blockB.practice}
                  </span>
                ) : null}
                {item.is_anchor ? (
                  <span className="rounded-full border px-2 py-0.5" style={{ borderColor: "rgba(167,139,250,0.35)", color: "rgba(199,210,254,0.85)" }}>
                    {dictionary.blockB.anchor}
                  </span>
                ) : null}
              </div>
              <p className="text-sm font-medium" style={{ color: "rgba(231,236,255,0.88)" }}>
                {blockBQuestion}
              </p>
              <p className="rounded-lg border border-cyan-300/20 bg-cyan-300/5 px-3 py-2 text-xs text-indigo-100/75">
                {assistiveNotice}
              </p>
              <p className="rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-xs text-indigo-100/65">
                {blockBSyntheticNotice}
              </p>
            </div>

            <div className="grid grid-cols-2 gap-3">
              {(["A", "B"] as const).map((side) => {
                const isLeftCard = side === "A";
                const imgUrl = isLeftCard
                  ? (item.show_a_left ? item.option_a_url : item.option_b_url)
                  : (item.show_a_left ? item.option_b_url : item.option_a_url);
                const choiceValue: PairChoice = isLeftCard
                  ? (item.show_a_left ? "A" : "B")
                  : (item.show_a_left ? "B" : "A");
                const selected = choice === choiceValue;
                return (
                  <motion.button
                    key={side}
                    type="button"
                    aria-pressed={selected}
                    onClick={() => setChoice(choiceValue)}
                    onMouseUp={(event) => event.currentTarget.blur()}
                    whileHover={{ scale: 1.015 }}
                    whileTap={{ scale: 0.985 }}
                    className="relative overflow-hidden rounded-2xl border text-left outline-none transition-all duration-300 focus:outline-none focus-visible:outline-none focus-visible:ring-0"
                    style={{
                      borderColor: selected ? "rgba(103,214,255,0.75)" : "rgba(255,255,255,0.08)",
                      background: selected ? "rgba(103,214,255,0.06)" : "rgba(10,15,31,0.5)",
                      boxShadow: selected
                        ? "0 0 0 2px rgba(103,214,255,0.35), 0 8px 32px -8px rgba(103,214,255,0.25)"
                        : "none",
                    }}
                  >
                    {imgUrl ? (
                      <img
                        src={resolveAssetUrl(imgUrl)}
                        alt={`Option ${side}`}
                        className="pointer-events-none h-[42vh] w-full select-none object-contain"
                        draggable={false}
                      />
                    ) : null}
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
                      style={{ color: selected ? "rgba(103,214,255,0.9)" : "rgba(179,190,223,0.5)" }}
                    >
                      {side}
                    </div>
                  </motion.button>
                );
              })}
            </div>

            <div className="flex gap-2">
              {(["Tie", "Unsure"] as const).map((val) => (
                <ConfidenceButton
                  key={val}
                  label={val === "Tie" ? dictionary.blockB.tie : dictionary.blockB.unsure}
                  active={choice === val}
                  onClick={() => setChoice(val)}
                />
              ))}
            </div>

            <div className="athena-panel space-y-2 rounded-xl p-3">
              <Label>{dictionary.blockB.confidence}</Label>
              <div className="flex gap-2">
                {[1, 2, 3, 4, 5].map((value) => (
                  <ConfidenceButton
                    key={`conf-${value}`}
                    label={`${value}`}
                    active={confidence === value}
                    onClick={() => setConfidence(value)}
                  />
                ))}
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="block-b-comment">{dictionary.blockB.commentLabel}</Label>
              <Textarea
                id="block-b-comment"
                value={comment}
                onChange={(event) => setComment(event.target.value)}
                placeholder={dictionary.blockB.commentPlaceholder}
                maxLength={2000}
                required
              />
              <p className="text-xs text-indigo-100/45">{dictionary.blockB.commentHint}</p>
              <p className="text-xs text-indigo-100/45">
                {dictionary.blockB.commentExample1}
                <br />
                {dictionary.blockB.commentExample2}
              </p>
            </div>
          </motion.div>
        </AnimatePresence>
      ) : null}
    </WizardShell>
  );
}
