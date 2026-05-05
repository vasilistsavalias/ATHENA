"use client";
/* eslint-disable @next/next/no-img-element */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";

import {
  ApiError,
  BlockAItem,
  getBlockANext,
  getProgress,
  getSession,
  ProgressInfo,
  resolveAssetUrl,
  SessionInfo,
  submitBlockA,
} from "@/lib/api";
import { formatEta, getAverageResponseMs, recordResponseTime } from "@/lib/progress";
import { useI18n } from "@/lib/i18n/context";
import { formatI18n } from "@/lib/i18n/format";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { WizardShell } from "@/components/wizard-shell";

type MetricKey = "authenticity" | "plausibility" | "confidence";

const BLOCK_A_TIME_KEY = "arch_eval_block_a_times";

function ScaleButton({
  value,
  selected,
  onClick,
}: {
  value: number;
  selected: boolean;
  onClick: () => void;
}) {
  return (
    <motion.button
      type="button"
      onClick={onClick}
      whileHover={{ scale: 1.12, y: -2 }}
      whileTap={{ scale: 0.88 }}
      animate={selected ? { scale: [1, 1.22, 1], transition: { duration: 0.28, ease: [0.22, 1, 0.36, 1] } } : {}}
      className={`h-12 w-12 rounded-xl border text-sm font-bold transition-colors ${
        selected
          ? "border-cyan-300 bg-cyan-300 text-slate-950"
          : "border-white/15 bg-white/5 text-indigo-100/80 hover:border-cyan-300/50 hover:bg-white/8"
      }`}
      style={selected ? { boxShadow: "0 4px 20px -4px rgba(103,214,255,0.5)" } : {}}
    >
      {value}
    </motion.button>
  );
}

export default function BlockAPage() {
  const router = useRouter();
  const { dictionary } = useI18n();
  const startedAtRef = useRef<number>(Date.now());

  const [session, setSession] = useState<SessionInfo | null>(null);
  const [progress, setProgress] = useState<ProgressInfo | null>(null);
  const [item, setItem] = useState<BlockAItem | null>(null);
  const [activeMetric, setActiveMetric] = useState<MetricKey>("authenticity");
  const [authenticity, setAuthenticity] = useState<number | null>(null);
  const [plausibility, setPlausibility] = useState<number | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [comment, setComment] = useState("");
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [itemKey, setItemKey] = useState(0);

  const progressPercent = useMemo(() => {
    if (!progress || progress.block_a_total === 0) {
      return 0;
    }
    return (progress.block_a_completed / progress.block_a_total) * 100;
  }, [progress]);

  const remaining = progress ? Math.max(0, progress.block_a_total - progress.block_a_completed) : 0;
  const avgMs = getAverageResponseMs(BLOCK_A_TIME_KEY);
  const eta = formatEta((remaining * avgMs) / 1000);

  function resetForm() {
    setAuthenticity(null);
    setPlausibility(null);
    setConfidence(null);
    setComment("");
    setActiveMetric("authenticity");
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
      if (progressPayload.block_a_completed >= progressPayload.block_a_total) {
        router.replace(progressPayload.block_a_feedback_completed ? "/block-b" : "/block-a-feedback");
        return;
      }
      const next = await getBlockANext();
      if (next.done || !next.item) {
        router.replace("/block-b");
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
      setError(err instanceof Error ? err.message : "Failed to load Block A.");
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
      if (["a", "A"].includes(event.key)) {
        setActiveMetric("authenticity");
      } else if (["p", "P"].includes(event.key)) {
        setActiveMetric("plausibility");
      } else if (["c", "C"].includes(event.key)) {
        setActiveMetric("confidence");
      } else if (/^[1-5]$/.test(event.key)) {
        const score = Number(event.key);
        if (activeMetric === "authenticity") {
          setAuthenticity(score);
        } else if (activeMetric === "plausibility") {
          setPlausibility(score);
        } else {
          setConfidence(score);
        }
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [activeMetric]);

  async function handleSubmit() {
    if (!item || authenticity === null || plausibility === null || confidence === null) {
      setError(dictionary.blockA.completeScalesError);
      return;
    }
    const trimmedComment = comment.trim();
    if (!trimmedComment) {
      setError(dictionary.blockA.commentRequiredError);
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      const responseTime = Date.now() - startedAtRef.current;
      const nextPayload = await submitBlockA({
        assignment_id: item.assignment_id,
        authenticity_likelihood: authenticity,
        archaeological_plausibility: plausibility,
        confidence,
        comment: trimmedComment,
        response_time_ms: responseTime,
      });
      recordResponseTime(BLOCK_A_TIME_KEY, responseTime);
      const updatedProgress = await getProgress();
      setProgress(updatedProgress);
      if (nextPayload.done || !nextPayload.item) {
        router.replace("/block-a-feedback");
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

  return (
    <WizardShell
      step="block-a"
      title={dictionary.blockA.title}
      description={dictionary.blockA.description}
      participantId={session?.participant_id}
      progressLabel={
        progress
          ? `${progress.block_a_completed}/${progress.block_a_total} ${dictionary.blockA.progressLabel} · ${formatI18n(
              dictionary.common.etaPrefix,
              { value: eta }
            )}`
          : undefined
      }
      progressValue={progressPercent}
      footer={
        <Button
          onClick={handleSubmit}
          disabled={
            submitting ||
            authenticity === null ||
            plausibility === null ||
            confidence === null ||
            !comment.trim()
          }
        >
          {submitting ? dictionary.blockA.submitting : dictionary.blockA.submit}
        </Button>
      }
    >
      {loading ? <p className="text-sm text-indigo-100/70">{dictionary.blockA.loading}</p> : null}
      {error ? <p className="text-sm text-red-600">{error}</p> : null}
      {!loading && item ? (
        <AnimatePresence mode="wait" initial={false}>
          <motion.div
            key={itemKey}
            initial={{ opacity: 0, x: 52 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -36 }}
            transition={{ duration: 0.38, ease: [0.22, 1, 0.36, 1] }}
          >
          <div className="space-y-2">
            <p className="text-xs" style={{ color: "rgba(179,190,223,0.5)" }}>
              {dictionary.blockA.itemPrefix} {item.item_order} / {item.total_items}
            </p>
            <div
              className="overflow-hidden rounded-2xl border"
              style={{ borderColor: "rgba(255,255,255,0.08)", background: "rgba(10,15,31,0.5)" }}
            >
              <img
                src={resolveAssetUrl(item.image_url)}
                alt={`Block A sample ${item.sample_id}`}
                className="h-[52vh] w-full object-contain"
              />
            </div>
          </div>

          <div className="grid gap-4 md:grid-cols-3">
            <div className="athena-panel space-y-2 rounded-xl p-3">
              <button type="button" className="text-left" onClick={() => setActiveMetric("authenticity")}>
                <Label className={activeMetric === "authenticity" ? "text-cyan-100" : "text-indigo-100/70"}>
                  {dictionary.blockA.authenticity}
                </Label>
              </button>
              <div className="flex gap-2">
                {[1, 2, 3, 4, 5].map((value) => (
                  <ScaleButton
                    key={`auth-${value}`}
                    value={value}
                    selected={authenticity === value}
                    onClick={() => setAuthenticity(value)}
                  />
                ))}
              </div>
            </div>

            <div className="athena-panel space-y-2 rounded-xl p-3">
              <button type="button" className="text-left" onClick={() => setActiveMetric("plausibility")}>
                <Label className={activeMetric === "plausibility" ? "text-cyan-100" : "text-indigo-100/70"}>
                  {dictionary.blockA.plausibility}
                </Label>
              </button>
              <div className="flex gap-2">
                {[1, 2, 3, 4, 5].map((value) => (
                  <ScaleButton
                    key={`plaus-${value}`}
                    value={value}
                    selected={plausibility === value}
                    onClick={() => setPlausibility(value)}
                  />
                ))}
              </div>
            </div>

            <div className="athena-panel space-y-2 rounded-xl p-3">
              <button type="button" className="text-left" onClick={() => setActiveMetric("confidence")}>
                <Label className={activeMetric === "confidence" ? "text-cyan-100" : "text-indigo-100/70"}>
                  {dictionary.blockA.confidence}
                </Label>
              </button>
              <div className="flex gap-2">
                {[1, 2, 3, 4, 5].map((value) => (
                  <ScaleButton
                    key={`conf-${value}`}
                    value={value}
                    selected={confidence === value}
                    onClick={() => setConfidence(value)}
                  />
                ))}
              </div>
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="block-a-comment">{dictionary.blockA.commentLabel}</Label>
            <Textarea
              id="block-a-comment"
              value={comment}
              onChange={(event) => setComment(event.target.value)}
              placeholder={dictionary.blockA.commentPlaceholder}
              maxLength={2000}
              required
            />
            <p className="text-xs text-indigo-100/45">{dictionary.blockA.commentHint}</p>
            <p className="text-xs text-indigo-100/45">
              {dictionary.blockA.commentExample1}
              <br />
              {dictionary.blockA.commentExample2}
            </p>
          </div>
          </motion.div>
        </AnimatePresence>
      ) : null}
    </WizardShell>
  );
}
