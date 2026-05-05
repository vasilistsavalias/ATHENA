"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import {
  ApiError,
  getProgress,
  getSession,
  getStageFeedback,
  ProgressInfo,
  SessionInfo,
  submitStageFeedback,
} from "@/lib/api";
import { useI18n } from "@/lib/i18n/context";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { WizardShell } from "@/components/wizard-shell";
import { InfoCallout } from "@/components/info-callout";

export default function BlockBFeedbackPage() {
  const router = useRouter();
  const { dictionary, locale } = useI18n();
  const [session, setSession] = useState<SessionInfo | null>(null);
  const [progress, setProgress] = useState<ProgressInfo | null>(null);
  const [comment, setComment] = useState("");
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function bootstrap() {
      try {
        const [sessionPayload, progressPayload, feedbackPayload] = await Promise.all([
          getSession(),
          getProgress(),
          getStageFeedback("B"),
        ]);
        setSession(sessionPayload);
        setProgress(progressPayload);
        if (progressPayload.block_b_completed < progressPayload.block_b_total) {
          router.replace("/block-b");
          return;
        }
        if (progressPayload.block_b_feedback_completed || feedbackPayload.completed) {
          if (progressPayload.block_c_total > 0 && progressPayload.block_c_completed < progressPayload.block_c_total) {
            router.replace("/block-c");
            return;
          }
          if (progressPayload.block_c_total > 0 && !progressPayload.block_c_feedback_completed) {
            router.replace("/block-c-feedback");
            return;
          }
          router.replace("/complete");
          return;
        }
        setComment(feedbackPayload.comment ?? "");
      } catch (err) {
        if (err instanceof ApiError && err.status === 401) {
          router.replace("/");
          return;
        }
        setError(err instanceof Error ? err.message : "Failed to load Part 2 feedback.");
      } finally {
        setLoading(false);
      }
    }
    void bootstrap();
  }, [router]);

  async function handleSubmit() {
    if (comment.trim().length < 12) {
      setError(dictionary.blockBFeedback.minLengthError);
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      await submitStageFeedback("B", comment);
      if (progress && progress.block_c_total > 0) {
        router.replace("/block-c");
      } else {
        router.replace("/complete");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to submit Part 2 feedback.");
    } finally {
      setSubmitting(false);
    }
  }

  const submitLabel =
    progress && progress.block_c_total > 0
      ? locale === "el"
        ? "Συνέχεια στο Μέρος 3"
        : "Continue to Part 3"
      : dictionary.blockBFeedback.submit;

  return (
    <WizardShell
      step="block-b-feedback"
      title={dictionary.blockBFeedback.title}
      description={dictionary.blockBFeedback.description}
      participantId={session?.participant_id}
      progressLabel={progress ? `${progress.block_b_completed}/${progress.block_b_total} ${dictionary.blockB.progressLabel}` : undefined}
      footer={
        <Button onClick={handleSubmit} disabled={loading || submitting}>
          {submitting ? dictionary.blockBFeedback.submitting : submitLabel}
        </Button>
      }
    >
      {loading ? <p className="text-sm text-indigo-100/70">{dictionary.common.loading}</p> : null}
      {error ? <p className="text-sm text-red-500">{error}</p> : null}
      {!loading ? (
        <>
          <InfoCallout title={dictionary.blockBFeedback.infoTitle} body={dictionary.blockBFeedback.infoBody} />
          <div className="space-y-2">
            <p className="text-sm text-indigo-100/80">{dictionary.blockBFeedback.prompt}</p>
            <Label htmlFor="block-b-stage-feedback">{dictionary.blockBFeedback.commentLabel}</Label>
            <Textarea
              id="block-b-stage-feedback"
              value={comment}
              onChange={(event) => setComment(event.target.value)}
              placeholder={dictionary.blockBFeedback.commentPlaceholder}
              maxLength={2000}
            />
          </div>
        </>
      ) : null}
    </WizardShell>
  );
}
