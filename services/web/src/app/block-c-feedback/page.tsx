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

export default function BlockCFeedbackPage() {
  const router = useRouter();
  const { dictionary } = useI18n();
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
          getStageFeedback("C"),
        ]);
        setSession(sessionPayload);
        setProgress(progressPayload);
        if (progressPayload.block_c_total <= 0) {
          router.replace("/complete");
          return;
        }
        if (progressPayload.block_b_total > 0 && !progressPayload.block_b_feedback_completed) {
          router.replace("/block-b-feedback");
          return;
        }
        if (progressPayload.block_c_completed < progressPayload.block_c_total) {
          router.replace("/block-c");
          return;
        }
        if (progressPayload.block_c_feedback_completed || feedbackPayload.completed) {
          router.replace("/complete");
          return;
        }
        setComment(feedbackPayload.comment ?? "");
      } catch (err) {
        if (err instanceof ApiError && err.status === 401) {
          router.replace("/");
          return;
        }
        setError(err instanceof Error ? err.message : "Failed to load Part 3 feedback.");
      } finally {
        setLoading(false);
      }
    }
    void bootstrap();
  }, [router]);

  async function handleSubmit() {
    if (comment.trim().length < 12) {
      setError(dictionary.blockCFeedback.minLengthError);
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      await submitStageFeedback("C", comment);
      router.replace("/complete");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to submit Part 3 feedback.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <WizardShell
      step="block-c-feedback"
      title={dictionary.blockCFeedback.title}
      description={dictionary.blockCFeedback.description}
      participantId={session?.participant_id}
      progressLabel={progress ? `${progress.block_c_completed}/${progress.block_c_total} ${dictionary.blockC.progressLabel}` : undefined}
      footer={
        <Button onClick={handleSubmit} disabled={loading || submitting}>
          {submitting ? dictionary.blockCFeedback.submitting : dictionary.blockCFeedback.submit}
        </Button>
      }
    >
      {loading ? <p className="text-sm text-indigo-100/70">{dictionary.common.loading}</p> : null}
      {error ? <p className="text-sm text-red-500">{error}</p> : null}
      {!loading ? (
        <>
          <InfoCallout title={dictionary.blockCFeedback.infoTitle} body={dictionary.blockCFeedback.infoBody} />
          <div className="space-y-2">
            <p className="text-sm text-indigo-100/80">{dictionary.blockCFeedback.prompt}</p>
            <Label htmlFor="block-c-stage-feedback">{dictionary.blockCFeedback.commentLabel}</Label>
            <Textarea
              id="block-c-stage-feedback"
              value={comment}
              onChange={(event) => setComment(event.target.value)}
              placeholder={dictionary.blockCFeedback.commentPlaceholder}
              maxLength={2000}
            />
          </div>
        </>
      ) : null}
    </WizardShell>
  );
}
