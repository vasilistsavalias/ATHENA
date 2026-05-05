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

export default function BlockAFeedbackPage() {
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
          getStageFeedback("A"),
        ]);
        setSession(sessionPayload);
        setProgress(progressPayload);
        if (progressPayload.block_a_completed < progressPayload.block_a_total) {
          router.replace("/block-a");
          return;
        }
        if (progressPayload.block_a_feedback_completed || feedbackPayload.completed) {
          router.replace("/block-b");
          return;
        }
        setComment(feedbackPayload.comment ?? "");
      } catch (err) {
        if (err instanceof ApiError && err.status === 401) {
          router.replace("/");
          return;
        }
        setError(err instanceof Error ? err.message : "Failed to load Part 1 feedback.");
      } finally {
        setLoading(false);
      }
    }
    void bootstrap();
  }, [router]);

  async function handleSubmit() {
    if (comment.trim().length < 12) {
      setError(dictionary.blockAFeedback.minLengthError);
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      await submitStageFeedback("A", comment);
      router.replace("/block-b");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to submit Part 1 feedback.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <WizardShell
      step="block-a-feedback"
      title={dictionary.blockAFeedback.title}
      description={dictionary.blockAFeedback.description}
      participantId={session?.participant_id}
      progressLabel={progress ? `${progress.block_a_completed}/${progress.block_a_total} ${dictionary.blockA.progressLabel}` : undefined}
      footer={
        <Button onClick={handleSubmit} disabled={loading || submitting}>
          {submitting ? dictionary.blockAFeedback.submitting : dictionary.blockAFeedback.submit}
        </Button>
      }
    >
      {loading ? <p className="text-sm text-indigo-100/70">{dictionary.common.loading}</p> : null}
      {error ? <p className="text-sm text-red-500">{error}</p> : null}
      {!loading ? (
        <>
          <InfoCallout title={dictionary.blockAFeedback.infoTitle} body={dictionary.blockAFeedback.infoBody} />
          <div className="space-y-2">
            <p className="text-sm text-indigo-100/80">{dictionary.blockAFeedback.prompt}</p>
            <Label htmlFor="block-a-stage-feedback">{dictionary.blockAFeedback.commentLabel}</Label>
            <Textarea
              id="block-a-stage-feedback"
              value={comment}
              onChange={(event) => setComment(event.target.value)}
              placeholder={dictionary.blockAFeedback.commentPlaceholder}
              maxLength={2000}
            />
          </div>
        </>
      ) : null}
    </WizardShell>
  );
}
