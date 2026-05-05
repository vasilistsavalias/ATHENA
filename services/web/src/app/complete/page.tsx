"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import confetti from "canvas-confetti";

import { ApiError, completeSession, getProgress, getSession, ProgressInfo, SessionInfo } from "@/lib/api";
import { nextRouteFromProgress } from "@/lib/flow";
import { useI18n } from "@/lib/i18n/context";
import { formatI18n } from "@/lib/i18n/format";
import { Button } from "@/components/ui/button";
import { WizardShell } from "@/components/wizard-shell";

export default function CompletePage() {
  const router = useRouter();
  const { dictionary } = useI18n();
  const [session, setSession] = useState<SessionInfo | null>(null);
  const [progress, setProgress] = useState<ProgressInfo | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const confettiFired = useRef(false);

  useEffect(() => {
    async function bootstrap() {
      try {
        const [sessionPayload, progressPayload] = await Promise.all([getSession(), getProgress()]);
        setSession(sessionPayload);
        setProgress(progressPayload);
        if (progressPayload.is_complete) {
          await completeSession().catch(() => undefined);
        }
      } catch (err) {
        if (err instanceof ApiError && err.status === 401) {
          router.replace("/");
          return;
        }
        setMessage("Unable to load completion state.");
      } finally {
        setLoading(false);
      }
    }
    void bootstrap();
  }, [router]);

  const isComplete = progress?.is_complete ?? false;

  // Fire confetti once when isComplete becomes true
  useEffect(() => {
    if (isComplete && !confettiFired.current) {
      confettiFired.current = true;
      const end = Date.now() + 1800;
      const colors = ["#67d6ff", "#818cf8", "#c084fc", "#ffffff"];
      (function frame() {
        confetti({
          particleCount: 4,
          angle: 60,
          spread: 55,
          origin: { x: 0 },
          colors,
        });
        confetti({
          particleCount: 4,
          angle: 120,
          spread: 55,
          origin: { x: 1 },
          colors,
        });
        if (Date.now() < end) requestAnimationFrame(frame);
      })();
    }
  }, [isComplete]);

  const statItems = !loading && !message && progress
      ? [
        { label: dictionary.complete.blockALabel, value: `${progress.block_a_completed}/${progress.block_a_total}` },
        { label: dictionary.complete.blockBLabel, value: `${progress.block_b_completed}/${progress.block_b_total}` },
        ...(progress.block_c_total > 0
          ? [{ label: dictionary.complete.blockCLabel, value: `${progress.block_c_completed}/${progress.block_c_total}` }]
          : []),
      ]
    : [];

  return (
    <WizardShell
      step="complete"
      title={isComplete ? dictionary.complete.titleDone : dictionary.complete.titlePending}
      description={isComplete ? dictionary.complete.descriptionDone : dictionary.complete.descriptionPending}
      participantId={session?.participant_id}
    >
      {loading ? (
        <p className="text-sm" style={{ color: "rgba(179,190,223,0.55)" }}>
          {dictionary.common.loading}
        </p>
      ) : null}
      {message ? <p className="text-sm text-red-400/80">{message}</p> : null}

      {!loading && !message ? (
        <div className="space-y-6">
          {/* Animated checkmark */}
          {isComplete && (
            <motion.div
              initial={{ opacity: 0, scale: 0.4 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
              className="flex justify-center"
            >
              <div
                className="flex h-20 w-20 items-center justify-center rounded-full"
                style={{
                  background: "linear-gradient(135deg, rgba(103,214,255,0.18), rgba(129,140,248,0.18))",
                  border: "2px solid rgba(103,214,255,0.5)",
                  boxShadow: "0 0 40px rgba(103,214,255,0.2)",
                }}
              >
                <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
                  <motion.path
                    d="M8 20L17 29L32 11"
                    stroke="#67d6ff"
                    strokeWidth="3.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    initial={{ pathLength: 0 }}
                    animate={{ pathLength: 1 }}
                    transition={{ duration: 0.55, delay: 0.18, ease: "easeOut" }}
                  />
                </svg>
              </div>
            </motion.div>
          )}

          {/* Stats */}
          <div className={`grid grid-cols-1 gap-3 ${statItems.length >= 3 ? "sm:grid-cols-3" : "sm:grid-cols-2"}`}>
            {statItems.map((stat, i) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.45, delay: 0.1 + i * 0.1, ease: [0.22, 1, 0.36, 1] }}
                className="athena-panel rounded-2xl p-4 text-center"
              >
                <p className="text-2xl font-bold" style={{ color: "#67d6ff" }}>
                  {stat.value}
                </p>
                <p className="mt-1 text-xs" style={{ color: "rgba(179,190,223,0.55)" }}>
                  {stat.label}
                </p>
              </motion.div>
            ))}
          </div>

          {/* Completion code / continue button */}
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45, delay: 0.3, ease: [0.22, 1, 0.36, 1] }}
            className="athena-panel rounded-2xl p-4"
          >
            {isComplete ? (
              <p className="text-sm" style={{ color: "rgba(179,190,223,0.75)" }}>
                {formatI18n(dictionary.complete.codeText, { code: session?.participant_id ?? "" })}
              </p>
            ) : (
              <Button onClick={() => router.push(progress ? nextRouteFromProgress(progress) : "/consent")}>
                {dictionary.complete.continue}
              </Button>
            )}
          </motion.div>
        </div>
      ) : null}
    </WizardShell>
  );
}
