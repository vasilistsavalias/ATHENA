"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import { ApiError, getProgress, getSession, ProgressInfo, SessionInfo } from "@/lib/api";
import { nextRouteFromProgress } from "@/lib/flow";
import { useI18n } from "@/lib/i18n/context";
import { formatI18n } from "@/lib/i18n/format";
import { Button } from "@/components/ui/button";
import { WizardShell } from "@/components/wizard-shell";

export default function ConsentPage() {
  const router = useRouter();
  const { dictionary, locale } = useI18n();
  const [session, setSession] = useState<SessionInfo | null>(null);
  const [progress, setProgress] = useState<ProgressInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function bootstrap() {
      try {
        const [sessionPayload, progressPayload] = await Promise.all([getSession(), getProgress()]);
        setSession(sessionPayload);
        setProgress(progressPayload);
      } catch (err) {
        if (err instanceof ApiError && err.status === 401) {
          router.replace("/");
          return;
        }
        setError("Failed to load session.");
      } finally {
        setLoading(false);
      }
    }
    void bootstrap();
  }, [router]);

  const totalTasks = progress ? progress.block_a_total + progress.block_b_total + progress.block_c_total : 0;
  const hasBlockA = (progress ? Math.max(progress.block_a_total, session?.campaign.block_a_target_count ?? 0) : (session?.campaign.block_a_target_count ?? 0)) > 0;
  const hasBlockB = (progress ? Math.max(progress.block_b_total, session?.campaign.block_b_target_count ?? 0) : (session?.campaign.block_b_target_count ?? 0)) > 0;
  const hasBlockC = (progress ? Math.max(progress.block_c_total, session?.campaign.block_c_target_count ?? 0) : (session?.campaign.block_c_target_count ?? 0)) > 0;
  const taskModeNotice = hasBlockA && hasBlockB && hasBlockC
    ? locale === "el"
      ? "Η τρέχουσα καμπάνια περιλαμβάνει και τα τρία μέρη: Μέρος 1 (βαθμολόγηση), Μέρος 2 (σύγκριση δίπλα-δίπλα) και Μέρος 3 (σύγκριση πολλαπλών μοντέλων)."
      : "This campaign includes all three parts: Part 1 (rating), Part 2 (side-by-side comparison), and Part 3 (multi-model comparison)."
    : hasBlockA && hasBlockB
      ? locale === "el"
        ? "Η τρέχουσα καμπάνια περιλαμβάνει τα Μέρη 1 και 2: βαθμολόγηση και σύγκριση."
        : "This campaign includes Part 1 (rating) and Part 2 (comparison)."
      : hasBlockB
        ? locale === "el"
          ? "Η τρέχουσα καμπάνια είναι μόνο συγκριτική: εμφανίζεται μόνο το Μέρος 2. Αυτό είναι αναμενόμενο και ορίζεται από τη διαμόρφωση της καμπάνιας."
          : "This campaign is comparison-only: only Part 2 is active. This is expected and set by campaign configuration."
        : locale === "el"
          ? "Η τρέχουσα καμπάνια ακολουθεί προσαρμοσμένη διάταξη με βάση τη διαμόρφωση της μελέτης."
          : "This campaign uses a custom task layout based on the study configuration.";
  const humanInLoopNotice =
    locale === "el"
      ? "Σκοπός του ATHENA είναι υποστηρικτικός (human-in-the-loop): να βοηθά τον αρχαιολόγο με προτάσεις αποκατάστασης, όχι να αντικαθιστά την επιστημονική κρίση ή τον επαγγελματία."
      : "ATHENA is assistive (human-in-the-loop): it supports archaeologists with restoration guidance and does not replace expert judgment or professional work.";
  const calibrationNotice = hasBlockB || hasBlockC
    ? locale === "el"
      ? "Στα συγκριτικά μέρη βλέπετε τη φθαρμένη είσοδο και υποψήφιες αποκαταστάσεις. Στο Μέρος 2 συγκρίνετε δύο επιλογές, ενώ στο Μέρος 3 συγκρίνετε τέσσερις επιλογές από διαφορετικά μοντέλα. Η ταυτότητα των υποψηφίων παραμένει κρυφή και η διάταξη χρησιμοποιείται αποκλειστικά για ερευνητική βαθμονόμηση."
      : "In the comparison sections you see the damaged input and candidate restorations. In Part 2 you compare two options, while in Part 3 you compare four options from different models. Candidate identities stay hidden and this setup is used only for research calibration."
    : null;
  const consentBullets = hasBlockA
    ? dictionary.consent.bullets
    : [
        dictionary.consent.bullets[0],
        locale === "el"
          ? "Η σύγκριση ξεκινά αμέσως μετά το σύντομο βήμα προφίλ."
          : "The comparison task starts immediately after the short profile step.",
        dictionary.consent.bullets[2],
      ];
  const commentNotice = hasBlockA
    ? dictionary.consent.commentNotice
    : locale === "el"
      ? "Στο τέλος της σύγκρισης θα σας ζητηθεί ένα σύντομο σχόλιο ειδικού, ώστε να καταγραφούν παρατηρήσεις που δεν φαίνονται μόνο από τις επιλογές A/B."
      : "At the end of the comparison stage, we will ask for one short expert comment so domain-specific observations are not reduced to A/B choices alone.";
  const dataUseNotice = dictionary.consent.dataUseNotice ?? (
    locale === "el"
      ? "Οι βαθμολογίες, οι επιλογές, οι χρόνοι απάντησης και τα σχόλιά σας αποθηκεύονται ανώνυμα για ερευνητική ανάλυση, γραφήματα και αναφορά στη διπλωματική."
      : "Your ratings, selections, timings, and comments are stored in anonymised form for research analysis, thesis figures, and reporting."
  );

  return (
    <WizardShell
      step="consent"
      title={dictionary.consent.title}
      description={dictionary.consent.description}
      participantId={session?.participant_id}
    >
      {loading ? <p className="text-sm text-indigo-100/70">{dictionary.common.loading}</p> : null}
      {error ? <p className="text-sm text-red-600">{error}</p> : null}
      {!loading && !error ? (
        <>
          <div className="athena-panel space-y-3 rounded-xl p-4 text-sm text-indigo-100/80">
            <p>{dictionary.consent.intro}</p>
            <ul className="list-disc space-y-1 pl-5">
              {consentBullets.map((bullet) => (
                <li key={bullet}>{bullet}</li>
              ))}
            </ul>
            <p>{dictionary.consent.profileNotice}</p>
            <p className="athena-note rounded-lg px-3 py-2">{taskModeNotice}</p>
            <p className="athena-note rounded-lg px-3 py-2">{humanInLoopNotice}</p>
            {calibrationNotice ? <p className="athena-note rounded-lg px-3 py-2">{calibrationNotice}</p> : null}
            <p>{commentNotice}</p>
            <p className="athena-note rounded-lg px-3 py-2">
              {dataUseNotice}
            </p>
            <p className="athena-note rounded-lg px-3 py-2">
              {dictionary.consent.oneTimeNotice}
            </p>
            <p>
              {formatI18n(dictionary.consent.totalItems, { count: totalTasks })}
            </p>
            {session?.campaign.protocol_version ? (
              <p className="text-xs text-indigo-100/60">
                {dictionary.common.protocolVersionLabel}: {session.campaign.protocol_version}
              </p>
            ) : null}
          </div>
          <div className="flex justify-end">
            <Button onClick={() => router.push(progress ? nextRouteFromProgress(progress) : "/profile")}>
              {dictionary.consent.cta}
            </Button>
          </div>
        </>
      ) : null}
    </WizardShell>
  );
}
