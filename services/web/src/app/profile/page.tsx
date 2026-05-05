"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import {
  ApiError,
  getProgress,
  getSession,
  getSessionProfile,
  SessionInfo,
  updateSessionProfile,
} from "@/lib/api";
import { nextRouteFromProgress } from "@/lib/flow";
import { useI18n } from "@/lib/i18n/context";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { WizardShell } from "@/components/wizard-shell";
import { InfoCallout } from "@/components/info-callout";

const DISCIPLINE_VALUES = [
  "Archaeology",
  "Philology / History / Archaeology",
  "Conservation / Restoration",
  "Museum / Curatorial",
  "Other",
] as const;

const DEFAULT_DISCIPLINE: (typeof DISCIPLINE_VALUES)[number] = "Archaeology";

export default function ProfilePage() {
  const router = useRouter();
  const { dictionary, locale } = useI18n();

  const [session, setSession] = useState<SessionInfo | null>(null);
  const [name, setName] = useState("");
  const [institution, setInstitution] = useState("");
  const [discipline, setDiscipline] = useState<(typeof DISCIPLINE_VALUES)[number]>(DEFAULT_DISCIPLINE);
  const [disciplineOther, setDisciplineOther] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function bootstrap() {
      try {
        const [sessionPayload, progressPayload, profilePayload] = await Promise.all([
          getSession(),
          getProgress(),
          getSessionProfile(),
        ]);
        setSession(sessionPayload);
        if (progressPayload.profile_completed) {
          router.replace(nextRouteFromProgress(progressPayload));
          return;
        }
        setName(profilePayload.name ?? "");
        setInstitution(profilePayload.institution ?? "");
        const incomingDiscipline = profilePayload.discipline;
        setDiscipline(
          incomingDiscipline && DISCIPLINE_VALUES.includes(incomingDiscipline as (typeof DISCIPLINE_VALUES)[number])
            ? (incomingDiscipline as (typeof DISCIPLINE_VALUES)[number])
            : DEFAULT_DISCIPLINE,
        );
        setDisciplineOther(profilePayload.discipline_other ?? "");
      } catch (err) {
        if (err instanceof ApiError && err.status === 401) {
          router.replace("/");
          return;
        }
        setError(err instanceof Error ? err.message : "Failed to load profile step.");
      } finally {
        setLoading(false);
      }
    }
    void bootstrap();
  }, [router]);

  async function handleSave() {
    if (!name.trim()) {
      setError(locale === "el" ? "Το όνομα είναι υποχρεωτικό." : "Name is required.");
      return;
    }
    if (!institution.trim()) {
      setError(locale === "el" ? "Το ίδρυμα είναι υποχρεωτικό." : "Institution is required.");
      return;
    }
    setSaving(true);
    setError(null);
    try {
      await updateSessionProfile({
        name,
        institution,
        discipline,
        discipline_other: discipline === "Other" ? disciplineOther : "",
      });
      const updatedProgress = await getProgress();
      router.replace(nextRouteFromProgress(updatedProgress));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save profile.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <WizardShell
      step="profile"
      title={dictionary.profile.title}
      description={dictionary.profile.description}
      participantId={session?.participant_id}
      footer={
        <Button onClick={handleSave} disabled={saving || loading}>
          {saving ? dictionary.profile.saving : dictionary.profile.save}
        </Button>
      }
    >
      {loading ? <p className="text-sm text-indigo-100/70">{dictionary.common.loading}</p> : null}
      {error ? <p className="text-sm text-red-500">{error}</p> : null}
      {!loading ? (
        <>
          <InfoCallout
            title={dictionary.profile.anonymityTitle}
            body={dictionary.profile.anonymityBody}
          />
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="profile-name">{dictionary.profile.nameLabel}</Label>
              <Input id="profile-name" value={name} onChange={(event) => setName(event.target.value)} maxLength={255} required />
            </div>
            <div className="space-y-2">
              <Label htmlFor="profile-institution">{dictionary.profile.institutionLabel}</Label>
              <Input
                id="profile-institution"
                value={institution}
                onChange={(event) => setInstitution(event.target.value)}
                maxLength={255}
                required
              />
            </div>
          </div>
          <div className="space-y-2">
            <Label htmlFor="profile-discipline">{dictionary.profile.disciplineLabel}</Label>
            <select
              id="profile-discipline"
              className="min-h-11 w-full rounded-xl border border-indigo-200/30 bg-indigo-950/50 px-3 py-2 text-sm text-indigo-50 outline-none focus:border-cyan-300/70 focus:ring-4 focus:ring-cyan-300/20"
              value={discipline}
              onChange={(event) => setDiscipline(event.target.value as (typeof DISCIPLINE_VALUES)[number])}
            >
              {DISCIPLINE_VALUES.map((option, index) => (
                <option key={option} value={option} className="bg-slate-950 text-slate-50">
                  {dictionary.profile.disciplineOptions[index]}
                </option>
              ))}
            </select>
          </div>
          {discipline === "Other" ? (
            <div className="space-y-2">
              <Label htmlFor="profile-discipline-other">{dictionary.profile.disciplineOtherLabel}</Label>
              <Input
                id="profile-discipline-other"
                value={disciplineOther}
                onChange={(event) => setDisciplineOther(event.target.value)}
                maxLength={255}
              />
            </div>
          ) : null}
        </>
      ) : null}
    </WizardShell>
  );
}
