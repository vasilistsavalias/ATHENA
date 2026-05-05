"use client";

import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import {
  AlertTriangle,
  Download,
  FileSpreadsheet,
  FileText,
  Filter,
  RefreshCcw,
  Search,
  ShieldCheck,
  Users,
} from "lucide-react";

import {
  adminExportCsv,
  adminExportJson,
  adminImportPack,
  adminLogout,
  adminQualityReport,
  ApiError,
  downloadText,
  getAdminDashboard,
  getAdminSession,
  type AdminCommentRow,
  type AdminDashboard,
} from "@/lib/api";
import { StudyPage } from "@/components/study-page";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

function formatDate(value: string | null) {
  if (!value) {
    return "—";
  }
  return new Date(value).toLocaleString();
}

function StatCard({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint: string;
}) {
  return (
    <Card className="rounded-2xl border-indigo-200/15 bg-indigo-950/45">
      <CardContent className="space-y-1 p-5">
        <p className="text-xs uppercase tracking-[0.22em] text-indigo-100/45">{label}</p>
        <p className="text-3xl font-semibold text-indigo-50">{value}</p>
        <p className="text-sm text-indigo-100/65">{hint}</p>
      </CardContent>
    </Card>
  );
}

function CommentCard({ entry }: { entry: AdminCommentRow }) {
  return (
    <div className="rounded-2xl border border-indigo-200/12 bg-indigo-950/35 p-4">
      <div className="mb-2 flex items-center justify-between gap-3">
        <div>
          <p className="text-sm font-semibold text-indigo-50">{entry.participant_id}</p>
          <p className="text-xs text-indigo-100/55">
            {entry.source === "stage_feedback" ? `Block ${entry.block} wrap-up` : `Block ${entry.block} item comment`}
            {entry.sample_id ? ` • ${entry.sample_id}` : ""}
          </p>
        </div>
        <span className="text-xs text-indigo-100/45">{formatDate(entry.created_at)}</span>
      </div>
      <p className="text-sm leading-6 text-indigo-100/82">{entry.comment}</p>
    </div>
  );
}

export default function AdminPage() {
  const router = useRouter();
  const [dashboard, setDashboard] = useState<AdminDashboard | null>(null);
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState<"all" | "completed" | "active">("all");
  const [busy, setBusy] = useState(false);
  const [exporting, setExporting] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [packDir, setPackDir] = useState("");
  const [stage13Samples, setStage13Samples] = useState("");
  const [campaignName, setCampaignName] = useState("ATHENA Expert Campaign");
  const [seed, setSeed] = useState(42);

  const loadDashboard = useCallback(async () => {
    try {
      await getAdminSession();
      const payload = await getAdminDashboard();
      setDashboard(payload);
      setCampaignName(payload.campaign?.name ?? "ATHENA Expert Campaign");
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        router.replace("/admin/login");
        return;
      }
      setStatus(err instanceof Error ? err.message : "Failed to load admin dashboard.");
    }
  }, [router]);

  useEffect(() => {
    void loadDashboard();
  }, [loadDashboard]);

  const filteredParticipants = useMemo(() => {
    const rows = dashboard?.participants ?? [];
    return rows.filter((participant) => {
      const matchesSearch =
        !search.trim() ||
        [participant.participant_id, participant.name, participant.institution, participant.discipline]
          .filter(Boolean)
          .join(" ")
          .toLowerCase()
          .includes(search.trim().toLowerCase());
      const matchesStatus =
        statusFilter === "all" ||
        (statusFilter === "completed" ? participant.status === "completed" : participant.status !== "completed");
      return matchesSearch && matchesStatus;
    });
  }, [dashboard?.participants, search, statusFilter]);

  async function runExport(type: "csv" | "json" | "quality") {
    setExporting(type);
    setStatus(null);
    try {
      if (type === "csv") {
        const payload = await adminExportCsv();
        downloadText(payload, "athena_responses.csv", "text/csv");
      } else if (type === "json") {
        const payload = await adminExportJson();
        downloadText(JSON.stringify(payload, null, 2), "athena_responses.json", "application/json");
      } else {
        const payload = await adminQualityReport();
        downloadText(JSON.stringify(payload, null, 2), "athena_quality_report.json", "application/json");
      }
      setStatus(`Export ${type.toUpperCase()} completed.`);
    } catch (err) {
      setStatus(err instanceof Error ? err.message : `Export ${type} failed.`);
    } finally {
      setExporting(null);
    }
  }

  async function handleImport(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setStatus(null);
    try {
      const payload = await adminImportPack(undefined, {
        pack_dir: packDir,
        campaign_name: campaignName,
        seed,
        stage13_samples: stage13Samples || undefined,
        activate: true,
        disjoint_blocks: true,
      });
      setStatus(`Imported campaign ${payload.campaign_name} (#${payload.campaign_id}).`);
      await loadDashboard();
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "Import failed.");
    } finally {
      setBusy(false);
    }
  }

  async function handleLogout() {
    await adminLogout();
    router.replace("/admin/login");
  }

  if (!dashboard) {
    return (
      <StudyPage
        title="ATHENA Admin Dashboard"
        description="Loading campaign analytics, participant summaries, and export tools."
      >
        <div className="rounded-2xl border border-indigo-200/10 bg-indigo-950/35 p-6 text-sm text-indigo-100/70">
          Loading admin dashboard…
        </div>
      </StudyPage>
    );
  }

  const { campaign, stats } = dashboard;
  const hasActiveCampaign = Boolean(campaign);

  return (
    <StudyPage
      title="ATHENA Admin Dashboard"
      description="Operational analytics, qualitative feedback, and export tooling for the live expert-evaluation study."
    >
      <section className="overflow-hidden rounded-[28px] border border-cyan-300/18 bg-[radial-gradient(circle_at_top_left,rgba(103,214,255,0.18),transparent_28%),radial-gradient(circle_at_bottom_right,rgba(129,140,248,0.16),transparent_30%),linear-gradient(160deg,rgba(18,28,56,0.94),rgba(10,16,32,0.98))]">
        <div className="grid gap-6 p-6 lg:grid-cols-[1.4fr_0.9fr]">
          <div className="space-y-4">
            <div className="flex items-start gap-4">
              <div className="rounded-2xl border border-cyan-300/20 bg-cyan-300/10 p-3 text-cyan-200">
                <ShieldCheck className="h-6 w-6" />
              </div>
              <div>
                <div className="mb-2 inline-flex rounded-full border border-cyan-300/20 bg-cyan-300/10 px-3 py-1 text-xs font-semibold tracking-[0.18em] text-cyan-200">
                  {campaign ? "ACTIVE CAMPAIGN" : "ADMIN READY"}
                </div>
                <h2 className="text-3xl font-semibold tracking-tight text-indigo-50">
                  {campaign ? campaign.name : "No active campaign imported"}
                </h2>
                <p className="mt-2 max-w-2xl text-sm leading-6 text-indigo-100/72">
                  Separate admin authentication is now in place. Participant invite codes cannot reach this surface.
                  All exports and analytics run against the active backend campaign without needing Render or Vercel
                  console access.
                </p>
              </div>
            </div>

            <div className="grid gap-3 sm:grid-cols-3">
              <div className="rounded-2xl border border-indigo-200/12 bg-indigo-950/40 p-4">
                <p className="text-xs uppercase tracking-[0.2em] text-indigo-100/45">Seed</p>
                <p className="mt-2 text-2xl font-semibold text-indigo-50">{campaign?.seed ?? "—"}</p>
              </div>
              <div className="rounded-2xl border border-indigo-200/12 bg-indigo-950/40 p-4">
                <p className="text-xs uppercase tracking-[0.2em] text-indigo-100/45">Block A target</p>
                <p className="mt-2 text-2xl font-semibold text-indigo-50">{campaign?.block_a_target_count ?? "—"}</p>
              </div>
              <div className="rounded-2xl border border-indigo-200/12 bg-indigo-950/40 p-4">
                <p className="text-xs uppercase tracking-[0.2em] text-indigo-100/45">Block B target</p>
                <p className="mt-2 text-2xl font-semibold text-indigo-50">{campaign?.block_b_target_count ?? "—"}</p>
              </div>
            </div>
          </div>

          <div className="rounded-3xl border border-indigo-200/14 bg-indigo-950/38 p-5">
            <div className="mb-4 flex items-center justify-between gap-3">
              <div>
                <p className="text-sm font-semibold text-indigo-50">Export and control</p>
                <p className="text-sm text-indigo-100/62">Download clean snapshots or refresh live metrics.</p>
              </div>
              <div className="rounded-2xl border border-indigo-200/12 bg-indigo-200/8 p-3 text-indigo-100/78">
                <Download className="h-5 w-5" />
              </div>
            </div>
            <div className="grid gap-2 sm:grid-cols-2">
              <Button disabled={exporting !== null || !hasActiveCampaign} onClick={() => runExport("csv")}>
                <FileSpreadsheet className="mr-2 h-4 w-4" />
                {exporting === "csv" ? "Exporting…" : "Export CSV"}
              </Button>
              <Button disabled={exporting !== null || !hasActiveCampaign} variant="outline" onClick={() => runExport("json")}>
                <FileText className="mr-2 h-4 w-4" />
                {exporting === "json" ? "Exporting…" : "Export JSON"}
              </Button>
              <Button disabled={exporting !== null || !hasActiveCampaign} variant="outline" onClick={() => runExport("quality")}>
                <AlertTriangle className="mr-2 h-4 w-4" />
                {exporting === "quality" ? "Exporting…" : "Quality report"}
              </Button>
              <Button variant="ghost" onClick={() => void loadDashboard()}>
                <RefreshCcw className="mr-2 h-4 w-4" />
                Refresh
              </Button>
            </div>
            <Button className="mt-3 w-full" variant="ghost" onClick={() => void handleLogout()}>
              Sign out
            </Button>
            {!hasActiveCampaign ? (
              <p className="mt-3 text-xs leading-5 text-indigo-100/58">
                No active campaign is loaded yet. Import a pack first; exports unlock automatically afterward.
              </p>
            ) : null}
          </div>
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <StatCard label="Participants" value={String(stats.participants_total)} hint={`${stats.participants_active} active / ${stats.participants_completed} completed`} />
        <StatCard label="Profiles" value={String(stats.profiles_completed)} hint="Profile acknowledgements completed" />
        <StatCard
          label="Responses"
          value={String(stats.block_a_responses + stats.block_b_responses + stats.block_c_responses)}
          hint={`${stats.block_a_responses} Block A • ${stats.block_b_responses} Block B • ${stats.block_c_responses} Block C`}
        />
        <StatCard label="Quality flags" value={String(stats.attention_flags_total)} hint={`${stats.block_a_feedback_completed} A wrap-ups • ${stats.block_b_feedback_completed} B wrap-ups`} />
      </section>

      <section className="grid gap-5 xl:grid-cols-[1.3fr_0.7fr]">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-xl">
              <Users className="h-5 w-5 text-cyan-200" />
              Participant overview
            </CardTitle>
            <CardDescription>
              Search by participant code, institution, or discipline. Review completion progress and qualitative notes.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-3 md:grid-cols-[1fr_200px]">
              <div className="relative">
                <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-indigo-100/40" />
                <Input
                  className="pl-10"
                  value={search}
                  onChange={(event) => setSearch(event.target.value)}
                  placeholder="Search participant, institution, or discipline"
                />
              </div>
              <div className="relative">
                <Filter className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-indigo-100/40" />
                <select
                  className="h-11 w-full rounded-xl border border-indigo-200/30 bg-indigo-950/50 pl-10 pr-3 text-sm text-indigo-50 outline-none focus:border-cyan-300/70 focus:ring-4 focus:ring-cyan-300/20"
                  value={statusFilter}
                  onChange={(event) => setStatusFilter(event.target.value as "all" | "completed" | "active")}
                >
                  <option value="all">All statuses</option>
                  <option value="active">Active only</option>
                  <option value="completed">Completed only</option>
                </select>
              </div>
            </div>

            <div className="overflow-hidden rounded-2xl border border-indigo-200/12">
              <div className="max-h-[560px] overflow-auto">
                <table className="min-w-full divide-y divide-indigo-200/10 text-sm">
                  <thead className="sticky top-0 bg-slate-950/95 backdrop-blur">
                    <tr className="text-left text-xs uppercase tracking-[0.16em] text-indigo-100/45">
                      <th className="px-4 py-3">Participant</th>
                      <th className="px-4 py-3">Profile</th>
                      <th className="px-4 py-3">Progress</th>
                      <th className="px-4 py-3">Flags</th>
                      <th className="px-4 py-3">Comments</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-indigo-200/8">
                    {filteredParticipants.map((participant) => (
                      <tr key={participant.participant_id} className="align-top">
                        <td className="px-4 py-4">
                          <div className="font-semibold text-indigo-50">{participant.participant_id}</div>
                          <div className="mt-1 text-xs text-indigo-100/55">
                            {participant.status.toUpperCase()} • {formatDate(participant.created_at)}
                          </div>
                          <div className="mt-2 space-y-1 text-xs text-indigo-100/72">
                            <div>{participant.name || "Anonymous"}</div>
                            <div>{participant.institution || "No institution supplied"}</div>
                            <div>{participant.discipline || "No discipline supplied"}</div>
                          </div>
                        </td>
                        <td className="px-4 py-4 text-xs text-indigo-100/74">
                          <div>{participant.profile_completed ? "Profile complete" : "Profile pending"}</div>
                          <div className="mt-2">{participant.completed_at ? `Completed ${formatDate(participant.completed_at)}` : "In progress"}</div>
                        </td>
                        <td className="space-y-3 px-4 py-4 text-xs text-indigo-100/75">
                          <div>
                            <div className="mb-1 flex justify-between">
                              <span>Block A</span>
                              <span>{participant.block_a_completed}/{participant.block_a_total}</span>
                            </div>
                            <div className="h-2 overflow-hidden rounded-full bg-indigo-200/10">
                              <div
                                className="h-full rounded-full bg-gradient-to-r from-cyan-400 to-indigo-500"
                                style={{ width: `${(participant.block_a_completed / Math.max(1, participant.block_a_total)) * 100}%` }}
                              />
                            </div>
                          </div>
                          <div>
                            <div className="mb-1 flex justify-between">
                              <span>Block B</span>
                              <span>{participant.block_b_completed}/{participant.block_b_total}</span>
                            </div>
                            <div className="h-2 overflow-hidden rounded-full bg-indigo-200/10">
                              <div
                                className="h-full rounded-full bg-gradient-to-r from-fuchsia-400 to-indigo-500"
                                style={{ width: `${(participant.block_b_completed / Math.max(1, participant.block_b_total)) * 100}%` }}
                              />
                            </div>
                          </div>
                          <div>
                            <div className="mb-1 flex justify-between">
                              <span>Block C</span>
                              <span>{participant.block_c_completed}/{participant.block_c_total}</span>
                            </div>
                            <div className="h-2 overflow-hidden rounded-full bg-indigo-200/10">
                              <div
                                className="h-full rounded-full bg-gradient-to-r from-emerald-300 to-cyan-400"
                                style={{ width: `${(participant.block_c_completed / Math.max(1, participant.block_c_total)) * 100}%` }}
                              />
                            </div>
                          </div>
                          <div className="flex flex-wrap gap-2">
                            <span className={`rounded-full px-2.5 py-1 ${participant.block_a_feedback_completed ? "bg-cyan-300/15 text-cyan-100" : "bg-indigo-200/8 text-indigo-100/55"}`}>
                              A note
                            </span>
                            <span className={`rounded-full px-2.5 py-1 ${participant.block_b_feedback_completed ? "bg-cyan-300/15 text-cyan-100" : "bg-indigo-200/8 text-indigo-100/55"}`}>
                              B note
                            </span>
                          </div>
                        </td>
                        <td className="px-4 py-4 text-sm text-indigo-100/82">{participant.attention_flags}</td>
                        <td className="px-4 py-4 text-xs text-indigo-100/74">
                          <div className="max-w-xs truncate">{participant.block_a_stage_comment || "—"}</div>
                          <div className="mt-2 max-w-xs truncate">{participant.block_b_stage_comment || "—"}</div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </CardContent>
        </Card>

        <div className="space-y-5">
          <Card>
            <CardHeader>
              <CardTitle className="text-xl">Discipline mix</CardTitle>
              <CardDescription>Who is actually participating in the study.</CardDescription>
            </CardHeader>
            <CardContent className="flex flex-wrap gap-2">
              {dashboard.discipline_breakdown.map((entry) => (
                <span
                  key={entry.discipline}
                  className="rounded-full border border-indigo-200/16 bg-indigo-200/8 px-3 py-1.5 text-sm text-indigo-100/82"
                >
                  {entry.discipline} • {entry.count}
                </span>
              ))}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-xl">Wrap-up feedback</CardTitle>
              <CardDescription>Latest stage-end domain commentary.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {dashboard.recent_stage_feedback.length === 0 ? (
                <p className="text-sm text-indigo-100/65">No stage feedback submitted yet.</p>
              ) : (
                dashboard.recent_stage_feedback.map((entry) => <CommentCard key={`${entry.participant_id}-${entry.block}-${entry.created_at}`} entry={entry} />)
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-xl">Item comments</CardTitle>
              <CardDescription>Recent qualitative comments written during scoring.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {dashboard.recent_item_comments.length === 0 ? (
                <p className="text-sm text-indigo-100/65">No item-level comments submitted yet.</p>
              ) : (
                dashboard.recent_item_comments.map((entry) => <CommentCard key={`${entry.participant_id}-${entry.block}-${entry.sample_id}-${entry.created_at}`} entry={entry} />)
              )}
            </CardContent>
          </Card>
        </div>
      </section>

      <Card>
        <CardHeader>
          <CardTitle className="text-xl">Campaign import</CardTitle>
          <CardDescription>
            Server-side import only. The pack directory must exist on the backend host or mounted storage; this does
            not upload local files from the browser.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form className="grid gap-4 md:grid-cols-2" onSubmit={handleImport}>
            <div className="space-y-2 md:col-span-2">
              <Label htmlFor="pack-dir">Pack directory on backend</Label>
              <Input
                id="pack-dir"
                value={packDir}
                onChange={(event) => setPackDir(event.target.value)}
                placeholder="/opt/render/project/src/.../Expert_Pack_v2"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="campaign-name">Campaign name</Label>
              <Input id="campaign-name" value={campaignName} onChange={(event) => setCampaignName(event.target.value)} />
            </div>
            <div className="space-y-2">
              <Label htmlFor="seed">Seed</Label>
              <Input id="seed" type="number" value={seed} onChange={(event) => setSeed(Number(event.target.value) || 0)} />
            </div>
            <div className="space-y-2 md:col-span-2">
              <Label htmlFor="stage13">Stage 13 samples directory (optional)</Label>
              <Input
                id="stage13"
                value={stage13Samples}
                onChange={(event) => setStage13Samples(event.target.value)}
                placeholder="/opt/render/project/src/.../outputs/13_model_evaluation/samples"
              />
            </div>
            <div className="md:col-span-2 flex flex-wrap gap-3">
              <Button disabled={busy || !packDir.trim() || !campaignName.trim()} type="submit">
                {busy ? "Importing…" : "Import campaign pack"}
              </Button>
              <Button type="button" variant="outline" onClick={() => { setPackDir(""); setStage13Samples(""); }}>
                Clear
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>

      {status ? (
        <div className="rounded-2xl border border-indigo-200/12 bg-indigo-950/30 px-4 py-3 text-sm text-indigo-100/78">
          {status}
        </div>
      ) : null}
    </StudyPage>
  );
}
