const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000/api/v1";
const API_ROOT = API_BASE_URL.replace(/\/api\/v1\/?$/, "");

type RequestOptions = RequestInit & {
  token?: string;
};

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function apiFetch<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const headers = new Headers(options.headers || {});
  if (!headers.has("Content-Type") && options.body) {
    headers.set("Content-Type", "application/json");
  }
  if (options.token) {
    headers.set("x-admin-secret", options.token);
  }

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers,
    credentials: "include",
    cache: "no-store",
  });

  if (!response.ok) {
    let detail = "Request failed";
    try {
      const payload = await response.json();
      detail = payload.detail || payload.message || detail;
    } catch {
      detail = response.statusText || detail;
    }
    throw new ApiError(detail, response.status);
  }

  const contentType = response.headers.get("content-type") || "";
  if (!contentType.includes("application/json")) {
    return (await response.text()) as T;
  }
  return (await response.json()) as T;
}

export function resolveAssetUrl(path: string): string {
  if (!path.startsWith("/")) {
    return path;
  }
  return `${API_ROOT}${path}`;
}

export type SessionInfo = {
  participant_id: string;
  status: string;
  profile_completed: boolean;
  block_b_comprehension_attempts: number;
  block_b_comprehension_passed: boolean;
  comprehension_risk: boolean;
  campaign: CampaignInfo;
  created_at: string;
  completed_at: string | null;
};

export type CampaignInfo = {
  id: number;
  name: string;
  seed: number;
  protocol_version: string;
  block_a_target_count: number;
  block_b_target_count: number;
  block_c_target_count: number;
};

export type BlockBComprehensionResponse = {
  passed: boolean;
  attempts: number;
  max_attempts: number;
  comprehension_risk: boolean;
  protocol_version: string;
};

export type ProgressInfo = {
  block_a_completed: number;
  block_a_total: number;
  block_b_completed: number;
  block_b_total: number;
  block_c_completed: number;
  block_c_total: number;
  profile_completed: boolean;
  block_a_feedback_completed: boolean;
  block_b_feedback_completed: boolean;
  block_c_feedback_completed: boolean;
  is_complete: boolean;
};

export type SessionProfile = {
  name: string | null;
  institution: string | null;
  discipline: string | null;
  discipline_other: string | null;
  profile_completed: boolean;
};

export type StageFeedback = {
  block: "A" | "B" | "C";
  comment: string | null;
  completed: boolean;
};

export type AdminSession = {
  authenticated: boolean;
  auth_mode: string;
  campaign_id: number | null;
  campaign_name: string | null;
};

export type AdminStats = {
  participants_total: number;
  participants_completed: number;
  participants_active: number;
  profiles_completed: number;
  block_a_feedback_completed: number;
  block_b_feedback_completed: number;
  block_c_feedback_completed: number;
  block_a_responses: number;
  block_b_responses: number;
  block_c_responses: number;
  attention_flags_total: number;
  comprehension_risk_total: number;
};

export type AdminDisciplineBreakdown = {
  discipline: string;
  count: number;
};

export type AdminParticipantRow = {
  participant_id: string;
  status: string;
  name: string | null;
  institution: string | null;
  discipline: string | null;
  profile_completed: boolean;
  block_a_completed: number;
  block_a_total: number;
  block_b_completed: number;
  block_b_total: number;
  block_c_completed: number;
  block_c_total: number;
  block_a_feedback_completed: boolean;
  block_b_feedback_completed: boolean;
  block_c_feedback_completed: boolean;
  block_b_comprehension_attempts: number;
  block_b_comprehension_passed: boolean;
  comprehension_risk: boolean;
  attention_flags: number;
  block_a_stage_comment: string | null;
  block_b_stage_comment: string | null;
  block_c_stage_comment: string | null;
  created_at: string;
  completed_at: string | null;
};

export type AdminCommentRow = {
  participant_id: string;
  block: string;
  source: string;
  comment: string;
  sample_id: string | null;
  created_at: string;
};

export type AdminDashboard = {
  campaign: CampaignInfo;
  stats: AdminStats;
  discipline_breakdown: AdminDisciplineBreakdown[];
  participants: AdminParticipantRow[];
  recent_stage_feedback: AdminCommentRow[];
  recent_item_comments: AdminCommentRow[];
};

export type BlockAItem = {
  assignment_id: number;
  item_order: number;
  total_items: number;
  sample_id: string;
  image_url: string;
  mask_type: string | null;
  mask_coverage_bin: string | null;
  source_label: string;
  is_attention_check: boolean;
};

export type BlockBItem = {
  assignment_id: number;
  item_order: number;
  total_items: number;
  sample_id: string;
  input_url: string;
  option_a_url: string;
  option_b_url: string;
  show_a_left: boolean;
  mask_type: string | null;
  mask_coverage_bin: string | null;
  is_practice: boolean;
  is_anchor: boolean;
  is_attention_check: boolean;
};

export type BlockCItem = {
  assignment_id: number;
  item_order: number;
  total_items: number;
  sample_id: string;
  input_url: string;
  option_a_url: string;
  option_b_url: string;
  option_c_url: string;
  option_d_url: string;
  show_a_left: boolean;
  mask_type: string | null;
  mask_coverage_bin: string | null;
  is_anchor: boolean;
  is_attention_check: boolean;
};

export type NextPayload<T> = {
  done: boolean;
  item: T | null;
};

export type InviteResponse = {
  participant_public_id: string;
  campaign: CampaignInfo;
  progress: ProgressInfo;
};

function toNumber(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function toBoolean(value: unknown, fallback = false): boolean {
  return typeof value === "boolean" ? value : fallback;
}

function normalizeProgress(raw: unknown): ProgressInfo {
  const source = (raw ?? {}) as Record<string, unknown>;
  return {
    block_a_completed: toNumber(source.block_a_completed),
    block_a_total: toNumber(source.block_a_total),
    block_b_completed: toNumber(source.block_b_completed),
    block_b_total: toNumber(source.block_b_total),
    block_c_completed: toNumber(source.block_c_completed),
    block_c_total: toNumber(source.block_c_total),
    profile_completed: toBoolean(source.profile_completed),
    block_a_feedback_completed: toBoolean(source.block_a_feedback_completed),
    block_b_feedback_completed: toBoolean(source.block_b_feedback_completed),
    block_c_feedback_completed: toBoolean(source.block_c_feedback_completed),
    is_complete: toBoolean(source.is_complete),
  };
}

export async function invite(inviteCode: string) {
  const response = await apiFetch<InviteResponse>("/auth/invite", {
    method: "POST",
    body: JSON.stringify({ invite_code: inviteCode }),
  });
  return {
    ...response,
    progress: normalizeProgress(response.progress),
  };
}

export async function login(username: string, password: string) {
  return apiFetch<{ participant_public_id: string; campaign: CampaignInfo }>("/auth/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
}

export async function logout() {
  return apiFetch<{ message: string }>("/auth/logout", { method: "POST" });
}

export async function getSession() {
  return apiFetch<SessionInfo>("/session/me");
}

export async function submitBlockBComprehension(selectedOption: string) {
  return apiFetch<BlockBComprehensionResponse>("/session/block-b-comprehension", {
    method: "POST",
    body: JSON.stringify({ selected_option: selectedOption }),
  });
}

export async function getActiveCampaign() {
  return apiFetch<CampaignInfo>("/campaign/active");
}

export async function getProgress() {
  const response = await apiFetch<ProgressInfo>("/progress");
  return normalizeProgress(response);
}

export async function getSessionProfile() {
  return apiFetch<SessionProfile>("/session/profile");
}

export async function updateSessionProfile(payload: {
  name?: string;
  institution?: string;
  discipline?: string;
  discipline_other?: string;
}) {
  return apiFetch<SessionProfile>("/session/profile", {
    method: "PUT",
    body: JSON.stringify(payload),
  });
}

export async function getStageFeedback(block: "A" | "B" | "C") {
  return apiFetch<StageFeedback>(`/session/feedback/${block}`);
}

export async function submitStageFeedback(block: "A" | "B" | "C", comment: string) {
  return apiFetch<StageFeedback>(`/session/feedback/${block}`, {
    method: "PUT",
    body: JSON.stringify({ comment }),
  });
}

export async function completeSession() {
  const response = await apiFetch<ProgressInfo>("/session/complete", { method: "POST" });
  return normalizeProgress(response);
}

export async function getBlockANext() {
  return apiFetch<NextPayload<BlockAItem>>("/block-a/next");
}

export async function submitBlockA(payload: {
  assignment_id: number;
  authenticity_likelihood: number;
  archaeological_plausibility: number;
  confidence: number;
  comment: string;
  response_time_ms: number;
}) {
  return apiFetch<NextPayload<BlockAItem>>("/block-a/submit", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function getBlockBNext() {
  return apiFetch<NextPayload<BlockBItem>>("/block-b/next");
}

export async function submitBlockB(payload: {
  assignment_id: number;
  choice: "A" | "B" | "Tie" | "Unsure";
  confidence: number;
  comment: string;
  response_time_ms: number;
}) {
  return apiFetch<NextPayload<BlockBItem>>("/block-b/submit", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function getBlockCNext() {
  return apiFetch<NextPayload<BlockCItem>>("/block-c/next");
}

export async function submitBlockC(payload: {
  assignment_id: number;
  choice: "A" | "B" | "C" | "D";
  confidence: number;
  comment: string;
  response_time_ms: number;
}) {
  return apiFetch<NextPayload<BlockCItem>>("/block-c/submit", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function adminImportPack(
  token: string | undefined,
  payload: {
    pack_dir: string;
    campaign_name: string;
    seed: number;
    stage13_samples?: string;
    activate?: boolean;
    disjoint_blocks?: boolean;
  }
) {
  return apiFetch<{ campaign_id: number; campaign_name: string; is_active: boolean }>("/admin/import-pack", {
    method: "POST",
    body: JSON.stringify(payload),
    token,
  });
}

export async function adminExportJson(token?: string) {
  return apiFetch<Record<string, unknown>>("/admin/export/responses.json", { token });
}

export async function adminQualityReport(token?: string) {
  return apiFetch<Record<string, unknown>>("/admin/export/quality_report.json", { token });
}

export async function adminExportCsv(token?: string): Promise<string> {
  return apiFetch<string>("/admin/export/responses.csv", { token });
}

export async function adminLogin(password: string) {
  return apiFetch<AdminSession>("/admin/auth/login", {
    method: "POST",
    body: JSON.stringify({ password }),
  });
}

export async function adminLogout() {
  return apiFetch<{ message: string }>("/admin/auth/logout", {
    method: "POST",
  });
}

export async function getAdminSession() {
  return apiFetch<AdminSession>("/admin/session");
}

export async function getAdminDashboard() {
  return apiFetch<AdminDashboard>("/admin/dashboard");
}

export function downloadText(content: string, filename: string, mimeType = "text/plain") {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}
