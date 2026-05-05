import { ProgressInfo } from "@/lib/api";

export type WizardStep =
  | "welcome"
  | "consent"
  | "profile"
  | "block-a"
  | "block-a-feedback"
  | "block-b"
  | "block-c"
  | "block-b-feedback"
  | "block-c-feedback"
  | "complete";

export type WizardStepDefinition = {
  id: WizardStep;
  route: string;
  labelKey: string;
};

export const WIZARD_STEPS: WizardStepDefinition[] = [
  { id: "welcome", route: "/", labelKey: "steps.welcome" },
  { id: "consent", route: "/consent", labelKey: "steps.consent" },
  { id: "profile", route: "/profile", labelKey: "steps.profile" },
  { id: "block-a", route: "/block-a", labelKey: "steps.blockA" },
  { id: "block-a-feedback", route: "/block-a-feedback", labelKey: "steps.blockAFeedback" },
  { id: "block-b", route: "/block-b", labelKey: "steps.blockB" },
  { id: "block-c", route: "/block-c", labelKey: "steps.blockC" },
  { id: "block-b-feedback", route: "/block-b-feedback", labelKey: "steps.blockBFeedback" },
  { id: "block-c-feedback", route: "/block-c-feedback", labelKey: "steps.blockCFeedback" },
  { id: "complete", route: "/complete", labelKey: "steps.complete" },
];

export function nextRouteFromProgress(
  progress: ProgressInfo,
): "/consent" | "/profile" | "/block-a" | "/block-a-feedback" | "/block-b" | "/block-b-feedback" | "/block-c" | "/block-c-feedback" | "/complete" {
  const hasBlockA = progress.block_a_total > 0;
  const hasBlockB = progress.block_b_total > 0;
  const hasBlockC = progress.block_c_total > 0;
  if (progress.is_complete) {
    return "/complete";
  }
  // Profile is a hard gate. If it is incomplete, always route there.
  if (!progress.profile_completed) {
    return "/profile";
  }
  if (hasBlockA && progress.block_a_completed < progress.block_a_total) {
    return "/block-a";
  }
  if (hasBlockA && !progress.block_a_feedback_completed) {
    return "/block-a-feedback";
  }
  if (hasBlockB && progress.block_b_completed < progress.block_b_total) {
    return "/block-b";
  }
  if (hasBlockB && !progress.block_b_feedback_completed) {
    return "/block-b-feedback";
  }
  if (hasBlockC && progress.block_c_completed < progress.block_c_total) {
    return "/block-c";
  }
  if (hasBlockC && !progress.block_c_feedback_completed) {
    return "/block-c-feedback";
  }
  return "/complete";
}

export function stepIndex(step: WizardStep): number {
  return WIZARD_STEPS.findIndex((candidate) => candidate.id === step);
}

export function isStepUnlocked(current: WizardStep, candidate: WizardStep): boolean {
  return stepIndex(candidate) <= stepIndex(current);
}

export function mapBlockBChoiceKey(key: string): "A" | "B" | "Tie" | "Unsure" | null {
  const lower = key.toLowerCase();
  if (lower === "a") return "A";
  if (lower === "b") return "B";
  if (lower === "t") return "Tie";
  if (lower === "u") return "Unsure";
  return null;
}
