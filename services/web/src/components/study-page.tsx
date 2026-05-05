import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";

type StudyPageProps = {
  title: string;
  description?: string;
  participantId?: string;
  progressLabel?: string;
  progressValue?: number;
  children: React.ReactNode;
};

export function StudyPage({
  title,
  description,
  participantId,
  progressLabel,
  progressValue,
  children,
}: StudyPageProps) {
  return (
    <main className="mx-auto flex min-h-screen w-full max-w-6xl items-center justify-center p-4 sm:p-6">
      <Card className="w-full">
        <CardHeader>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <CardTitle>{title}</CardTitle>
            {participantId ? (
              <span className="rounded-full border border-indigo-200/30 bg-indigo-200/10 px-3 py-1 text-xs font-medium text-indigo-100/85">
                {participantId}
              </span>
            ) : null}
          </div>
          {description ? <CardDescription>{description}</CardDescription> : null}
          {typeof progressValue === "number" ? (
            <div className="space-y-1">
              {progressLabel ? <div className="text-xs font-medium text-indigo-100/70">{progressLabel}</div> : null}
              <Progress value={Math.max(0, Math.min(100, progressValue))} />
            </div>
          ) : null}
        </CardHeader>
        <CardContent className="space-y-5">{children}</CardContent>
      </Card>
    </main>
  );
}
