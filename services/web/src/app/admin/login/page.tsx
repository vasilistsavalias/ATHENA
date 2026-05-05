"use client";

import { FormEvent, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { LockKeyhole, ShieldCheck } from "lucide-react";

import { ApiError, adminLogin, getAdminSession } from "@/lib/api";
import { StudyPage } from "@/components/study-page";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export default function AdminLoginPage() {
  const router = useRouter();
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function bootstrap() {
      try {
        await getAdminSession();
        router.replace("/admin");
      } catch {
        return;
      }
    }
    void bootstrap();
  }, [router]);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await adminLogin(password);
      router.replace("/admin");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Admin sign-in failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <StudyPage
      title="ATHENA Admin Access"
      description="Separate admin authentication protects exports, campaign operations, and participant analytics."
    >
      <div className="grid gap-5 lg:grid-cols-[1.15fr_0.85fr]">
        <section className="rounded-3xl border border-cyan-300/15 bg-[radial-gradient(circle_at_top_left,rgba(103,214,255,0.12),transparent_38%),linear-gradient(160deg,rgba(18,28,56,0.92),rgba(11,18,36,0.96))] p-6">
          <div className="mb-5 flex items-center gap-3">
            <div className="rounded-2xl border border-cyan-300/20 bg-cyan-300/10 p-3 text-cyan-200">
              <ShieldCheck className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-indigo-50">Secure admin boundary</h2>
              <p className="text-sm text-indigo-100/70">
                Participant invite codes never grant admin access. Export and campaign operations stay behind a
                dedicated admin login.
              </p>
            </div>
          </div>
          <ul className="space-y-3 text-sm text-indigo-100/78">
            <li>Inspect participant progress, profile completion, stage comments, and attention flags.</li>
            <li>Download CSV, JSON, and quality-report exports without touching Render or Vercel consoles.</li>
            <li>Trigger server-side campaign imports when the pack already exists on the backend host.</li>
          </ul>
        </section>

        <form
          className="rounded-3xl border border-indigo-200/20 bg-indigo-950/50 p-6 shadow-[0_28px_80px_-42px_rgba(3,8,24,0.88)]"
          onSubmit={handleSubmit}
        >
          <div className="mb-6 flex items-center gap-3">
            <div className="rounded-2xl border border-indigo-200/20 bg-indigo-200/10 p-3 text-indigo-100">
              <LockKeyhole className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-indigo-50">Admin login</h2>
              <p className="text-sm text-indigo-100/70">Use the dedicated admin password configured on the backend.</p>
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="admin-password">Admin password</Label>
            <Input
              id="admin-password"
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              placeholder="Enter admin password"
              autoComplete="current-password"
            />
          </div>

          {error ? (
            <div className="mt-4 rounded-2xl border border-rose-400/30 bg-rose-400/10 px-4 py-3 text-sm text-rose-100">
              {error}
            </div>
          ) : null}

          <Button className="mt-6 w-full" disabled={busy || !password.trim()} type="submit">
            {busy ? "Signing in..." : "Open admin dashboard"}
          </Button>
        </form>
      </div>
    </StudyPage>
  );
}
