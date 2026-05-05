"use client";

type InfoCalloutProps = {
  title: string;
  body: string;
};

export function InfoCallout({ title, body }: InfoCalloutProps) {
  return (
    <div
      className="rounded-xl border px-4 py-3"
      style={{
        borderColor: "rgba(103,214,255,0.22)",
        background: "rgba(103,214,255,0.06)",
      }}
    >
      <div className="flex items-start gap-3">
        <div
          className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs font-bold"
          style={{ background: "rgba(103,214,255,0.18)", color: "#67d6ff" }}
          aria-hidden="true"
        >
          i
        </div>
        <div className="space-y-1">
          <p className="text-sm font-semibold text-cyan-100">{title}</p>
          <p className="text-sm" style={{ color: "rgba(224,231,255,0.78)" }}>
            {body}
          </p>
        </div>
      </div>
    </div>
  );
}
