import * as React from "react";

import { cn } from "@/lib/utils";

function Input({ className, type, ...props }: React.ComponentProps<"input">) {
  return (
    <input
      type={type}
      className={cn(
        "min-h-11 w-full rounded-xl border border-indigo-200/30 bg-indigo-950/50 px-3 py-2 text-sm text-indigo-50 outline-none transition placeholder:text-indigo-100/40",
        "focus:border-cyan-300/70 focus:ring-4 focus:ring-cyan-300/20",
        "disabled:cursor-not-allowed disabled:opacity-60",
        className
      )}
      {...props}
    />
  );
}

export { Input };
