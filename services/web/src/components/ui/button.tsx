import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex min-h-11 items-center justify-center whitespace-nowrap rounded-xl px-4 text-sm font-semibold transition duration-200 disabled:pointer-events-none disabled:opacity-50 focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-blue-300/30",
  {
    variants: {
      variant: {
        default:
          "bg-gradient-to-r from-cyan-400 to-indigo-500 text-slate-950 shadow-[0_8px_28px_-16px_rgba(103,214,255,0.8)] hover:from-cyan-300 hover:to-indigo-400 active:scale-[0.99]",
        outline:
          "border border-indigo-200/35 bg-indigo-200/10 text-indigo-100 backdrop-blur hover:border-indigo-200/60 hover:bg-indigo-200/20 active:scale-[0.99]",
        ghost: "text-indigo-100 hover:bg-indigo-200/10 active:scale-[0.99]",
      },
      size: {
        default: "h-11 py-2.5",
        sm: "h-10 rounded-md px-3",
        lg: "h-12 rounded-xl px-8 text-base",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, ...props }, ref) => {
    return <button className={cn(buttonVariants({ variant, size, className }))} ref={ref} {...props} />;
  }
);
Button.displayName = "Button";

export { Button, buttonVariants };
