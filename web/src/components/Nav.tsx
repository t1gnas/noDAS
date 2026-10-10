"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { AudioWaveform } from "lucide-react";

const TABS = [
  { href: "/", label: "Live" },
  { href: "/results", label: "Results" },
];

export function Nav() {
  const path = usePathname();
  return (
    <header className="glass sticky top-0 z-30 border-b border-line">
      <div className="max-w-[1280px] mx-auto px-4 sm:px-8 h-14 flex items-center justify-between gap-4">
        <Link href="/" className="flex items-center gap-2.5">
          <span className="grid place-items-center size-8 rounded-[10px] bg-gradient-to-br from-[#0a84ff] to-[#5e5ce6] text-white shadow-sm">
            <AudioWaveform size={18} strokeWidth={2.4} />
          </span>
          <span className="font-semibold tracking-tight text-[17px]">noDAS</span>
        </Link>
        <nav className="flex p-1 rounded-full bg-black/[0.05]">
          {TABS.map((t) => {
            const active = t.href === "/" ? path === "/" : path.startsWith(t.href);
            return (
              <Link
                key={t.href}
                href={t.href}
                className={`px-4 sm:px-5 py-1.5 rounded-full text-[13px] font-medium transition-all ${
                  active ? "bg-white shadow-sm text-ink" : "text-ink-2 hover:text-ink"
                }`}
              >
                {t.label}
              </Link>
            );
          })}
        </nav>
        <span className="hidden sm:flex items-center gap-2 text-[12px] text-ink-2">
          <span className="size-2 rounded-full bg-green" />
          CNN on GPU
        </span>
      </div>
    </header>
  );
}
