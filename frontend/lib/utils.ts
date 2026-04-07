import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/** Merge Tailwind classes safely. */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** Format a date string to a readable locale format. */
export function formatDate(dateStr: string | null | undefined): string {
  if (!dateStr) return "N/A";
  return new Date(dateStr).toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

/** Truncate a string to a max length with ellipsis. */
export function truncate(str: string | null | undefined, maxLen: number = 100): string {
  if (!str) return "";
  return str.length > maxLen ? str.slice(0, maxLen) + "…" : str;
}

/** Get domain label and color. */
export function getDomainInfo(domain: string): { label: string; color: string; bgColor: string } {
  if (domain === "movie") {
    return { label: "Movie", color: "text-blue-600", bgColor: "bg-blue-100" };
  }
  return { label: "Game", color: "text-purple-600", bgColor: "bg-purple-100" };
}
