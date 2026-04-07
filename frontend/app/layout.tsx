import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "CrossRec — Hybrid Movie & Game Recommender",
  description: "Cross-domain hybrid recommendation portal powered by LightGCN, MF, and content embeddings",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} min-h-screen`}>
        {children}
      </body>
    </html>
  );
}
