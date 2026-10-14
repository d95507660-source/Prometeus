// Корневой layout: общая обёртка для всех страниц.

import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Content Repurposing Engine",
  description: "Одно YouTube-видео → 5 готовых твитов",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ru">
      {/* Тёмная тема для всего сайта */}
      <body className="min-h-screen bg-neutral-950 text-neutral-100 antialiased">{children}</body>
    </html>
  );
}
