"use client";

// Карточка одного твита: hook жирным, body, cta серым, кнопка «Скопировать».

import { useEffect, useRef, useState } from "react";
import type { Tweet } from "@/lib/types";

type Props = {
  tweet: Tweet;
  index: number; // номер твита (1-5) для подписи
};

// Подписи углов подачи — в том же порядке, в каком их генерирует промпт
const ANGLES = ["Провокация", "История", "Факт", "Инструкция", "Миф"];

export default function TweetCard({ tweet, index }: Props) {
  const [copied, setCopied] = useState(false);
  const [copyError, setCopyError] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Убираем таймер, если карточка исчезла раньше, чем прошли 2 секунды
  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(tweet.full_text);
      setCopied(true);
      setCopyError(false);
      if (timerRef.current) clearTimeout(timerRef.current);
      timerRef.current = setTimeout(() => setCopied(false), 2000);
    } catch {
      // Буфер обмена недоступен (старый браузер или нет разрешения)
      setCopyError(true);
    }
  }

  return (
    <article className="rounded-xl border border-neutral-800 bg-neutral-900 p-5">
      <div className="mb-3 flex items-center justify-between text-xs text-neutral-500">
        <span>
          #{index} · {ANGLES[index - 1] ?? "Твит"}
        </span>
        <span>{tweet.full_text.length}/280</span>
      </div>

      <p className="font-semibold">{tweet.hook}</p>
      <p className="mt-2 whitespace-pre-line text-neutral-200">{tweet.body}</p>
      <p className="mt-2 text-neutral-400">{tweet.cta}</p>

      <div className="mt-4 flex items-center gap-3">
        <button
          type="button"
          onClick={handleCopy}
          className="rounded-lg border border-neutral-700 px-4 py-2 text-sm transition hover:bg-neutral-800"
        >
          {copied ? "Скопировано" : "Скопировать"}
        </button>
        {copyError && (
          <span className="text-xs text-red-400">Не удалось скопировать — выделите текст вручную</span>
        )}
      </div>
    </article>
  );
}
