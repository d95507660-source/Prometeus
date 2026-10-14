"use client";

// Форма создания задачи: ссылка на YouTube → POST /api/jobs → переход на страницу результата.

import { useState, type FormEvent } from "react";
import { useRouter } from "next/navigation";

type Props = {
  // true — лимит исчерпан, форма неактивна
  disabled?: boolean;
};

export default function JobForm({ disabled = false }: Props) {
  const router = useRouter();
  const [url, setUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);

    // Простая проверка в браузере. Полная — на сервере и в бэкенде.
    const trimmed = url.trim();
    if (!trimmed) {
      setError("Вставьте ссылку на YouTube-видео");
      return;
    }
    if (!trimmed.startsWith("http")) {
      setError("Ссылка должна начинаться с http:// или https://");
      return;
    }

    setLoading(true);
    try {
      const response = await fetch("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: trimmed }),
      });
      const data = await response.json().catch(() => ({}));

      if (response.status === 429) {
        // Заглушка: оплата появится позже
        setError("Лимит исчерпан, оформите подписку");
        return;
      }
      if (response.status === 401) {
        // Сессия истекла — на страницу входа
        router.push("/login");
        return;
      }
      if (!response.ok || !data.job_id) {
        setError(data.error ?? "Не удалось создать задачу. Попробуйте ещё раз.");
        return;
      }

      // Успех — переходим на страницу результата. loading не сбрасываем,
      // чтобы кнопку нельзя было нажать повторно, пока идёт переход.
      router.push(`/jobs/${data.job_id}`);
      return;
    } catch {
      setError("Нет связи с сервером. Проверьте интернет и попробуйте ещё раз.");
    }
    setLoading(false);
  }

  const isDisabled = disabled || loading;

  return (
    <form onSubmit={handleSubmit}>
      <div className="flex flex-col gap-3 sm:flex-row">
        <input
          type="url"
          inputMode="url"
          placeholder="https://www.youtube.com/watch?v=..."
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          disabled={isDisabled}
          className="min-w-0 flex-1 rounded-lg border border-neutral-700 bg-neutral-900 px-4 py-3 outline-none placeholder:text-neutral-500 focus:border-neutral-400 disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={isDisabled}
          className="rounded-lg bg-white px-6 py-3 font-medium text-neutral-950 transition hover:bg-neutral-200 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading ? "Отправляем..." : "Обработать"}
        </button>
      </div>
      {error && <p className="mt-3 text-sm text-red-400">{error}</p>}
    </form>
  );
}
