"use client";

// Страница результата. Каждые 3 секунды спрашивает GET /api/jobs/[id],
// пока задача не станет 'done' или 'error'.

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import JobStatus from "@/components/JobStatus";
import TweetCard from "@/components/TweetCard";
import type { JobStatusResponse } from "@/lib/types";

const POLL_INTERVAL_MS = 3000;

export default function JobPage({ params }: { params: { id: string } }) {
  const router = useRouter();
  const [job, setJob] = useState<JobStatusResponse | null>(null);
  const [notFound, setNotFound] = useState(false);
  // Временная проблема со связью — показываем подсказку, но продолжаем опрос
  const [connectionIssue, setConnectionIssue] = useState(false);

  useEffect(() => {
    // stopped — флаг остановки: страницу закрыли или задача завершилась
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    // Решение: setTimeout после ответа, а не setInterval. Так следующий запрос
    // уходит только когда пришёл предыдущий — запросы не накапливаются
    // при медленном интернете.
    async function poll() {
      try {
        const response = await fetch(`/api/jobs/${params.id}`, { cache: "no-store" });
        if (stopped) return;

        if (response.status === 401) {
          router.push("/login");
          return;
        }
        if (response.status === 404) {
          setNotFound(true);
          return; // опрос прекращаем
        }
        if (response.ok) {
          const data: JobStatusResponse = await response.json();
          if (stopped) return;
          setJob(data);
          setConnectionIssue(false);
          if (data.status === "done" || data.status === "error") {
            return; // финальный статус — опрос прекращаем
          }
        } else {
          setConnectionIssue(true);
        }
      } catch {
        if (stopped) return;
        setConnectionIssue(true);
      }

      timer = setTimeout(poll, POLL_INTERVAL_MS);
    }

    poll();

    // При уходе со страницы останавливаем опрос
    return () => {
      stopped = true;
      if (timer) clearTimeout(timer);
    };
  }, [params.id, router]);

  return (
    <main className="mx-auto max-w-2xl px-4 py-12">
      <Link href="/dashboard" className="text-sm text-neutral-400 hover:text-neutral-200">
        ← В кабинет
      </Link>

      <h1 className="mt-4 text-2xl font-bold">Результат</h1>

      <div className="mt-6">
        {notFound ? (
          <ErrorBlock message="Задача не найдена." />
        ) : !job ? (
          <p className="text-neutral-400">Загружаем...</p>
        ) : job.status === "error" ? (
          <>
            <JobStatus status="error" />
            <ErrorBlock message={job.error_message ?? "Что-то пошло не так. Попробуйте ещё раз."} />
          </>
        ) : job.status === "done" ? (
          <>
            <JobStatus status="done" />
            <div className="mt-6 flex flex-col gap-4">
              {(job.output_tweets ?? []).map((tweet, i) => (
                <TweetCard key={i} tweet={tweet} index={i + 1} />
              ))}
            </div>
          </>
        ) : (
          <JobStatus status={job.status} />
        )}

        {connectionIssue && !notFound && (
          <p className="mt-4 text-sm text-neutral-500">
            Проблема со связью — продолжаем попытки...
          </p>
        )}
      </div>
    </main>
  );
}

// Блок ошибки с кнопкой возврата в кабинет
function ErrorBlock({ message }: { message: string }) {
  return (
    <div className="mt-4 rounded-lg border border-red-900 bg-red-950/40 p-4">
      <p className="text-red-300">{message}</p>
      <Link
        href="/dashboard"
        className="mt-4 inline-block rounded-lg bg-white px-4 py-2 text-sm font-medium text-neutral-950 transition hover:bg-neutral-200"
      >
        На главную
      </Link>
    </div>
  );
}
