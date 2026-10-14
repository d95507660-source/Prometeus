// Личный кабинет (серверный компонент): приветствие, счётчик, форма, последние задачи.
// Данные читаются на сервере от имени пользователя — RLS отдаёт только его строки.

import Link from "next/link";
import { redirect } from "next/navigation";
import JobForm from "@/components/JobForm";
import { getCurrentUser, getSupabaseServer } from "@/lib/supabase-server";
import type { Job, JobStatus } from "@/lib/types";

// Лимит бесплатного тарифа (такой же, как в RPC create_job)
const FREE_LIMIT = 3;

// Страница всегда строится заново — счётчик и список задач должны быть свежими
export const dynamic = "force-dynamic";

const STATUS_LABELS: Record<JobStatus, { text: string; className: string }> = {
  pending: { text: "В очереди", className: "text-neutral-400" },
  processing: { text: "Обработка", className: "text-amber-400" },
  done: { text: "Готово", className: "text-emerald-400" },
  error: { text: "Ошибка", className: "text-red-400" },
};

export default async function DashboardPage() {
  const user = await getCurrentUser();
  if (!user) {
    redirect("/login");
  }

  const supabase = getSupabaseServer();

  // Профиль и последние задачи грузим параллельно
  const [profileResult, jobsResult] = await Promise.all([
    supabase.from("users").select("usage_count, plan").eq("id", user.id).maybeSingle(),
    supabase
      .from("jobs")
      .select("id, url, status, created_at")
      .order("created_at", { ascending: false })
      .limit(5),
  ]);

  const usageCount = profileResult.data?.usage_count ?? 0;
  const plan = profileResult.data?.plan ?? "free";
  const jobs = (jobsResult.data ?? []) as Pick<Job, "id" | "url" | "status" | "created_at">[];
  const limitReached = plan === "free" && usageCount >= FREE_LIMIT;

  return (
    <main className="mx-auto max-w-2xl px-4 py-12">
      <h1 className="text-2xl font-bold">Привет, {user.email}</h1>

      <p className="mt-2 text-neutral-400">
        {plan === "free" ? (
          <>
            Использовано переработок:{" "}
            <span className="text-neutral-100">
              {usageCount} из {FREE_LIMIT}
            </span>
          </>
        ) : (
          "Тариф Pro: без ограничений"
        )}
      </p>

      <section className="mt-8">
        <JobForm disabled={limitReached} />
        {limitReached && (
          <p className="mt-3 text-sm text-amber-400">
            Бесплатные переработки закончились. Подписка появится совсем скоро.
          </p>
        )}
      </section>

      <section className="mt-12">
        <h2 className="text-lg font-semibold">Последние задачи</h2>

        {jobs.length === 0 ? (
          <p className="mt-3 text-neutral-500">Пока пусто. Вставьте первую ссылку выше.</p>
        ) : (
          <ul className="mt-3 divide-y divide-neutral-800 rounded-lg border border-neutral-800">
            {jobs.map((job) => {
              const label = STATUS_LABELS[job.status];
              return (
                <li key={job.id}>
                  <Link
                    href={`/jobs/${job.id}`}
                    className="flex items-center justify-between gap-4 px-4 py-3 transition hover:bg-neutral-900"
                  >
                    <span className="min-w-0 truncate text-sm text-neutral-300">{job.url}</span>
                    <span className={`shrink-0 text-sm ${label.className}`}>{label.text}</span>
                  </Link>
                </li>
              );
            })}
          </ul>
        )}
      </section>
    </main>
  );
}
