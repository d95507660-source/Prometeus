// Индикатор статуса задачи: спиннер для «в очереди/обработка», значок для «готово/ошибка».

import type { JobStatus as JobStatusType } from "@/lib/types";

type Props = {
  status: JobStatusType;
};

const CONTENT: Record<JobStatusType, { text: string; className: string }> = {
  pending: { text: "В очереди...", className: "text-neutral-300" },
  processing: {
    text: "Обрабатываем... (обычно 30-60 секунд)",
    className: "text-amber-300",
  },
  done: { text: "Готово", className: "text-emerald-400" },
  error: { text: "Ошибка", className: "text-red-400" },
};

function Spinner() {
  return (
    <span
      aria-hidden
      className="inline-block h-5 w-5 animate-spin rounded-full border-2 border-neutral-600 border-t-neutral-100"
    />
  );
}

export default function JobStatus({ status }: Props) {
  const { text, className } = CONTENT[status];
  const inProgress = status === "pending" || status === "processing";

  return (
    // role="status" — экранные дикторы озвучат смену статуса
    <div role="status" className={`flex items-center gap-3 ${className}`}>
      {inProgress && <Spinner />}
      {status === "done" && <span aria-hidden>✓</span>}
      {status === "error" && <span aria-hidden>✕</span>}
      <span className="font-medium">{text}</span>
    </div>
  );
}
