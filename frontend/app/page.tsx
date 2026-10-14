// Лендинг-заглушка. Полноценный лендинг будет позже.

import Link from "next/link";

export default function HomePage() {
  return (
    <main className="mx-auto flex min-h-screen max-w-2xl flex-col items-center justify-center px-4 text-center">
      <h1 className="text-4xl font-bold tracking-tight sm:text-5xl">
        Одно видео → 5 вирусных твитов
      </h1>
      <p className="mt-4 text-lg text-neutral-400">
        Вставьте ссылку на YouTube — получите 5 готовых твитов с разными углами подачи.
      </p>
      <Link
        href="/login"
        className="mt-8 rounded-lg bg-white px-6 py-3 font-medium text-neutral-950 transition hover:bg-neutral-200"
      >
        Войти
      </Link>
    </main>
  );
}
