"use client";

// Страница входа: пользователь вводит email, Supabase присылает ссылку (magic link).
// Пароли не нужны. Новый пользователь регистрируется автоматически при первом входе.

import { useEffect, useState, type FormEvent } from "react";
import { getSupabaseBrowser } from "@/lib/supabase-browser";

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);
  const [sent, setSent] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Если пришли сюда из /auth/callback с ошибкой — показываем пояснение.
  // Читаем window.location, а не useSearchParams: так страница не требует Suspense.
  useEffect(() => {
    if (new URLSearchParams(window.location.search).get("error") === "link") {
      setError(
        "Ссылка для входа устарела или открыта в другом браузере. Запросите новую.",
      );
    }
  }, []);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);

    const trimmed = email.trim();
    if (!trimmed) {
      setError("Введите email");
      return;
    }

    setLoading(true);
    const supabase = getSupabaseBrowser();
    const { error: otpError } = await supabase.auth.signInWithOtp({
      email: trimmed,
      options: {
        // Куда вернуть пользователя после клика по ссылке в письме.
        // Этот адрес нужно добавить в Supabase: Authentication → URL Configuration → Redirect URLs.
        emailRedirectTo: `${window.location.origin}/auth/callback`,
      },
    });
    setLoading(false);

    if (otpError) {
      // Самая частая ошибка — слишком частые запросы (Supabase ограничивает письма)
      if (otpError.status === 429) {
        setError("Слишком много попыток. Подождите минуту и попробуйте снова.");
      } else {
        setError("Не удалось отправить письмо. Проверьте email и попробуйте ещё раз.");
      }
      console.error("[login] signInWithOtp error:", otpError.message);
      return;
    }

    setSent(true);
  }

  return (
    <main className="mx-auto flex min-h-screen max-w-sm flex-col justify-center px-4">
      <h1 className="text-2xl font-bold">Вход</h1>

      {sent ? (
        <div className="mt-6 rounded-lg border border-neutral-800 bg-neutral-900 p-4">
          <p className="font-medium">Проверьте почту</p>
          <p className="mt-1 text-sm text-neutral-400">
            Мы отправили ссылку для входа на <span className="text-neutral-200">{email}</span>.
            Откройте её в этом же браузере.
          </p>
        </div>
      ) : (
        <form onSubmit={handleSubmit} className="mt-6 flex flex-col gap-3">
          <input
            type="email"
            required
            autoComplete="email"
            placeholder="you@example.com"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            disabled={loading}
            className="rounded-lg border border-neutral-700 bg-neutral-900 px-4 py-3 outline-none placeholder:text-neutral-500 focus:border-neutral-400 disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={loading}
            className="rounded-lg bg-white px-4 py-3 font-medium text-neutral-950 transition hover:bg-neutral-200 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading ? "Отправляем..." : "Отправить ссылку для входа"}
          </button>
        </form>
      )}

      {error && <p className="mt-4 text-sm text-red-400">{error}</p>}
    </main>
  );
}
