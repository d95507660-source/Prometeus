"use client";

// Страница входа по одноразовому коду (OTP).
//
// Почему код, а не magic link: почтовые сервисы (Gmail, Outlook, корпоративные
// антивирусы) заранее «открывают» ссылки из писем для проверки — и одноразовая
// ссылка сгорает до того, как пользователь по ней кликнет. Код вводится вручную,
// поэтому такой проблемы нет.
//
// Два шага:
//   1. Ввод email → Supabase отправляет письмо с 6-значным кодом.
//   2. Ввод кода → verifyOtp создаёт сессию (cookies) → переход в кабинет.
// Новый пользователь регистрируется автоматически при первом входе.

import { useEffect, useState, type FormEvent } from "react";
import { useRouter } from "next/navigation";
import { getSupabaseBrowser } from "@/lib/supabase-browser";

// Длина кода. Совпадает с настройкой Supabase (Authentication → Email → OTP length).
const CODE_LENGTH = 6;
// Пауза перед повторной отправкой. Supabase и сам не даёт слать письма чаще раза в минуту.
const RESEND_COOLDOWN_SEC = 60;

type Step = "email" | "code";

export default function LoginPage() {
  const router = useRouter();

  const [step, setStep] = useState<Step>("email");
  const [email, setEmail] = useState("");
  const [code, setCode] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  // Сколько секунд осталось до разрешения повторной отправки
  const [cooldown, setCooldown] = useState(0);

  // Обратный отсчёт таймера повторной отправки: раз в секунду уменьшаем на 1
  useEffect(() => {
    if (cooldown <= 0) return;
    const timer = setTimeout(() => setCooldown((s) => s - 1), 1000);
    return () => clearTimeout(timer);
  }, [cooldown]);

  // Отправка письма с кодом. Используется и на шаге 1, и для «Отправить код заново».
  // Возвращает true, если письмо ушло.
  async function sendCode(targetEmail: string): Promise<boolean> {
    const supabase = getSupabaseBrowser();
    const { error: otpError } = await supabase.auth.signInWithOtp({
      email: targetEmail,
      options: { shouldCreateUser: true },
    });

    if (otpError) {
      console.error("[login] signInWithOtp error:", otpError.message);
      if (otpError.status === 429) {
        setError("Слишком много попыток. Подождите минуту и попробуйте снова.");
      } else {
        setError("Не удалось отправить код. Проверьте email и попробуйте ещё раз.");
      }
      return false;
    }

    setCooldown(RESEND_COOLDOWN_SEC);
    return true;
  }

  // Шаг 1: ввод email → «Получить код»
  async function handleEmailSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setInfo(null);

    const trimmed = email.trim().toLowerCase();
    if (!trimmed) {
      setError("Введите email");
      return;
    }

    // Вернулись «Назад» и снова отправили тот же email, пока не прошла минута:
    // код уже в почте — не шлём новое письмо (Supabase всё равно ответил бы 429).
    if (trimmed === email && cooldown > 0) {
      setCode("");
      setStep("code");
      return;
    }

    setLoading(true);
    const ok = await sendCode(trimmed);
    setLoading(false);

    if (ok) {
      setEmail(trimmed);
      setCode("");
      setStep("code");
    }
  }

  // Шаг 2: ввод кода → «Войти»
  async function handleCodeSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setInfo(null);

    if (code.length !== CODE_LENGTH) {
      setError(`Код должен состоять из ${CODE_LENGTH} цифр`);
      return;
    }

    setLoading(true);
    const supabase = getSupabaseBrowser();
    const { error: verifyError } = await supabase.auth.verifyOtp({
      email,
      token: code,
      type: "email",
    });

    if (verifyError) {
      setLoading(false);
      console.error("[login] verifyOtp error:", verifyError.message);
      if (verifyError.status === 429) {
        setError("Слишком много попыток. Подождите минуту и попробуйте снова.");
      } else {
        // Неверный код и просроченный код Supabase возвращает одинаково
        setError("Неверный или просроченный код. Проверьте письмо или запросите новый код.");
      }
      return;
    }

    // Успех: сессия сохранена в cookies. replace — чтобы кнопка «Назад» в браузере
    // не возвращала на страницу входа; refresh — чтобы серверные страницы
    // увидели новую сессию. loading не сбрасываем, пока идёт переход.
    router.replace("/dashboard");
    router.refresh();
  }

  // «Отправить код заново»
  async function handleResend() {
    if (cooldown > 0 || loading) return;
    setError(null);
    setInfo(null);

    setLoading(true);
    const ok = await sendCode(email);
    setLoading(false);

    if (ok) {
      setCode("");
      setInfo("Новый код отправлен. Предыдущий больше не действует.");
    }
  }

  // «Назад» — вернуться к вводу email (например, если ошиблись адресом)
  function handleBack() {
    setStep("email");
    setCode("");
    setError(null);
    setInfo(null);
  }

  // В поле кода оставляем только цифры (на случай вставки «123 456» или «код: 123456»)
  function handleCodeChange(value: string) {
    setCode(value.replace(/\D/g, "").slice(0, CODE_LENGTH));
  }

  return (
    <main className="mx-auto flex min-h-screen max-w-sm flex-col justify-center px-4">
      <h1 className="text-2xl font-bold">Вход по коду</h1>

      {step === "email" ? (
        <form onSubmit={handleEmailSubmit} className="mt-6 flex flex-col gap-3">
          <p className="text-sm text-neutral-400">
            Введите email — мы пришлём код для входа. Пароль не нужен.
          </p>
          <input
            type="email"
            required
            autoComplete="email"
            autoFocus
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
            {loading ? "Отправляем..." : "Получить код"}
          </button>
        </form>
      ) : (
        <form onSubmit={handleCodeSubmit} className="mt-6 flex flex-col gap-3">
          <p className="text-sm text-neutral-400">
            Введите 6-значный код из письма. Мы отправили его на{" "}
            <span className="text-neutral-200">{email}</span>.
          </p>
          <input
            type="text"
            // Цифровая клавиатура на телефоне
            inputMode="numeric"
            // iOS и Android предложат вставить код из письма/SMS
            autoComplete="one-time-code"
            autoFocus
            pattern="[0-9]*"
            maxLength={CODE_LENGTH}
            placeholder="000000"
            value={code}
            onChange={(e) => handleCodeChange(e.target.value)}
            disabled={loading}
            aria-label="Код из письма"
            className="rounded-lg border border-neutral-700 bg-neutral-900 px-4 py-3 text-center font-mono text-2xl tracking-[0.5em] outline-none placeholder:text-neutral-700 focus:border-neutral-400 disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={loading || code.length !== CODE_LENGTH}
            className="rounded-lg bg-white px-4 py-3 font-medium text-neutral-950 transition hover:bg-neutral-200 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading ? "Проверяем..." : "Войти"}
          </button>

          <div className="mt-2 flex items-center justify-between text-sm">
            <button
              type="button"
              onClick={handleBack}
              disabled={loading}
              className="text-neutral-400 transition hover:text-neutral-200 disabled:opacity-50"
            >
              ← Назад
            </button>
            <button
              type="button"
              onClick={handleResend}
              disabled={loading || cooldown > 0}
              className="text-neutral-400 transition hover:text-neutral-200 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {cooldown > 0 ? `Отправить код заново (${cooldown} с)` : "Отправить код заново"}
            </button>
          </div>
        </form>
      )}

      {info && <p className="mt-4 text-sm text-emerald-400">{info}</p>}
      {error && <p className="mt-4 text-sm text-red-400">{error}</p>}
    </main>
  );
}
