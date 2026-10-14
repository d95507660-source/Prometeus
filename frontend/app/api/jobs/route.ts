// POST /api/jobs — создать задачу на переработку видео.
// Тело запроса: { url: string }
// Ответ: { job_id: string } или { error: string }

import { NextResponse } from "next/server";
import { getCurrentUser, getSupabaseServer } from "@/lib/supabase-server";
import { isYoutubeUrl } from "@/lib/youtube";

// Текст ошибки из RPC create_job (см. backend/supabase/003_rpc_functions.sql)
const LIMIT_ERROR = "Лимит бесплатных переработок исчерпан";

export async function POST(request: Request) {
  // 1. Только для залогиненных
  const user = await getCurrentUser();
  if (!user) {
    return NextResponse.json({ error: "Требуется вход" }, { status: 401 });
  }

  // 2. Разбираем тело запроса
  let url: unknown;
  try {
    const body = await request.json();
    url = body?.url;
  } catch {
    return NextResponse.json({ error: "Некорректный запрос" }, { status: 400 });
  }

  if (typeof url !== "string" || !isYoutubeUrl(url.trim())) {
    return NextResponse.json(
      { error: "Нужна ссылка на YouTube-видео (youtube.com или youtu.be)" },
      { status: 400 },
    );
  }

  // 3. Создаём задачу через RPC. Функция в базе атомарно проверяет лимит,
  //    создаёт задачу и списывает попытку. Вызов идёт от имени пользователя
  //    (его сессия в cookies), поэтому auth.uid() в базе = user.id.
  const supabase = getSupabaseServer();
  const { data, error } = await supabase.rpc("create_job", { p_url: url.trim() });

  if (error) {
    if (error.message.includes(LIMIT_ERROR)) {
      return NextResponse.json({ error: LIMIT_ERROR }, { status: 429 });
    }
    if (error.message.includes("Некорректная ссылка")) {
      return NextResponse.json({ error: error.message }, { status: 400 });
    }
    if (error.message.includes("Требуется авторизация")) {
      return NextResponse.json({ error: "Требуется вход" }, { status: 401 });
    }
    // Неожиданная ошибка — подробности в лог Vercel, пользователю общий текст
    console.error("[api/jobs] create_job error:", error);
    return NextResponse.json(
      { error: "Не удалось создать задачу. Попробуйте ещё раз." },
      { status: 500 },
    );
  }

  // RPC возвращает uuid созданной задачи
  return NextResponse.json({ job_id: data as string });
}
