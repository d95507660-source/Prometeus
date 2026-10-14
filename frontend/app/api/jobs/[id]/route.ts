// GET /api/jobs/[id] — текущий статус задачи (страница результата опрашивает его каждые 3 сек).
// Ответ: { status, output_tweets, error_message, transcript_source }

import { NextResponse } from "next/server";
import { getCurrentUser, getSupabaseServer } from "@/lib/supabase-server";

// Формат uuid. Без этой проверки кривой id дал бы ошибку базы (500), а не 404.
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

// Ответ всегда свежий: никакого кэширования
export const dynamic = "force-dynamic";

export async function GET(_request: Request, { params }: { params: { id: string } }) {
  const user = await getCurrentUser();
  if (!user) {
    return NextResponse.json({ error: "Требуется вход" }, { status: 401 });
  }

  if (!UUID_RE.test(params.id)) {
    return NextResponse.json({ error: "Задача не найдена" }, { status: 404 });
  }

  // RLS в базе вернёт задачу, только если она принадлежит текущему пользователю.
  // Чужая задача выглядит так же, как несуществующая, — 404.
  const supabase = getSupabaseServer();
  const { data, error } = await supabase
    .from("jobs")
    .select("status, output_tweets, error_message, transcript_source")
    .eq("id", params.id)
    .maybeSingle();

  if (error) {
    console.error("[api/jobs/id] select error:", error);
    return NextResponse.json({ error: "Не удалось получить статус задачи" }, { status: 500 });
  }
  if (!data) {
    return NextResponse.json({ error: "Задача не найдена" }, { status: 404 });
  }

  return NextResponse.json(data);
}
