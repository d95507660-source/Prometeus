// GET /auth/callback — сюда ведёт ссылка из письма Supabase (magic link).
// В адресе есть ?code=..., меняем его на сессию (cookies) и отправляем в кабинет.

import { NextResponse } from "next/server";
import { getSupabaseServer } from "@/lib/supabase-server";

export async function GET(request: Request) {
  const { searchParams, origin } = new URL(request.url);
  const code = searchParams.get("code");

  if (code) {
    const supabase = getSupabaseServer();
    const { error } = await supabase.auth.exchangeCodeForSession(code);
    if (!error) {
      return NextResponse.redirect(`${origin}/dashboard`);
    }
    console.error("[auth/callback] exchangeCodeForSession error:", error.message);
  }

  // Ссылка устарела, уже использована или открыта в другом браузере
  // (magic link работает только в том браузере, где запрашивали вход).
  return NextResponse.redirect(`${origin}/login?error=link`);
}
