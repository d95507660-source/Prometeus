// Клиент Supabase для СЕРВЕРА: серверные компоненты, API-роуты, route handlers.
// Сессия пользователя хранится в cookies — клиент читает её оттуда,
// поэтому все запросы в базу идут от имени текущего пользователя (с RLS).

import { createServerClient } from "@supabase/ssr";
import { cookies } from "next/headers";

export function getSupabaseServer() {
  const cookieStore = cookies();

  return createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return cookieStore.getAll();
        },
        setAll(cookiesToSet) {
          try {
            cookiesToSet.forEach(({ name, value, options }) =>
              cookieStore.set(name, value, options),
            );
          } catch {
            // В серверных компонентах (страницах) записывать cookies нельзя —
            // Next.js кидает ошибку. Это нормально: сессию обновляет middleware.
          }
        },
      },
    },
  );
}

// Текущий пользователь или null.
// getUser() (а не getSession()) проверяет токен на сервере Supabase,
// поэтому подделанную cookie он не примет.
export async function getCurrentUser() {
  const supabase = getSupabaseServer();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  return user;
}
