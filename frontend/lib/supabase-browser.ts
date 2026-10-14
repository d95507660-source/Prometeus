// Клиент Supabase для БРАУЗЕРА (клиентские компоненты с "use client").
// Использует публичный anon-ключ: он и так виден в браузере,
// а данные защищены RLS-политиками в базе.

import { createBrowserClient } from "@supabase/ssr";

export function getSupabaseBrowser() {
  return createBrowserClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
  );
}
