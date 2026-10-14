// Middleware выполняется перед каждым запросом (кроме статики, см. matcher).
// Задачи:
// 1. Обновлять сессию Supabase: если токен истёк, обновляем его и пишем
//    новые cookies в ответ (стандартный паттерн @supabase/ssr).
// 2. Не пускать неавторизованных на /dashboard и /jobs/* — редирект на /login.

import { createServerClient } from "@supabase/ssr";
import { NextResponse, type NextRequest } from "next/server";

// Страницы, доступные только после входа
const PROTECTED_PREFIXES = ["/dashboard", "/jobs"];

export async function middleware(request: NextRequest) {
  let response = NextResponse.next({ request });

  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return request.cookies.getAll();
        },
        setAll(cookiesToSet) {
          // Записываем обновлённые cookies и в запрос (для серверного кода дальше),
          // и в ответ (чтобы браузер их сохранил).
          cookiesToSet.forEach(({ name, value }) => request.cookies.set(name, value));
          response = NextResponse.next({ request });
          cookiesToSet.forEach(({ name, value, options }) =>
            response.cookies.set(name, value, options),
          );
        },
      },
    },
  );

  // ВАЖНО: между createServerClient и getUser() не должно быть другого кода —
  // getUser() и обновляет сессию.
  const {
    data: { user },
  } = await supabase.auth.getUser();

  const path = request.nextUrl.pathname;
  const isProtected = PROTECTED_PREFIXES.some(
    (prefix) => path === prefix || path.startsWith(prefix + "/"),
  );

  if (!user && isProtected) {
    const loginUrl = request.nextUrl.clone();
    loginUrl.pathname = "/login";
    loginUrl.search = "";
    return NextResponse.redirect(loginUrl);
  }

  return response;
}

export const config = {
  // Не запускаем middleware для статики и картинок — это лишние запросы к Supabase.
  matcher: ["/((?!_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp|ico)$).*)"],
};
