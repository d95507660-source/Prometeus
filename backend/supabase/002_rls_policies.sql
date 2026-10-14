-- =============================================================================
-- 002_rls_policies.sql
-- Row Level Security: кто какие строки может видеть и менять.
--
-- Принцип:
-- - Залогиненный пользователь (роль authenticated) может только ЧИТАТЬ свои строки.
-- - Все изменения — только через RPC-функции (003) или бэкенд с service_role.
--   service_role обходит RLS автоматически, политики для него не нужны.
-- - Анонимам (роль anon) — ничего.
-- =============================================================================


-- Включаем RLS. Без политик RLS запрещает всё — это и есть «запрет по умолчанию».
alter table public.users enable row level security;
alter table public.jobs  enable row level security;


-- -----------------------------------------------------------------------------
-- users: пользователь видит только свою строку
-- -----------------------------------------------------------------------------
drop policy if exists "users_select_own" on public.users;
create policy "users_select_own"
    on public.users
    for select
    to authenticated
    using ((select auth.uid()) = id);
-- (select auth.uid()) вместо auth.uid() — рекомендация Supabase:
-- функция вычисляется один раз на запрос, а не для каждой строки.

-- Политик на INSERT/UPDATE/DELETE нет → эти операции запрещены.
-- Иначе пользователь мог бы из консоли браузера обнулить себе usage_count.


-- -----------------------------------------------------------------------------
-- jobs: пользователь видит только свои задачи
-- -----------------------------------------------------------------------------
drop policy if exists "jobs_select_own" on public.jobs;
create policy "jobs_select_own"
    on public.jobs
    for select
    to authenticated
    using ((select auth.uid()) = user_id);

-- Политик на INSERT/UPDATE/DELETE нет → создание задач только через RPC create_job.


-- -----------------------------------------------------------------------------
-- Дополнительная защита на уровне прав (GRANT/REVOKE).
-- Supabase по умолчанию выдаёт anon и authenticated все права на таблицы
-- в схеме public, а защищает только RLS. Отбираем лишнее явно: если кто-то
-- когда-нибудь случайно выключит RLS, данные всё равно не утекут.
-- -----------------------------------------------------------------------------
revoke all on public.users from anon;
revoke all on public.jobs  from anon;

revoke insert, update, delete, truncate on public.users from authenticated;
revoke insert, update, delete, truncate on public.jobs  from authenticated;

grant select on public.users to authenticated;
grant select on public.jobs  to authenticated;
