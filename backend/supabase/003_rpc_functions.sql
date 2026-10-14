-- =============================================================================
-- 003_rpc_functions.sql
-- RPC-функции (вызываются через supabase.rpc('имя', {...})).
--
-- Все функции SECURITY DEFINER: работают с правами владельца (postgres)
-- и поэтому могут менять таблицы, закрытые RLS.
-- set search_path = '' — защита от подмены объектов; все имена полные.
--
-- ВАЖНО про права:
-- - create_job вызывает пользователь с фронтенда → EXECUTE для authenticated.
-- - claim_job, release_failed_job, cleanup_stuck_jobs — ТОЛЬКО для бэкенда
--   (service_role). Иначе любой пользователь мог бы вызвать release_failed_job
--   на своей задаче и вернуть себе попытку. Supabase по умолчанию разрешает
--   вызывать любые функции из public всем ролям, поэтому права отбираем явно.
-- =============================================================================


-- -----------------------------------------------------------------------------
-- create_job(p_url) — создать задачу от имени текущего пользователя.
-- Проверка лимита, создание задачи и увеличение счётчика — в ОДНОЙ транзакции.
-- FOR UPDATE блокирует строку пользователя: если он быстро нажмёт кнопку
-- 5 раз, запросы выполнятся по очереди и лимит не обойти.
-- -----------------------------------------------------------------------------
create or replace function public.create_job(p_url text)
returns uuid
language plpgsql
security definer
set search_path = ''
as $$
declare
    v_user_id     uuid := auth.uid();
    v_usage_count int;
    v_plan        text;
    v_url         text := btrim(p_url);
    v_job_id      uuid;
begin
    -- 1. Пользователь должен быть залогинен
    if v_user_id is null then
        raise exception 'Требуется авторизация';
    end if;

    -- 2. Базовая проверка ссылки. Полная проверка — в бэкенде (yt-dlp),
    --    здесь отсекаем мусор, чтобы не тратить на него попытку и место в базе.
    if v_url is null
       or length(v_url) > 500
       or v_url !~* '^https?://([a-z0-9-]+\.)?(youtube\.com|youtu\.be)/' then
        raise exception 'Некорректная ссылка: нужна ссылка на YouTube-видео';
    end if;

    -- 3. Блокируем строку пользователя до конца транзакции
    select u.usage_count, u.plan
      into v_usage_count, v_plan
      from public.users u
     where u.id = v_user_id
       for update;

    if not found then
        raise exception 'Профиль пользователя не найден';
    end if;

    -- 4. Лимит бесплатного тарифа
    if v_plan = 'free' and v_usage_count >= 3 then
        raise exception 'Лимит бесплатных переработок исчерпан';
    end if;

    -- 5. Создаём задачу
    insert into public.jobs (user_id, url, status)
    values (v_user_id, v_url, 'pending')
    returning id into v_job_id;

    -- 6. Списываем попытку
    update public.users
       set usage_count = usage_count + 1
     where id = v_user_id;

    return v_job_id;
end;
$$;

alter function public.create_job(text) owner to postgres;
revoke all on function public.create_job(text) from public, anon;
grant execute on function public.create_job(text) to authenticated, service_role;


-- -----------------------------------------------------------------------------
-- claim_job() — воркер атомарно забирает самую старую pending-задачу.
-- FOR UPDATE SKIP LOCKED: если строку уже забирает другой воркер,
-- она пропускается. Одну задачу никогда не возьмут дважды.
-- Возвращает 0 или 1 строку.
-- -----------------------------------------------------------------------------
create or replace function public.claim_job()
returns setof public.jobs
language plpgsql
security definer
set search_path = ''
as $$
begin
    return query
    update public.jobs j
       set status     = 'processing',
           claimed_at = now(),
           attempts   = j.attempts + 1
     where j.id = (
               select p.id
                 from public.jobs p
                where p.status = 'pending'
                order by p.created_at asc
                limit 1
                  for update skip locked
           )
    returning j.*;
end;
$$;

alter function public.claim_job() owner to postgres;
revoke all on function public.claim_job() from public, anon, authenticated;
grant execute on function public.claim_job() to service_role;


-- -----------------------------------------------------------------------------
-- release_failed_job(p_job_id, p_error) — пометить задачу как упавшую
-- и вернуть пользователю попытку (сбой системы не должен её съедать).
--
-- Решение: меняем только задачу в статусе 'processing'. Если уборщик уже
-- пометил её как error (и вернул попытку), повторного возврата не будет —
-- иначе счётчик можно было бы «накрутить» в минус.
-- -----------------------------------------------------------------------------
create or replace function public.release_failed_job(p_job_id uuid, p_error text)
returns void
language plpgsql
security definer
set search_path = ''
as $$
declare
    v_user_id uuid;
begin
    update public.jobs
       set status        = 'error',
           error_message = left(coalesce(p_error, 'Неизвестная ошибка'), 500)
     where id = p_job_id
       and status = 'processing'
    returning user_id into v_user_id;

    -- Задача не найдена или уже не в processing — ничего не делаем
    if v_user_id is null then
        return;
    end if;

    update public.users
       set usage_count = greatest(0, usage_count - 1)
     where id = v_user_id;
end;
$$;

alter function public.release_failed_job(uuid, text) owner to postgres;
revoke all on function public.release_failed_job(uuid, text) from public, anon, authenticated;
grant execute on function public.release_failed_job(uuid, text) to service_role;


-- -----------------------------------------------------------------------------
-- cleanup_stuck_jobs() — уборщик «зависших» задач.
-- Если воркер упал или Railway перезапустил контейнер посреди обработки,
-- задача навсегда осталась бы в 'processing'. Через 15 минут переводим её
-- в 'error' и возвращаем попытку пользователю.
--
-- Всё делается одним SQL-запросом с CTE:
--   stuck  — помечаем задачи как error и получаем их user_id;
--   counts — считаем, сколько задач было у каждого пользователя;
--   refund — возвращаем каждому пользователю столько попыток.
-- Возвращает количество очищенных задач.
-- -----------------------------------------------------------------------------
create or replace function public.cleanup_stuck_jobs()
returns int
language plpgsql
security definer
set search_path = ''
as $$
declare
    v_count int;
begin
    with stuck as (
        update public.jobs
           set status        = 'error',
               error_message = 'Превышено время обработки'
         where status = 'processing'
           and claimed_at < now() - interval '15 minutes'
        returning user_id
    ),
    counts as (
        select user_id, count(*)::int as cnt
          from stuck
         group by user_id
    ),
    refund as (
        update public.users u
           set usage_count = greatest(0, u.usage_count - c.cnt)
          from counts c
         where u.id = c.user_id
        returning u.id
    )
    -- На refund ниже никто не ссылается — это нормально: Postgres выполняет
    -- изменяющие CTE (UPDATE внутри WITH) всегда, даже без ссылки на них.
    select count(*)::int into v_count from stuck;

    return v_count;
end;
$$;

alter function public.cleanup_stuck_jobs() owner to postgres;
revoke all on function public.cleanup_stuck_jobs() from public, anon, authenticated;
grant execute on function public.cleanup_stuck_jobs() to service_role;
