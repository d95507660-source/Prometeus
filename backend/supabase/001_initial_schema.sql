-- =============================================================================
-- 001_initial_schema.sql
-- Базовая схема: таблицы users и jobs, индексы, триггеры.
-- Выполнять в Supabase: SQL Editor → вставить файл целиком → Run.
-- Миграции выполняются по порядку: 001 → 002 → 003.
-- =============================================================================


-- -----------------------------------------------------------------------------
-- Таблица users — профиль пользователя и счётчик использований.
-- Строка создаётся автоматически триггером при регистрации (см. ниже).
-- -----------------------------------------------------------------------------
create table if not exists public.users (
    -- id совпадает с id в auth.users. При удалении аккаунта профиль удаляется тоже.
    id          uuid primary key references auth.users (id) on delete cascade,
    email       text not null unique,
    -- Сколько переработок сделал пользователь (лимит free — 3).
    usage_count int  not null default 0 check (usage_count >= 0),
    plan        text not null default 'free' check (plan in ('free', 'pro')),
    created_at  timestamptz not null default now()
);

comment on table public.users is 'Профили пользователей: тариф и счётчик переработок';


-- -----------------------------------------------------------------------------
-- Таблица jobs — задачи на переработку видео.
-- Жизненный цикл статуса: pending → processing → done | error
-- -----------------------------------------------------------------------------
create table if not exists public.jobs (
    id                uuid primary key default gen_random_uuid(),
    user_id           uuid not null references public.users (id) on delete cascade,
    url               text not null,
    status            text not null default 'pending'
                      check (status in ('pending', 'processing', 'done', 'error')),
    -- Понятный текст ошибки для пользователя (если status = 'error')
    error_message     text,
    transcript        text,
    -- Откуда взят транскрипт: субтитры YouTube или Whisper
    transcript_source text check (transcript_source in ('subtitles', 'whisper')),
    -- Массив из 5 твитов: [{hook, body, cta, full_text}, ...]
    output_tweets     jsonb,
    -- Сколько раз воркер брал задачу в работу (для отладки)
    attempts          int  not null default 0,
    created_at        timestamptz not null default now(),
    updated_at        timestamptz not null default now(),
    -- Когда воркер взял задачу. По нему уборщик находит «зависшие» задачи.
    claimed_at        timestamptz
);

comment on table public.jobs is 'Задачи: YouTube-ссылка → транскрипт → 5 твитов';


-- -----------------------------------------------------------------------------
-- Индексы
-- -----------------------------------------------------------------------------

-- Воркер ищет задачи по статусу ('pending')
create index if not exists idx_jobs_status on public.jobs (status);

-- Фронтенд показывает список задач конкретного пользователя
create index if not exists idx_jobs_user_id on public.jobs (user_id);

-- Воркер берёт самую старую pending-задачу, уборщик ищет старые processing
create index if not exists idx_jobs_status_created on public.jobs (status, created_at);


-- -----------------------------------------------------------------------------
-- Триггер: при регистрации в auth.users создаём строку в public.users.
--
-- SECURITY DEFINER — функция работает с правами владельца (postgres),
-- потому что Supabase Auth вставляет пользователя от служебной роли,
-- у которой нет прав на нашу таблицу.
-- set search_path = '' — защита от подмены таблиц через search_path;
-- поэтому все имена пишем полностью: public.users.
--
-- Решение: вход только по email, поэтому NEW.email всегда заполнен.
-- on conflict do nothing — на случай повторного срабатывания.
-- -----------------------------------------------------------------------------
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
begin
    insert into public.users (id, email, plan, usage_count)
    values (new.id, new.email, 'free', 0)
    on conflict (id) do nothing;
    return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
    after insert on auth.users
    for each row execute function public.handle_new_user();

-- Пользователи, которые зарегистрировались ДО создания триггера,
-- тоже получают профиль (иначе create_job для них упадёт).
insert into public.users (id, email)
select id, email
from auth.users
where email is not null
on conflict (id) do nothing;


-- -----------------------------------------------------------------------------
-- Триггер: при любом UPDATE задачи автоматически ставим updated_at = now().
-- -----------------------------------------------------------------------------
create or replace function public.set_updated_at()
returns trigger
language plpgsql
set search_path = ''
as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

drop trigger if exists jobs_set_updated_at on public.jobs;
create trigger jobs_set_updated_at
    before update on public.jobs
    for each row execute function public.set_updated_at();
