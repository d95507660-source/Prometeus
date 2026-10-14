// Общие типы данных фронтенда. Совпадают с таблицей jobs в Supabase.

// Статус задачи: в очереди → обрабатывается → готово | ошибка
export type JobStatus = "pending" | "processing" | "done" | "error";

// Один твит, сгенерированный Claude
export type Tweet = {
  hook: string; // цепляющая первая строка
  body: string; // суть
  cta: string; // призыв к действию
  full_text: string; // готовый твит целиком (до 280 символов)
};

// Задача на переработку видео
export type Job = {
  id: string;
  url: string;
  status: JobStatus;
  error_message: string | null;
  output_tweets: Tweet[] | null;
  transcript_source: "subtitles" | "whisper" | null;
  created_at: string;
};

// Ответ GET /api/jobs/[id]
export type JobStatusResponse = Pick<
  Job,
  "status" | "output_tweets" | "error_message" | "transcript_source"
>;
