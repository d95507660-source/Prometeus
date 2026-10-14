// Проверка, что ссылка ведёт на YouTube.
// Используется и в форме (браузер), и в API-роуте (сервер).
// Окончательную проверку (существует ли видео, длина) делает бэкенд.

const YOUTUBE_HOSTS = new Set([
  "youtube.com",
  "www.youtube.com",
  "m.youtube.com",
  "music.youtube.com",
  "youtu.be",
]);

export function isYoutubeUrl(value: string): boolean {
  if (value.length > 500) return false;
  try {
    const url = new URL(value);
    return (url.protocol === "https:" || url.protocol === "http:") && YOUTUBE_HOSTS.has(url.hostname);
  } catch {
    // new URL() кидает ошибку на строку, которая вообще не похожа на ссылку
    return false;
  }
}
