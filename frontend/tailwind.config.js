/** @type {import('tailwindcss').Config} */
module.exports = {
  // Где Tailwind ищет классы. Классы из других папок попадут в CSS только если добавить их сюда.
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {},
  },
  plugins: [],
};
