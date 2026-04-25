/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        arxiv: {
          50:  '#f0f4ff',
          100: '#e0eaff',
          500: '#4a6cf7',
          600: '#3a56e8',
          700: '#2a44d4',
          900: '#1a2a8f',
        }
      }
    }
  },
  plugins: [],
}
