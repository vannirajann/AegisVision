/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        paper: '#F6F7F5',
        'paper-dim': '#EDEFEA',
        ink: '#12181F',
        'ink-soft': '#5B6472',
        'ink-faint': '#8A93A0',
        line: '#DADFE1',
        signal: '#2557D6',
        'signal-dim': '#E1E9FB',
        amber: '#B4711F',
        'amber-dim': '#F5E9D6',
        red: '#AE372B',
        'red-dim': '#F6DFDC',
        green: '#2E7A50',
        'green-dim': '#DFEBE3',
      },
      fontFamily: {
        display: ['"Space Grotesk"', 'sans-serif'],
        sans: ['Inter', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
      borderRadius: {
        sm: '3px',
        DEFAULT: '4px',
        md: '6px',
      },
    },
  },
  plugins: [],
};
