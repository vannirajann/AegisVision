import { useState, type FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { ShieldCheck, ArrowRight } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

export default function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const { login } = useAuth();
  const navigate = useNavigate();

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!email || !password) {
      setError('Enter your email and password to continue.');
      return;
    }
    setError('');
    login(email);
    navigate('/overview');
  }

  return (
    <div className="min-h-screen grid lg:grid-cols-[1.1fr_1fr]">
      {/* Brand panel */}
      <div className="hidden lg:flex relative flex-col justify-between bg-ink text-paper p-12 overflow-hidden">
        <svg
          className="absolute inset-0 w-full h-full opacity-[0.14]"
          viewBox="0 0 600 800"
          preserveAspectRatio="xMidYMid slice"
        >
          {Array.from({ length: 9 }).map((_, i) => (
            <circle
              key={i}
              cx="300"
              cy="260"
              r={40 + i * 45}
              fill="none"
              stroke="#F6F7F5"
              strokeWidth="1"
            />
          ))}
          <line x1="300" y1="0" x2="300" y2="800" stroke="#F6F7F5" strokeWidth="1" />
          <line x1="0" y1="260" x2="600" y2="260" stroke="#F6F7F5" strokeWidth="1" />
        </svg>

        <div className="relative flex items-center gap-2.5">
          <ShieldCheck size={22} strokeWidth={2} />
          <span className="font-display font-semibold text-lg tracking-tight">Perimeter</span>
        </div>

        <div className="relative max-w-sm">
          <p className="font-display text-3xl leading-snug">
            Every camera, sensor, and door — watched from one console.
          </p>
          <p className="mt-4 text-paper/60 text-sm leading-relaxed">
            Sign in to review overnight activity, acknowledge open alerts, and check
            the health of every connected device across your sites.
          </p>
        </div>

        <div className="relative flex items-center gap-6 text-xs text-paper/50 font-mono">
          <span>128 devices monitored</span>
          <span>·</span>
          <span>6 sites</span>
        </div>
      </div>

      {/* Form panel */}
      <div className="flex items-center justify-center p-8">
        <div className="w-full max-w-sm">
          <div className="lg:hidden flex items-center gap-2.5 mb-10">
            <ShieldCheck size={22} strokeWidth={2} className="text-ink" />
            <span className="font-display font-semibold text-lg text-ink">Perimeter</span>
          </div>

          <h1 className="font-display text-2xl font-semibold text-ink">Sign in</h1>
          <p className="mt-2 text-sm text-ink-soft">
            Use any email and password — this is a demo console.
          </p>

          <form onSubmit={handleSubmit} className="mt-8 space-y-5">
            <div>
              <label htmlFor="email" className="block text-sm font-medium text-ink mb-1.5">
                Work email
              </label>
              <input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@company.com"
                className="w-full rounded border border-line bg-white px-3.5 py-2.5 text-sm text-ink placeholder:text-ink-faint focus:outline-none focus:ring-2 focus:ring-signal/30 focus:border-signal transition-colors"
              />
            </div>

            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label htmlFor="password" className="block text-sm font-medium text-ink">
                  Password
                </label>
                <button
                  type="button"
                  className="text-xs text-signal hover:underline underline-offset-2"
                >
                  Forgot password
                </button>
              </div>
              <input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full rounded border border-line bg-white px-3.5 py-2.5 text-sm text-ink placeholder:text-ink-faint focus:outline-none focus:ring-2 focus:ring-signal/30 focus:border-signal transition-colors"
              />
            </div>

            {error && (
              <p className="text-sm text-red bg-red-dim border border-red/20 rounded px-3 py-2">
                {error}
              </p>
            )}

            <button
              type="submit"
              className="w-full inline-flex items-center justify-center gap-2 rounded bg-ink text-paper text-sm font-medium py-2.5 hover:bg-ink/90 transition-colors"
            >
              Continue to dashboard
              <ArrowRight size={16} />
            </button>
          </form>

          <p className="mt-8 text-xs text-ink-faint leading-relaxed">
            By continuing you agree this is sample data for demonstration purposes
            only. No real cameras or sites are connected.
          </p>
        </div>
      </div>
    </div>
  );
}
