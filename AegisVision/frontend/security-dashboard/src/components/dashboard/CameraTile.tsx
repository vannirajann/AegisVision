import { VideoOff, Maximize2 } from 'lucide-react';
import type { Camera } from '../../types';
import StatusBadge from '../ui/StatusBadge';

export default function CameraTile({ camera }: { camera: Camera }) {
  const isOffline = camera.status === 'offline';

  return (
    <div className="group rounded border border-line bg-white overflow-hidden">
      <div className="relative aspect-video bg-ink flex items-center justify-center overflow-hidden">
        {isOffline ? (
          <div className="flex flex-col items-center gap-2 text-paper/40">
            <VideoOff size={22} />
            <span className="text-xs font-mono">No signal</span>
          </div>
        ) : (
          <>
            {/* Simulated feed texture */}
            <div
              className="absolute inset-0 opacity-30"
              style={{
                backgroundImage:
                  'repeating-linear-gradient(115deg, rgba(246,247,245,0.08) 0px, rgba(246,247,245,0.08) 1px, transparent 1px, transparent 14px)',
              }}
            />
            <span className="font-mono text-[11px] text-paper/50 absolute top-2.5 left-2.5">
              {camera.id}
            </span>
            <span className="font-mono text-[11px] text-paper/50 absolute top-2.5 right-2.5">
              {camera.resolution}
            </span>
            <button
              aria-label="Expand feed"
              className="absolute bottom-2.5 right-2.5 text-paper/60 opacity-0 group-hover:opacity-100 transition-opacity"
            >
              <Maximize2 size={15} />
            </button>
            <span className="absolute bottom-2.5 left-2.5 flex items-center gap-1.5 text-paper/70 text-[11px] font-mono">
              <span className="w-1.5 h-1.5 rounded-full bg-red animate-pulse" />
              LIVE
            </span>
          </>
        )}
      </div>

      <div className="p-3.5 flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="text-sm font-medium text-ink truncate">{camera.name}</p>
          <p className="text-xs text-ink-faint mt-0.5">
            {camera.zone} · motion {camera.lastMotion}
          </p>
        </div>
        <StatusBadge status={camera.status} />
      </div>
    </div>
  );
}
