import { useMemo, useState } from 'react';
import type { Camera } from '../../types';
import CameraTile from './CameraTile';

export default function CameraGrid({ cameras }: { cameras: Camera[] }) {
  const [zone, setZone] = useState('All zones');
  const zones = useMemo(() => ['All zones', ...new Set(cameras.map((c) => c.zone))], [cameras]);
  const filtered = zone === 'All zones' ? cameras : cameras.filter((c) => c.zone === zone);

  return (
    <div>
      <div className="flex items-center gap-2 mb-4 overflow-x-auto no-scrollbar pb-1">
        {zones.map((z) => (
          <button
            key={z}
            onClick={() => setZone(z)}
            className={`shrink-0 rounded-full px-3 py-1.5 text-xs font-medium transition-colors ${
              zone === z
                ? 'bg-ink text-paper'
                : 'bg-white border border-line text-ink-soft hover:text-ink'
            }`}
          >
            {z}
          </button>
        ))}
      </div>

      <div className="grid sm:grid-cols-2 xl:grid-cols-3 gap-4">
        {filtered.map((camera) => (
          <CameraTile key={camera.id} camera={camera} />
        ))}
      </div>
    </div>
  );
}
