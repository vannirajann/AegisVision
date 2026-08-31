import CameraGrid from '../components/dashboard/CameraGrid';
import { cameras } from '../data/mockData';

export default function Cameras() {
  return (
    <div>
      <p className="text-sm text-ink-soft mb-5">
        {cameras.length} cameras across {new Set(cameras.map((c) => c.zone)).size} zones.
      </p>
      <CameraGrid cameras={cameras} />
    </div>
  );
}
