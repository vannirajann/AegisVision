import { useState } from 'react';
import { UserPlus, PlusCircle } from 'lucide-react';
import DeviceTable from '../components/dashboard/DeviceTable';
import UserTable from '../components/dashboard/UserTable';
import { devices, users } from '../data/mockData';

export default function UsersDevices() {
  const [tab, setTab] = useState<'devices' | 'users'>('devices');

  return (
    <div className="rounded border border-line bg-white p-5">
      <div className="flex items-center justify-between mb-5 flex-wrap gap-3">
        <div className="flex items-center gap-1 bg-paper-dim rounded p-1">
          <button
            onClick={() => setTab('devices')}
            className={`rounded px-3.5 py-1.5 text-sm font-medium transition-colors ${
              tab === 'devices' ? 'bg-white text-ink shadow-sm' : 'text-ink-soft'
            }`}
          >
            Devices ({devices.length})
          </button>
          <button
            onClick={() => setTab('users')}
            className={`rounded px-3.5 py-1.5 text-sm font-medium transition-colors ${
              tab === 'users' ? 'bg-white text-ink shadow-sm' : 'text-ink-soft'
            }`}
          >
            Users ({users.length})
          </button>
        </div>

        <button className="inline-flex items-center gap-1.5 rounded bg-ink text-paper text-sm font-medium px-3.5 py-2 hover:bg-ink/90 transition-colors">
          {tab === 'devices' ? <PlusCircle size={15} /> : <UserPlus size={15} />}
          {tab === 'devices' ? 'Add device' : 'Invite user'}
        </button>
      </div>

      {tab === 'devices' ? <DeviceTable devices={devices} /> : <UserTable users={users} />}
    </div>
  );
}
