import type { AppUser } from '../../types';
import StatusBadge from '../ui/StatusBadge';

export default function UserTable({ users }: { users: AppUser[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-xs text-ink-faint border-b border-line">
            <th className="font-medium py-2.5 pr-4">Name</th>
            <th className="font-medium py-2.5 pr-4">Email</th>
            <th className="font-medium py-2.5 pr-4">Role</th>
            <th className="font-medium py-2.5 pr-4">Last active</th>
            <th className="font-medium py-2.5">Status</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-line">
          {users.map((u) => (
            <tr key={u.id} className="hover:bg-paper-dim/60 transition-colors">
              <td className="py-3 pr-4">
                <div className="flex items-center gap-2.5">
                  <span className="w-7 h-7 rounded-full bg-paper-dim border border-line flex items-center justify-center text-xs font-medium text-ink-soft uppercase">
                    {u.name.charAt(0)}
                  </span>
                  <span className="text-ink font-medium">{u.name}</span>
                </div>
              </td>
              <td className="py-3 pr-4 text-ink-soft">{u.email}</td>
              <td className="py-3 pr-4 text-ink-soft">{u.role}</td>
              <td className="py-3 pr-4 text-ink-soft">{u.lastActive}</td>
              <td className="py-3"><StatusBadge status={u.status} /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
