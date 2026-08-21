import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard, Users, Database, Layers, GitBranch,
  Cpu, Target, Activity, Zap, BarChart3, FileText,
  Bell, ShieldAlert, Sliders, LogOut
} from 'lucide-react';
import { clsx } from 'clsx';

const navItems = [
  { name: 'Dashboard', path: '/dashboard', icon: LayoutDashboard },
  { name: 'Customers', path: '/customers', icon: Users },
  { name: 'Datasets', path: '/datasets', icon: Database },
  { name: 'Features', path: '/features', icon: Layers },
  { name: 'Experiments', path: '/experiments', icon: GitBranch },
  { name: 'Model Registry', path: '/models', icon: Cpu },
  { name: 'Predictions', path: '/predictions', icon: Target },
  { name: 'Monitoring & Drift', path: '/drift', icon: Activity },
  { name: 'Analytics', path: '/analytics', icon: BarChart3 },
  { name: 'Reports', path: '/reports', icon: FileText },
  { name: 'Audit Logs', path: '/audit', icon: ShieldAlert },
  { name: 'Settings', path: '/settings', icon: Sliders },
];

export const Sidebar: React.FC = () => {
  return (
    <aside className="w-64 bg-slate-950 border-r border-slate-800/80 flex flex-col h-screen fixed left-0 top-0 z-30">
      <div className="h-16 flex items-center px-6 border-b border-slate-800/80 gap-3">
        <div className="p-2 bg-blue-600 rounded-lg text-white font-bold">
          <Zap className="w-5 h-5" />
        </div>
        <div>
          <h1 className="text-sm font-bold tracking-tight text-white leading-tight">CHURN MLOPS</h1>
          <p className="text-[10px] text-blue-400 font-mono tracking-wider">ENTERPRISE PLATFORM</p>
        </div>
      </div>

      <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                clsx(
                  'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors',
                  isActive
                    ? 'bg-blue-600/15 text-blue-400 border border-blue-500/30'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
                )
              }
            >
              <Icon className="w-4 h-4 flex-shrink-0" />
              <span>{item.name}</span>
            </NavLink>
          );
        })}
      </nav>

      <div className="p-4 border-t border-slate-800/80">
        <button
          onClick={() => {
            localStorage.removeItem('access_token');
            window.location.href = '/login';
          }}
          className="w-full flex items-center justify-center gap-2 px-3 py-2 text-xs font-medium text-slate-400 hover:text-rose-400 hover:bg-rose-950/30 rounded-lg transition-colors border border-slate-800"
        >
          <LogOut className="w-4 h-4" />
          <span>Sign Out</span>
        </button>
      </div>
    </aside>
  );
};
