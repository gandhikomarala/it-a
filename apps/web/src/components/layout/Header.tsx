import React from 'react';
import { Bell } from 'lucide-react';

export const Header: React.FC = () => {
  return (
    <header className="h-16 bg-slate-950/60 backdrop-blur-md border-b border-slate-800/80 sticky top-0 z-20 flex items-center justify-between px-8 ml-64">
      <div className="flex items-center gap-3">
        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-mono bg-emerald-950/80 text-emerald-400 border border-emerald-800/60">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
          PRODUCTION v3 ACTIVE (LightGBM)
        </span>
      </div>

      <div className="flex items-center gap-4">
        <button className="p-2 text-slate-400 hover:text-white rounded-lg hover:bg-slate-900 relative">
          <Bell className="w-5 h-5" />
          <span className="w-2 h-2 bg-blue-500 rounded-full absolute top-1.5 right-1.5"></span>
        </button>

        <div className="flex items-center gap-3 pl-3 border-l border-slate-800">
          <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center font-bold text-xs text-white">
            AD
          </div>
          <div>
            <p className="text-xs font-semibold text-slate-200">Admin User</p>
            <p className="text-[10px] text-slate-400 font-mono">SUPER_ADMIN</p>
          </div>
        </div>
      </div>
    </header>
  );
};
