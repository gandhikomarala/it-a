import React from 'react';
import { LucideIcon } from 'lucide-react';
import { clsx } from 'clsx';

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  change?: string;
  isPositive?: boolean;
  icon: LucideIcon;
  color?: 'blue' | 'emerald' | 'amber' | 'rose' | 'purple';
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  subtitle,
  change,
  isPositive,
  icon: Icon,
  color = 'blue'
}) => {
  const colorMap = {
    blue: 'text-blue-400 bg-blue-950/50 border-blue-800/40',
    emerald: 'text-emerald-400 bg-emerald-950/50 border-emerald-800/40',
    amber: 'text-amber-400 bg-amber-950/50 border-amber-800/40',
    rose: 'text-rose-400 bg-rose-950/50 border-rose-800/40',
    purple: 'text-purple-400 bg-purple-950/50 border-purple-800/40',
  };

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 backdrop-blur-sm relative overflow-hidden group hover:border-slate-700 transition-all">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs font-medium uppercase tracking-wider text-slate-400">{title}</p>
          <h3 className="text-2xl font-bold text-slate-100 mt-1 font-mono tracking-tight">{value}</h3>
          {subtitle && <p className="text-xs text-slate-500 mt-1">{subtitle}</p>}
          {change && (
            <p className={clsx('text-xs mt-1.5 font-medium flex items-center gap-1', isPositive ? 'text-emerald-400' : 'text-rose-400')}>
              <span>{isPositive ? '↑' : '↓'}</span> {change}
            </p>
          )}
        </div>
        <div className={clsx('p-3 rounded-xl border', colorMap[color])}>
          <Icon className="w-6 h-6" />
        </div>
      </div>
    </div>
  );
};
