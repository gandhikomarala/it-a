import React from 'react';
import { clsx } from 'clsx';
import { RiskLevel, ModelStage, DriftStatus } from '@/types';

interface StatusBadgeProps {
  status: RiskLevel | ModelStage | DriftStatus | string;
  className?: string;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status, className }) => {
  const getBadgeStyle = () => {
    switch (status) {
      case 'LOW':
      case 'PRODUCTION':
      case 'NORMAL':
      case 'COMPLETED':
      case 'EXCELLENT':
        return 'bg-emerald-950/80 text-emerald-400 border-emerald-800/60';
      case 'MEDIUM':
      case 'STAGING':
      case 'WARNING':
      case 'PROCESSING':
      case 'GOOD':
        return 'bg-amber-950/80 text-amber-400 border-amber-800/60';
      case 'HIGH':
      case 'CRITICAL':
      case 'FAILED':
      case 'REJECTED':
      case 'POOR':
        return 'bg-rose-950/80 text-rose-400 border-rose-800/60';
      default:
        return 'bg-slate-800 text-slate-300 border-slate-700';
    }
  };

  return (
    <span className={clsx('inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border font-mono', getBadgeStyle(), className)}>
      {status}
    </span>
  );
};
