import React from 'react';
import { clsx } from 'clsx';
import { RiskLevel } from '@/types';

interface RiskMeterProps {
  probability: number;
  riskLevel: RiskLevel;
}

export const RiskMeter: React.FC<RiskMeterProps> = ({ probability, riskLevel }) => {
  const pct = Math.round(probability * 100);

  const getMeterColor = () => {
    if (probability >= 0.70) return 'bg-rose-500 text-rose-400';
    if (probability >= 0.30) return 'bg-amber-500 text-amber-400';
    return 'bg-emerald-500 text-emerald-400';
  };

  return (
    <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-5 text-center">
      <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">Churn Probability</p>
      <div className="text-4xl font-extrabold font-mono my-2 tracking-tight">
        <span className={getMeterColor().split(' ')[1]}>{pct}%</span>
      </div>
      <div className="w-full bg-slate-800 h-2.5 rounded-full overflow-hidden mt-3">
        <div
          className={clsx('h-full transition-all duration-500 rounded-full', getMeterColor().split(' ')[0])}
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="flex justify-between text-[11px] text-slate-500 font-mono mt-1.5">
        <span>Low Risk (&lt;30%)</span>
        <span>High Risk (&gt;70%)</span>
      </div>
    </div>
  );
};
