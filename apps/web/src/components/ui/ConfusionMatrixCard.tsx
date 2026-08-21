import React from 'react';

interface ConfusionMatrixProps {
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  savings: number;
}

export const ConfusionMatrixCard: React.FC<ConfusionMatrixProps> = ({ tp, fp, fn, tn, savings }) => {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-sm font-semibold text-white">Confusion Matrix & ROI Impact</h3>
        <span className="text-xs font-mono text-emerald-400 font-bold">+${savings.toLocaleString()} Net Retention Gain</span>
      </div>
      <div className="grid grid-cols-2 gap-3 text-center text-xs">
        <div className="p-3 bg-emerald-950/40 border border-emerald-800/60 rounded-lg">
          <p className="text-slate-400">True Positive (Retained)</p>
          <p className="text-lg font-bold font-mono text-emerald-400">{tp.toLocaleString()}</p>
        </div>
        <div className="p-3 bg-amber-950/40 border border-amber-800/60 rounded-lg">
          <p className="text-slate-400">False Positive (False Alarm)</p>
          <p className="text-lg font-bold font-mono text-amber-400">{fp.toLocaleString()}</p>
        </div>
        <div className="p-3 bg-rose-950/40 border border-rose-800/60 rounded-lg">
          <p className="text-slate-400">False Negative (Missed Churn)</p>
          <p className="text-lg font-bold font-mono text-rose-400">{fn.toLocaleString()}</p>
        </div>
        <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-lg">
          <p className="text-slate-400">True Negative (Loyal Customer)</p>
          <p className="text-lg font-bold font-mono text-slate-200">{tn.toLocaleString()}</p>
        </div>
      </div>
    </div>
  );
};
