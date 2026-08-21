import React from 'react';
import { Activity } from 'lucide-react';
import { ChartCard } from '@/components/ui/ChartCard';
import { StatusBadge } from '@/components/ui/StatusBadge';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip } from 'recharts';

const psiData = [
  { feature: 'monthly_charge', psi: 0.04, status: 'NORMAL' },
  { feature: 'tenure_months', psi: 0.06, status: 'NORMAL' },
  { feature: 'payment_failures', psi: 0.14, status: 'WARNING' },
  { feature: 'days_since_login', psi: 0.09, status: 'NORMAL' },
];

export const DriftPage: React.FC = () => {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white tracking-tight">Data & Prediction Drift Monitoring</h2>
          <p className="text-xs text-slate-400 mt-1">Population Stability Index (PSI) and Kolmogorov-Smirnov distribution shifts.</p>
        </div>
        <StatusBadge status="NORMAL" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ChartCard title="Feature PSI Values" subtitle="PSI &lt; 0.10: Normal | 0.10-0.25: Moderate shift">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={psiData}>
              <XAxis dataKey="feature" stroke="#64748b" fontSize={11} />
              <YAxis stroke="#64748b" fontSize={11} />
              <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }} />
              <Bar dataKey="psi" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4">
          <h3 className="text-base font-semibold text-white">Automated Retraining Safeguards</h3>
          <p className="text-xs text-slate-400">
            If feature drift exceeds PSI &ge; 0.25 or production ROC-AUC drops by &gt;5%, automated retraining is automatically queued.
          </p>
          <div className="p-4 bg-slate-950 border border-slate-800 rounded-lg text-xs space-y-2">
            <div className="flex justify-between">
              <span className="text-slate-400">Retraining Policy:</span>
              <span className="text-emerald-400 font-mono">ENABLED</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Max Tolerable PSI:</span>
              <span className="text-amber-400 font-mono">0.25</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Last Audit:</span>
              <span className="text-slate-300 font-mono">Today, 12:00 UTC</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
