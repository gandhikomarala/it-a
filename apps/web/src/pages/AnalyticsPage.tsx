import React from 'react';
import { BarChart3 } from 'lucide-react';
import { ChartCard } from '@/components/ui/ChartCard';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip } from 'recharts';

const segmentData = [
  { tier: 'Basic', mrrRisk: 28500, churnRate: 22.4 },
  { tier: 'Standard', mrrRisk: 52100, churnRate: 13.1 },
  { tier: 'Premium', mrrRisk: 28600, churnRate: 8.5 },
  { tier: 'Enterprise', mrrRisk: 33100, churnRate: 4.2 },
];

export const AnalyticsPage: React.FC = () => {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-white tracking-tight">Business Intelligence & Revenue Risk</h2>
        <p className="text-xs text-slate-400 mt-1">Cohort retention, subscriber lifetime value, and financial impact analytics.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ChartCard title="Monthly Revenue at Risk by Subscription Tier" subtitle="Estimated dollar risk per tier">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={segmentData}>
              <XAxis dataKey="tier" stroke="#64748b" fontSize={12} />
              <YAxis stroke="#64748b" fontSize={12} />
              <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }} />
              <Bar dataKey="mrrRisk" fill="#ef4444" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Churn Rate by Subscription Tier (%)" subtitle="Percentage of customers churning per tier">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={segmentData}>
              <XAxis dataKey="tier" stroke="#64748b" fontSize={12} />
              <YAxis stroke="#64748b" fontSize={12} />
              <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }} />
              <Bar dataKey="churnRate" fill="#f59e0b" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>
    </div>
  );
};
