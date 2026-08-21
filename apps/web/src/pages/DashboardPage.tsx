import React from 'react';
import { Users, AlertTriangle, DollarSign, Cpu } from 'lucide-react';
import { MetricCard } from '@/components/ui/MetricCard';
import { ChartCard } from '@/components/ui/ChartCard';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, BarChart, Bar, Cell } from 'recharts';

const trendData = [
  { month: 'Jan', churners: 120, rate: 11.2 },
  { month: 'Feb', churners: 135, rate: 12.1 },
  { month: 'Mar', churners: 145, rate: 12.8 },
  { month: 'Apr', churners: 160, rate: 13.5 },
  { month: 'May', churners: 178, rate: 14.2 },
];

const riskDistribution = [
  { name: 'Low Risk', count: 8200, color: '#10b981' },
  { name: 'Medium Risk', count: 2560, color: '#f59e0b' },
  { name: 'High Risk', count: 1780, color: '#ef4444' },
];

export const DashboardPage: React.FC = () => {
  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-white tracking-tight">Executive ML Dashboard</h2>
        <p className="text-xs text-slate-400 mt-1">Real-time churn risk indicators and active model health telemetry.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
        <MetricCard
          title="Total Customers"
          value="12,540"
          subtitle="11,820 Active Subscriptions"
          change="+6.2% MoM"
          isPositive={true}
          icon={Users}
          color="blue"
        />
        <MetricCard
          title="Overall Churn Rate"
          value="14.2%"
          subtitle="Target threshold: <15.0%"
          change="+0.7% MoM"
          isPositive={false}
          icon={AlertTriangle}
          color="amber"
        />
        <MetricCard
          title="Revenue at Risk"
          value="$142,300"
          subtitle="Monthly Recurring Revenue"
          change="-$12K from last week"
          isPositive={true}
          icon={DollarSign}
          color="rose"
        />
        <MetricCard
          title="Production Model"
          value="LightGBM-v3"
          subtitle="ROC-AUC: 0.884 | Latency: 1.2ms"
          change="Healthy"
          isPositive={true}
          icon={Cpu}
          color="emerald"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <ChartCard title="Monthly Churn Trend" subtitle="Predicted churn volume and rate over time">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={trendData}>
                <defs>
                  <linearGradient id="churnColor" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="month" stroke="#64748b" fontSize={12} tickLine={false} />
                <YAxis stroke="#64748b" fontSize={12} tickLine={false} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }}
                />
                <Area type="monotone" dataKey="churners" stroke="#3b82f6" strokeWidth={2} fillOpacity={1} fill="url(#churnColor)" />
              </AreaChart>
            </ResponsiveContainer>
          </ChartCard>
        </div>

        <div>
          <ChartCard title="Cohort Risk Distribution" subtitle="Active customer breakdown by risk tier">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={riskDistribution} layout="vertical">
                <XAxis type="number" stroke="#64748b" fontSize={11} />
                <YAxis dataKey="name" type="category" stroke="#64748b" fontSize={11} width={80} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }}
                />
                <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                  {riskDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>
        </div>
      </div>
    </div>
  );
};
