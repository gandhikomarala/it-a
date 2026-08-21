import React from 'react';
import { SHAPContribution } from '@/types';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell, ReferenceLine } from 'recharts';

interface WaterfallProps {
  baseValue: number;
  predictionValue: number;
  contributions: SHAPContribution[];
}

export const SHAPWaterfallChart: React.FC<WaterfallProps> = ({ baseValue, predictionValue, contributions }) => {
  const data = contributions.map((c) => ({
    name: c.display_name,
    value: c.shap_value,
    impact: c.impact_direction,
  }));

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
      <div className="flex justify-between items-center mb-4">
        <div>
          <h3 className="text-sm font-semibold text-white">SHAP Factor Waterfall Plot</h3>
          <p className="text-xs text-slate-400">Baseline Churn Rate: {(baseValue * 100).toFixed(1)}% → Predicted: {(predictionValue * 100).toFixed(1)}%</p>
        </div>
      </div>
      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical">
            <XAxis type="number" stroke="#64748b" fontSize={11} />
            <YAxis dataKey="name" type="category" stroke="#64748b" fontSize={11} width={140} />
            <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }} />
            <ReferenceLine x={0} stroke="#475569" />
            <Bar dataKey="value">
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.value > 0 ? '#ef4444' : '#10b981'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
