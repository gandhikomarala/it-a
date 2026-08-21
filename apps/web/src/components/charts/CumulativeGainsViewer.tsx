import React from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

interface CumulativeGainsViewerProps {
  title?: string;
  data?: any[];
}

export const CumulativeGainsViewer: React.FC<CumulativeGainsViewerProps> = ({
  title = "CumulativeGainsViewer",
  data = [
    { x: 0.0, y: 0.0 },
    { x: 0.2, y: 0.55 },
    { x: 0.4, y: 0.78 },
    { x: 0.6, y: 0.89 },
    { x: 0.8, y: 0.96 },
    { x: 1.0, y: 1.0 }
  ]
}) => {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
      <div className="flex justify-between items-center mb-4 pb-2 border-b border-slate-800">
        <div>
          <h3 className="text-sm font-semibold text-white">{title}</h3>
          <p className="text-xs text-slate-400">Cumulative Gains chart showing percentage of churners captured across top deciles.</p>
        </div>
      </div>
      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="x" stroke="#64748b" fontSize={11} />
            <YAxis dataKey="y" stroke="#64748b" fontSize={11} />
            <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }} />
            <Line type="monotone" dataKey="y" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
