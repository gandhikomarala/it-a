import React from 'react';

interface CohortHeatmapProps {
  months: string[];
  matrix: number[][];
}

export const CohortHeatmap: React.FC<CohortHeatmapProps> = ({ months, matrix }) => {
  const getColor = (val: number) => {
    if (val === 0) return 'bg-slate-900 text-slate-700';
    if (val >= 90) return 'bg-emerald-600 text-white';
    if (val >= 80) return 'bg-emerald-700 text-emerald-100';
    if (val >= 70) return 'bg-amber-600 text-white';
    return 'bg-rose-600 text-white';
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 overflow-x-auto">
      <h3 className="text-sm font-semibold text-white mb-4">Cohort Retention Heatmap (%)</h3>
      <table className="w-full text-xs text-center border-collapse">
        <thead>
          <tr>
            <th className="p-2 text-left text-slate-400">Cohort</th>
            {months.map((m, i) => (
              <th key={i} className="p-2 text-slate-400">Month {i}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, rowIdx) => (
            <tr key={rowIdx}>
              <td className="p-2 text-left font-mono text-slate-300 font-semibold">{months[rowIdx]}</td>
              {row.map((val, colIdx) => (
                <td key={colIdx} className="p-1">
                  <div className={`p-2 rounded font-mono font-medium ${getColor(val)}`}>
                    {val > 0 ? `${val.toFixed(1)}%` : '-'}
                  </div>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};
