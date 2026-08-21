import React from 'react';
import { Layers, Search, Sparkles } from 'lucide-react';
import { DataTable, Column } from '@/components/ui/DataTable';

interface FeatureDef {
  id: string;
  name: string;
  category: string;
  dataType: string;
  importance: number;
  description: string;
}

const mockFeatures: FeatureDef[] = [
  { id: '1', name: 'contract_type_encoded', category: 'Contract', dataType: 'Categorical', importance: 0.284, description: 'Month-to-Month vs Annual contract indicator' },
  { id: '2', name: 'payment_failures_count', category: 'Billing', dataType: 'Numeric', importance: 0.218, description: 'Failed transaction count in last 90 days' },
  { id: '3', name: 'satisfaction_score_deficit', category: 'Support', dataType: 'Numeric', importance: 0.174, description: 'CSAT drop relative to benchmark 3.5' },
  { id: '4', name: 'days_since_last_login', category: 'Engagement', dataType: 'Numeric', importance: 0.152, description: 'Inactivity duration in calendar days' },
  { id: '5', name: 'tenure_months_log', category: 'Lifecycle', dataType: 'Numeric', importance: 0.112, description: 'Log-transformed subscriber lifetime' }
];

export const FeaturesPage: React.FC = () => {
  const columns: Column<FeatureDef>[] = [
    { header: 'Feature Name', accessorKey: 'name', className: 'font-mono text-xs font-semibold text-blue-400' },
    { header: 'Category', accessorKey: 'category' },
    { header: 'Type', accessorKey: 'dataType' },
    {
      header: 'Global Importance',
      cell: (f) => (
        <div className="flex items-center gap-2">
          <div className="w-24 bg-slate-800 h-2 rounded-full overflow-hidden">
            <div className="bg-blue-500 h-full rounded-full" style={{ width: `${f.importance * 100 * 3}%` }} />
          </div>
          <span className="font-mono text-xs text-slate-300">{(f.importance * 100).toFixed(1)}%</span>
        </div>
      )
    },
    { header: 'Description', accessorKey: 'description', className: 'text-xs text-slate-400 max-w-md' }
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white tracking-tight">Feature Store & Catalog</h2>
          <p className="text-xs text-slate-400 mt-1">Curated behavioral features, global SHAP importance, and lineage.</p>
        </div>
      </div>
      <DataTable columns={columns} data={mockFeatures} />
    </div>
  );
};
