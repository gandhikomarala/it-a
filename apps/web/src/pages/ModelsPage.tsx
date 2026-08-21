import React from 'react';
import { Cpu } from 'lucide-react';
import { DataTable, Column } from '@/components/ui/DataTable';
import { StatusBadge } from '@/components/ui/StatusBadge';
import { MLModel } from '@/types';

const mockModels: MLModel[] = [
  {
    id: 'm-1',
    name: 'Customer_Churn_LightGBM',
    description: 'Gradient boosted tree classifier with optimized leaf nodes.',
    active_production_version: 3,
    production_roc_auc: 0.884,
    versions_count: 5,
    created_at: '2026-05-15T12:00:00Z'
  },
  {
    id: 'm-2',
    name: 'Random_Forest_Baseline',
    description: 'Ensemble bagging classifier.',
    active_production_version: 1,
    production_roc_auc: 0.841,
    versions_count: 2,
    created_at: '2026-04-10T12:00:00Z'
  }
];

export const ModelsPage: React.FC = () => {
  const columns: Column<MLModel>[] = [
    { header: 'Model Name', accessorKey: 'name', className: 'font-semibold text-slate-100' },
    {
      header: 'Production Version',
      cell: (m) => <span className="font-mono text-blue-400">v{m.active_production_version}</span>
    },
    {
      header: 'ROC-AUC',
      cell: (m) => <span className="font-mono text-amber-400 font-bold">{m.production_roc_auc}</span>
    },
    {
      header: 'Stage',
      cell: () => <StatusBadge status="PRODUCTION" />
    }
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white tracking-tight">Model Registry</h2>
          <p className="text-xs text-slate-400 mt-1">Immutable model artifacts, version promotion, and zero-downtime rollback.</p>
        </div>
      </div>

      <DataTable columns={columns} data={mockModels} />
    </div>
  );
};
