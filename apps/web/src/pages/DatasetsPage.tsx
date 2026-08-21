import React from 'react';
import { Database, Upload } from 'lucide-react';
import { DataTable, Column } from '@/components/ui/DataTable';
import { StatusBadge } from '@/components/ui/StatusBadge';
import { Dataset } from '@/types';

const mockDatasets: Dataset[] = [
  {
    id: 'ds-1',
    name: 'enterprise_churn_2026_q2.parquet',
    description: 'Gold standard customer behavioral cohort dataset.',
    latest_version: 3,
    latest_quality_score: 94.2,
    row_count: 50000,
    column_count: 24,
    tags: ['production', 'gold'],
    versions_count: 3,
    created_at: '2026-06-01T10:00:00Z'
  }
];

export const DatasetsPage: React.FC = () => {
  const columns: Column<Dataset>[] = [
    { header: 'Dataset Name', accessorKey: 'name', className: 'font-mono font-medium text-slate-100' },
    { header: 'Rows', accessorKey: 'row_count' },
    { header: 'Columns', accessorKey: 'column_count' },
    {
      header: 'Quality Score',
      cell: (d) => (
        <span className="font-mono text-emerald-400 font-bold">{d.latest_quality_score}%</span>
      )
    },
    {
      header: 'Status',
      cell: () => <StatusBadge status="EXCELLENT" />
    }
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white tracking-tight">Dataset Hub</h2>
          <p className="text-xs text-slate-400 mt-1">Manage data ingestion pipelines, profiling, and quality validation.</p>
        </div>
        <button className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded-lg text-xs font-semibold text-white">
          <Upload className="w-4 h-4" /> Upload Dataset
        </button>
      </div>

      <DataTable columns={columns} data={mockDatasets} />
    </div>
  );
};
