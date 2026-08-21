import React from 'react';
import { ShieldAlert, User, Clock } from 'lucide-react';
import { DataTable, Column } from '@/components/ui/DataTable';
import { StatusBadge } from '@/components/ui/StatusBadge';

interface AuditEntry {
  id: string;
  actor: string;
  action: string;
  resource: string;
  timestamp: string;
  status: string;
}

const mockAudits: AuditEntry[] = [
  { id: '1', actor: 'admin@enterprise-mlops.io', action: 'PROMOTE_MODEL', resource: 'ModelVersion:LightGBM-v3 -> PRODUCTION', timestamp: '2026-08-21 12:00:00 UTC', status: 'SUCCESS' },
  { id: '2', actor: 'system-retraining-worker', action: 'AUTO_RETRAIN_TRIGGER', resource: 'Dataset:enterprise_churn_2026_q2.parquet', timestamp: '2026-08-21 11:30:00 UTC', status: 'SUCCESS' },
  { id: '3', actor: 'analyst@enterprise-mlops.io', action: 'EXPORT_CUSTOMERS_CSV', resource: 'High_Risk_Cohort_1780_Records', timestamp: '2026-08-21 10:15:00 UTC', status: 'SUCCESS' }
];

export const AuditPage: React.FC = () => {
  const columns: Column<AuditEntry>[] = [
    { header: 'Timestamp', accessorKey: 'timestamp', className: 'font-mono text-xs text-slate-400' },
    { header: 'Actor', accessorKey: 'actor', className: 'font-semibold text-slate-200' },
    { header: 'Action', accessorKey: 'action', className: 'font-mono text-xs text-blue-400 font-bold' },
    { header: 'Resource / Details', accessorKey: 'resource', className: 'text-xs text-slate-300' },
    {
      header: 'Status',
      cell: (a) => <StatusBadge status={a.status} />
    }
  ];

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-white tracking-tight">Compliance & Audit Trail</h2>
        <p className="text-xs text-slate-400 mt-1">Immutable record of all administrative operations, model promotions, and data access.</p>
      </div>
      <DataTable columns={columns} data={mockAudits} />
    </div>
  );
};
