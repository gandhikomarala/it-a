import React, { useState } from 'react';
import { Search, Download } from 'lucide-react';
import { DataTable, Column } from '@/components/ui/DataTable';
import { StatusBadge } from '@/components/ui/StatusBadge';
import { Customer } from '@/types';

const mockCustomers: Customer[] = [
  {
    id: '1', customer_id: 'CUS-100291', first_name: 'Sarah', last_name: 'Connor',
    email: 'sarah.c@cyberdyne.io', age: 34, gender: 'Female', region: 'North America',
    city: 'San Francisco', income: 145000, signup_date: '2024-03-15',
    subscription_type: 'Premium', contract_type: 'Month-to-Month', payment_method: 'Credit Card',
    monthly_charge: 149.0, tenure_months: 14, total_spend: 2086.0, is_active: true,
    latest_churn_probability: 0.874, latest_risk_level: 'HIGH'
  },
  {
    id: '2', customer_id: 'CUS-100292', first_name: 'John', last_name: 'Doe',
    email: 'johndoe@initech.corp', age: 45, gender: 'Male', region: 'Europe',
    city: 'London', income: 82000, signup_date: '2023-11-01',
    subscription_type: 'Standard', contract_type: 'One-Year', payment_method: 'Bank Transfer',
    monthly_charge: 79.0, tenure_months: 28, total_spend: 2212.0, is_active: true,
    latest_churn_probability: 0.125, latest_risk_level: 'LOW'
  }
];

export const CustomersPage: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState('');

  const columns: Column<Customer>[] = [
    { header: 'Customer ID', accessorKey: 'customer_id', className: 'font-mono text-xs font-bold text-blue-400' },
    {
      header: 'Name',
      cell: (c) => (
        <div>
          <p className="font-medium text-slate-100">{c.first_name} {c.last_name}</p>
          <p className="text-xs text-slate-400 font-mono">{c.email}</p>
        </div>
      )
    },
    { header: 'Plan', accessorKey: 'subscription_type' },
    { header: 'Contract', accessorKey: 'contract_type' },
    {
      header: 'Monthly Fee',
      cell: (c) => <span className="font-mono">${c.monthly_charge.toFixed(2)}</span>
    },
    {
      header: 'Tenure',
      cell: (c) => <span className="font-mono">{c.tenure_months} mo</span>
    },
    {
      header: 'Churn Risk',
      cell: (c) => (
        <div className="flex items-center gap-2">
          <StatusBadge status={c.latest_risk_level || 'LOW'} />
          <span className="font-mono text-xs">
            {c.latest_churn_probability ? `${(c.latest_churn_probability * 100).toFixed(1)}%` : '-'}
          </span>
        </div>
      )
    }
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white tracking-tight">Customer Intelligence</h2>
          <p className="text-xs text-slate-400 mt-1">Search, segment, and explore real-time customer risk profiles.</p>
        </div>
        <div className="flex gap-3">
          <button className="flex items-center gap-2 px-3.5 py-2 bg-slate-900 border border-slate-800 rounded-lg text-xs font-medium text-slate-300 hover:text-white">
            <Download className="w-4 h-4" /> Export CSV
          </button>
        </div>
      </div>

      <div className="flex gap-4">
        <div className="flex-1 relative">
          <Search className="w-4 h-4 text-slate-500 absolute left-3 top-3" />
          <input
            type="text"
            placeholder="Search customers by ID, name, or email..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full bg-slate-900 border border-slate-800 rounded-lg pl-9 pr-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500"
          />
        </div>
      </div>

      <DataTable columns={columns} data={mockCustomers} />
    </div>
  );
};
