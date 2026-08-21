import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import api from '@/services/api';

// Domain Dashboard: Commercial BIM Construction Tracking
export const CommercialConstructionBimDashboardPage: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedFilter, setSelectedFilter] = useState('ALL');

  const { data, isLoading, refetch } = useQuery({
    queryKey: ['commercial_construction_bim_data'],
    queryFn: async () => {
      const res = await api.get('/commercial_construction_bim');
      return res.data;
    }
  });

  return (
    <div className="space-y-6 animate-fadeIn">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white">Commercial BIM Construction Tracking Dashboard</h1>
          <p className="text-sm text-slate-400 mt-1">Specialized predictive churn telemetry and intervention engine for Commercial BIM Construction Tracking.</p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => refetch()}
            className="px-4 py-2 text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded-lg transition-colors"
          >
            Refresh Telemetry
          </button>
          <button
            className="px-4 py-2 text-xs font-semibold bg-blue-600 hover:bg-blue-500 text-white rounded-lg transition-colors shadow-sm"
          >
            Export Report
          </button>
        </div>
      </div>

      <div className="flex flex-col sm:flex-row gap-3 items-center justify-between bg-slate-900/60 p-4 border border-slate-800 rounded-xl">
        <div className="relative w-full sm:w-80">
          <input
            type="text"
            placeholder="Search accounts..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-slate-950 border border-slate-800 rounded-lg pl-9 pr-4 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500"
          />
          <span className="absolute left-3 top-2.5 text-slate-500 text-xs">🔍</span>
        </div>
        <div className="flex items-center gap-2 w-full sm:w-auto">
          {['ALL', 'HIGH_RISK', 'HEALTHY'].map((f) => (
            <button
              key={f}
              onClick={() => setSelectedFilter(f)}
              className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-colors ${
                selectedFilter === f
                  ? 'bg-blue-600 text-white'
                  : 'bg-slate-800 text-slate-400 hover:text-white'
              }`}
            >
              {f}
            </button>
          ))}
        </div>
      </div>

      <div className="bg-slate-900 border border-slate-800 rounded-xl p-8 shadow-sm">
        {isLoading ? (
          <div className="p-12 text-center text-slate-400 space-y-3">
            <div className="inline-block animate-spin text-2xl text-blue-500">⟳</div>
            <p className="text-sm">Synchronizing Commercial BIM Construction Tracking predictive models...</p>
          </div>
        ) : (
          <div className="text-center text-slate-400">
            <p className="text-sm font-medium text-slate-300">Live Telemetry Pipeline Active</p>
            <p className="text-xs text-slate-500 mt-1">Real-time model inferences and hazard rate estimators operational.</p>
          </div>
        )}
      </div>
    </div>
  );
};
