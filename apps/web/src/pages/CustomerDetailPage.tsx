import React from 'react';
import { useParams } from 'react-router-dom';
import { User, Activity, AlertTriangle, ShieldCheck, Mail, Phone, MapPin } from 'lucide-react';
import { StatusBadge } from '@/components/ui/StatusBadge';
import { RiskMeter } from '@/components/ui/RiskMeter';

export const CustomerDetailPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();

  return (
    <div className="space-y-8">
      {/* Header Profile */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 flex items-start justify-between">
        <div className="flex items-center gap-4">
          <div className="w-14 h-14 rounded-full bg-blue-600 flex items-center justify-center text-xl font-bold text-white">
            SC
          </div>
          <div>
            <div className="flex items-center gap-3">
              <h2 className="text-xl font-bold text-white">Sarah Connor</h2>
              <span className="text-xs font-mono text-blue-400">CUS-100291</span>
              <StatusBadge status="HIGH" />
            </div>
            <div className="flex items-center gap-4 text-xs text-slate-400 mt-1.5">
              <span className="flex items-center gap-1"><Mail className="w-3.5 h-3.5" /> sarah.c@cyberdyne.io</span>
              <span className="flex items-center gap-1"><MapPin className="w-3.5 h-3.5" /> San Francisco, CA</span>
              <span className="font-mono text-slate-300">Member since Mar 2024 (14 mo)</span>
            </div>
          </div>
        </div>
        <div className="text-right">
          <p className="text-xs text-slate-400 uppercase tracking-wider">Monthly Recurring Fee</p>
          <p className="text-2xl font-bold font-mono text-white mt-0.5">$149.00</p>
        </div>
      </div>

      {/* Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
            <h3 className="text-sm font-semibold text-white mb-4">Behavioral Timeline & Events</h3>
            <div className="space-y-3">
              <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-lg text-xs flex justify-between">
                <div>
                  <span className="font-semibold text-rose-400">Payment Failed (Credit Card Expired)</span>
                  <p className="text-slate-400 mt-0.5">Attempted automatic billing charge of $149.00</p>
                </div>
                <span className="text-slate-500 font-mono">3 days ago</span>
              </div>
              <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-lg text-xs flex justify-between">
                <div>
                  <span className="font-semibold text-amber-400">Support Ticket Logged: Invoicing issue</span>
                  <p className="text-slate-400 mt-0.5">Resolution time: 54 hours (SLA breach)</p>
                </div>
                <span className="text-slate-500 font-mono">12 days ago</span>
              </div>
            </div>
          </div>
        </div>

        <div>
          <RiskMeter probability={0.874} riskLevel="HIGH" />
        </div>
      </div>
    </div>
  );
};
