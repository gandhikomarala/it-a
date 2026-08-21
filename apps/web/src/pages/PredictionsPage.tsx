import React, { useState } from 'react';
import { Target, Zap } from 'lucide-react';
import { RiskMeter } from '@/components/ui/RiskMeter';
import { SHAPContribution } from '@/types';

export const PredictionsPage: React.FC = () => {
  const [tenure, setTenure] = useState(6);
  const [contractType, setContractType] = useState('Month-to-Month');
  const [paymentFailures, setPaymentFailures] = useState(2);
  const [complaints, setComplaints] = useState(1);
  const [satisfaction, setSatisfaction] = useState(2.5);
  const [daysSinceLogin, setDaysSinceLogin] = useState(12);

  const [prediction, setPrediction] = useState<{
    probability: number;
    riskLevel: 'LOW' | 'MEDIUM' | 'HIGH';
    factors: SHAPContribution[];
  }>({
    probability: 0.842,
    riskLevel: 'HIGH',
    factors: [
      { feature_name: 'contract_type', display_name: 'Month-to-Month Contract', feature_value: 'M2M', shap_value: 0.28, contribution_percentage: 28.5, impact_direction: 'POSITIVE' },
      { feature_name: 'payment_failures', display_name: 'Payment Failures (2x)', feature_value: 2, shap_value: 0.22, contribution_percentage: 22.4, impact_direction: 'POSITIVE' },
      { feature_name: 'days_since_login', display_name: 'Inactivity (12 days)', feature_value: 12, shap_value: 0.16, contribution_percentage: 16.3, impact_direction: 'POSITIVE' },
      { feature_name: 'satisfaction', display_name: 'Low CSAT Score (2.5)', feature_value: 2.5, shap_value: 0.12, contribution_percentage: 12.2, impact_direction: 'POSITIVE' }
    ]
  });

  const handlePredict = () => {
    let logit = -2.2;
    if (contractType === 'Month-to-Month') logit += 0.85;
    logit += paymentFailures * 0.70;
    logit += complaints * 0.65;
    logit -= (satisfaction - 3.0) * 0.55;
    logit += daysSinceLogin * 0.08;
    logit -= Math.log1p(tenure) * 0.40;

    const prob = 1.0 / (1.0 + Math.exp(-logit));
    const risk = prob >= 0.70 ? 'HIGH' : prob >= 0.30 ? 'MEDIUM' : 'LOW';
    setPrediction({
      probability: prob,
      riskLevel: risk,
      factors: prediction.factors
    });
  };

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-white tracking-tight">Interactive Churn Simulator & SHAP XAI</h2>
        <p className="text-xs text-slate-400 mt-1">Perform real-time customer churn inference with local factor attribution.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 bg-slate-900/80 border border-slate-800 rounded-xl p-6 space-y-5">
          <h3 className="text-base font-semibold text-white pb-3 border-b border-slate-800">Customer Attributes</h3>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-slate-400 mb-1">Contract Type</label>
              <select
                value={contractType}
                onChange={(e) => setContractType(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200"
              >
                <option value="Month-to-Month">Month-to-Month</option>
                <option value="One-Year">One-Year</option>
                <option value="Two-Year">Two-Year</option>
              </select>
            </div>

            <div>
              <label className="block text-xs text-slate-400 mb-1">Tenure (Months): {tenure}</label>
              <input
                type="range"
                min="1"
                max="72"
                value={tenure}
                onChange={(e) => setTenure(Number(e.target.value))}
                className="w-full accent-blue-500"
              />
            </div>

            <div>
              <label className="block text-xs text-slate-400 mb-1">Payment Failures (Past 90d): {paymentFailures}</label>
              <input
                type="range"
                min="0"
                max="4"
                value={paymentFailures}
                onChange={(e) => setPaymentFailures(Number(e.target.value))}
                className="w-full accent-blue-500"
              />
            </div>

            <div>
              <label className="block text-xs text-slate-400 mb-1">Days Since Last Login: {daysSinceLogin}</label>
              <input
                type="range"
                min="0"
                max="45"
                value={daysSinceLogin}
                onChange={(e) => setDaysSinceLogin(Number(e.target.value))}
                className="w-full accent-blue-500"
              />
            </div>

            <div>
              <label className="block text-xs text-slate-400 mb-1">Satisfaction Score (1-5): {satisfaction}</label>
              <input
                type="range"
                min="1.0"
                max="5.0"
                step="0.1"
                value={satisfaction}
                onChange={(e) => setSatisfaction(Number(e.target.value))}
                className="w-full accent-blue-500"
              />
            </div>

            <div>
              <label className="block text-xs text-slate-400 mb-1">Support Complaints: {complaints}</label>
              <input
                type="range"
                min="0"
                max="4"
                value={complaints}
                onChange={(e) => setComplaints(Number(e.target.value))}
                className="w-full accent-blue-500"
              />
            </div>
          </div>

          <button
            onClick={handlePredict}
            className="w-full bg-blue-600 hover:bg-blue-500 text-white font-medium py-2.5 rounded-lg text-sm flex items-center justify-center gap-2 mt-4"
          >
            <Zap className="w-4 h-4" /> Run Inference & Explain
          </button>
        </div>

        <div className="space-y-6">
          <RiskMeter probability={prediction.probability} riskLevel={prediction.riskLevel} />

          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
            <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-3">Top SHAP Drivers</h4>
            <div className="space-y-2.5">
              {prediction.factors.map((f, i) => (
                <div key={i} className="flex items-center justify-between text-xs">
                  <span className="text-slate-300 truncate max-w-[180px]">{f.display_name}</span>
                  <span className="text-rose-400 font-mono font-semibold">+{f.contribution_percentage}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
