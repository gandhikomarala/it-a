import React from 'react';
// Design System Component: AlertBanner
// Informational, warning, and error banner with dismissal support.

interface AlertBannerProps {
  type?: 'info' | 'warning' | 'error' | 'success';
  title: string;
  message?: string;
}
export const AlertBanner: React.FC<AlertBannerProps> = ({ type = 'info', title, message }) => {
  const styles = {
    info: 'bg-blue-950/60 border-blue-800 text-blue-300',
    warning: 'bg-amber-950/60 border-amber-800 text-amber-300',
    error: 'bg-rose-950/60 border-rose-800 text-rose-300',
    success: 'bg-emerald-950/60 border-emerald-800 text-emerald-300'
  };
  return (
    <div className={`p-4 border rounded-xl flex items-start gap-3 ${styles[type]}`}>
      <div>
        <h4 className="text-xs font-bold uppercase tracking-wider">{title}</h4>
        {message && <p className="text-xs mt-0.5 opacity-90">{message}</p>}
      </div>
    </div>
  );
};
