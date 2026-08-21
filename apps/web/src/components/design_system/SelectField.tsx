import React from 'react';
// Design System Component: SelectField
// Accessible form select component with dark theme styling.

interface SelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
  label?: string;
  error?: string;
}
export const SelectField: React.FC<SelectProps> = ({ label, error, children, className, ...props }) => {
  return (
    <div className="space-y-1">
      {label && <label className="block text-xs font-medium text-slate-400">{label}</label>}
      <select className={`w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500 ${className || ''}`} {...props}>
        {children}
      </select>
      {error && <p className="text-xs text-rose-400 mt-1">{error}</p>}
    </div>
  );
};
