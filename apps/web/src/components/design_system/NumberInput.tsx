import React from 'react';
// Design System Component: NumberInput
// Numeric stepper and formatted currency / percentage input.

interface NumberInputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  prefix?: string;
  suffix?: string;
}
export const NumberInput: React.FC<NumberInputProps> = ({ label, prefix, suffix, className, ...props }) => {
  return (
    <div className="space-y-1">
      {label && <label className="block text-xs font-medium text-slate-400">{label}</label>}
      <div className="relative flex items-center">
        {prefix && <span className="absolute left-3 text-xs text-slate-500 font-mono">{prefix}</span>}
        <input type="number" className={`w-full bg-slate-950 border border-slate-800 rounded-lg py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500 ${prefix ? 'pl-7' : 'pl-3'} ${suffix ? 'pr-7' : 'pr-3'} ${className || ''}`} {...props} />
        {suffix && <span className="absolute right-3 text-xs text-slate-500 font-mono">{suffix}</span>}
      </div>
    </div>
  );
};
