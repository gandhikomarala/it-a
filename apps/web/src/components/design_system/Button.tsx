import React from 'react';
// Design System Component: Button
// Primary, secondary, outline, danger, and ghost action buttons with loading spinners.

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'danger' | 'ghost';
  isLoading?: boolean;
}
export const Button: React.FC<ButtonProps> = ({ variant = 'primary', isLoading, children, className, ...props }) => {
  const base = "inline-flex items-center justify-center px-4 py-2 text-xs font-semibold rounded-lg transition-colors focus:outline-none";
  const styles = {
    primary: "bg-blue-600 hover:bg-blue-500 text-white shadow-sm",
    secondary: "bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700",
    danger: "bg-rose-600 hover:bg-rose-500 text-white shadow-sm",
    ghost: "text-slate-400 hover:text-white hover:bg-slate-800/60"
  };
  return (
    <button className={`${base} ${styles[variant]} ${className || ''}`} disabled={isLoading || props.disabled} {...props}>
      {isLoading ? <span className="animate-spin mr-2">⟳</span> : null}
      {children}
    </button>
  );
};
