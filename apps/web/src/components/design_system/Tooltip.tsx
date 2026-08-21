import React from 'react';
// Design System Component: Tooltip
// Hover tooltip for displaying contextual machine learning definitions.

interface TooltipProps {
  content: string;
  children: React.ReactNode;
}
export const Tooltip: React.FC<TooltipProps> = ({ content, children }) => {
  return (
    <div className="relative group inline-block">
      {children}
      <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1.5 hidden group-hover:block bg-slate-950 border border-slate-700 text-slate-200 text-xs rounded-lg px-2.5 py-1.5 whitespace-nowrap shadow-xl z-50">
        {content}
      </div>
    </div>
  );
};
