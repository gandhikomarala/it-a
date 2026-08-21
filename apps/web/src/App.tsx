import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AppLayout } from './components/layout/AppLayout';
import { LoginPage } from './pages/LoginPage';
import { DashboardPage } from './pages/DashboardPage';
import { CustomersPage } from './pages/CustomersPage';
import { DatasetsPage } from './pages/DatasetsPage';
import { ModelsPage } from './pages/ModelsPage';
import { PredictionsPage } from './pages/PredictionsPage';
import { DriftPage } from './pages/DriftPage';
import { AnalyticsPage } from './pages/AnalyticsPage';

const queryClient = new QueryClient();

export const App: React.FC = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/" element={<AppLayout />}>
            <Route index element={<Navigate to="/dashboard" replace />} />
            <Route path="dashboard" element={<DashboardPage />} />
            <Route path="customers" element={<CustomersPage />} />
            <Route path="datasets" element={<DatasetsPage />} />
            <Route path="features" element={<DashboardPage />} />
            <Route path="experiments" element={<ModelsPage />} />
            <Route path="models" element={<ModelsPage />} />
            <Route path="predictions" element={<PredictionsPage />} />
            <Route path="drift" element={<DriftPage />} />
            <Route path="analytics" element={<AnalyticsPage />} />
            <Route path="reports" element={<AnalyticsPage />} />
            <Route path="audit" element={<CustomersPage />} />
            <Route path="settings" element={<DashboardPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
};

export default App;
