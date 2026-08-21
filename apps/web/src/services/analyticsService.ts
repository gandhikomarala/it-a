import api from './api';
import { AnalyticsDashboard } from '@/types';

export const analyticsService = {
  getDashboardAnalytics: async (): Promise<AnalyticsDashboard> => {
    const res = await api.get('/analytics/dashboard');
    return res.data;
  }
};
