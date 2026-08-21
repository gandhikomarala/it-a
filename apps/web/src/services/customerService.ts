import api from './api';
import { Customer, CustomerSegmentationSummary } from '@/types';

export const customerService = {
  getCustomers: async (params?: any) => {
    const res = await api.get('/customers', { params });
    return res.data;
  },
  getCustomerById: async (id: string): Promise<Customer> => {
    const res = await api.get(`/customers/${id}`);
    return res.data;
  },
  getSegmentationSummary: async (): Promise<CustomerSegmentationSummary> => {
    const res = await api.get('/customers/summary/segmentation');
    return res.data;
  }
};
