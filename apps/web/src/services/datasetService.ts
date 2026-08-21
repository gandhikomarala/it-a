import api from './api';
import { Dataset } from '@/types';

export const datasetService = {
  getDatasets: async (page = 1, pageSize = 20) => {
    const res = await api.get('/datasets', { params: { page, page_size: pageSize } });
    return res.data;
  },
  uploadDataset: async (formData: FormData) => {
    const res = await api.post('/datasets/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
    return res.data;
  }
};
