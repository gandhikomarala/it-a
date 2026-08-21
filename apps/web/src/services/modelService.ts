import api from './api';
import { MLModel } from '@/types';

export const modelService = {
  getModels: async (): Promise<MLModel[]> => {
    const res = await api.get('/models');
    return res.data;
  },
  promoteModel: async (modelId: string, versionId: string) => {
    const res = await api.post(`/models/${modelId}/promote`, { version_id: versionId });
    return res.data;
  }
};
