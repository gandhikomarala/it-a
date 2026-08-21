import api from './api';
import { PredictionResult } from '@/types';

export const predictionService = {
  predictSingle: async (payload: any): Promise<PredictionResult> => {
    const res = await api.post('/predictions', payload);
    return res.data;
  }
};
