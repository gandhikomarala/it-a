import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import api from '@/services/api';

// useRetrainingPolicies: Automated retraining triggers and policy configuration hook.
export const useRetrainingPolicies = () => {
  const queryClient = useQueryClient();

  const getItems = useQuery({
    queryKey: ['useRetrainingPolicies'],
    queryFn: async () => {
      const res = await api.get('/retrainingpolicies');
      return res.data;
    }
  });

  const createItem = useMutation({
    mutationFn: async (payload: any) => {
      const res = await api.post('/retrainingpolicies', payload);
      return res.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['useRetrainingPolicies'] });
    }
  });

  return {
    items: getItems.data?.items || [],
    isLoading: getItems.isLoading,
    createItem: createItem.mutateAsync,
    isCreating: createItem.isPending
  };
};
