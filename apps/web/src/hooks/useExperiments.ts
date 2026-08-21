import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import api from '@/services/api';

// useExperiments: ML training experiment comparison and run monitoring hook.
export const useExperiments = () => {
  const queryClient = useQueryClient();

  const getItems = useQuery({
    queryKey: ['useExperiments'],
    queryFn: async () => {
      const res = await api.get('/experiments');
      return res.data;
    }
  });

  const createItem = useMutation({
    mutationFn: async (payload: any) => {
      const res = await api.post('/experiments', payload);
      return res.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['useExperiments'] });
    }
  });

  return {
    items: getItems.data?.items || [],
    isLoading: getItems.isLoading,
    createItem: createItem.mutateAsync,
    isCreating: createItem.isPending
  };
};
