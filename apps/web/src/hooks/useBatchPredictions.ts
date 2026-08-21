import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import api from '@/services/api';

// useBatchPredictions: Batch inference job upload and progress polling hook.
export const useBatchPredictions = () => {
  const queryClient = useQueryClient();

  const getItems = useQuery({
    queryKey: ['useBatchPredictions'],
    queryFn: async () => {
      const res = await api.get('/batchpredictions');
      return res.data;
    }
  });

  const createItem = useMutation({
    mutationFn: async (payload: any) => {
      const res = await api.post('/batchpredictions', payload);
      return res.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['useBatchPredictions'] });
    }
  });

  return {
    items: getItems.data?.items || [],
    isLoading: getItems.isLoading,
    createItem: createItem.mutateAsync,
    isCreating: createItem.isPending
  };
};
