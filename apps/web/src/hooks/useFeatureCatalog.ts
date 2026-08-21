import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import api from '@/services/api';

// useFeatureCatalog: Feature store catalog and lineage inspection hook.
export const useFeatureCatalog = () => {
  const queryClient = useQueryClient();

  const getItems = useQuery({
    queryKey: ['useFeatureCatalog'],
    queryFn: async () => {
      const res = await api.get('/featurecatalog');
      return res.data;
    }
  });

  const createItem = useMutation({
    mutationFn: async (payload: any) => {
      const res = await api.post('/featurecatalog', payload);
      return res.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['useFeatureCatalog'] });
    }
  });

  return {
    items: getItems.data?.items || [],
    isLoading: getItems.isLoading,
    createItem: createItem.mutateAsync,
    isCreating: createItem.isPending
  };
};
