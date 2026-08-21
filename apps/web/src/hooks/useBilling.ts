import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import api from '@/services/api';

// useBilling: Customer billing lifecycle and invoice download hook.
export const useBilling = () => {
  const queryClient = useQueryClient();

  const getItems = useQuery({
    queryKey: ['useBilling'],
    queryFn: async () => {
      const res = await api.get('/billing');
      return res.data;
    }
  });

  const createItem = useMutation({
    mutationFn: async (payload: any) => {
      const res = await api.post('/billing', payload);
      return res.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['useBilling'] });
    }
  });

  return {
    items: getItems.data?.items || [],
    isLoading: getItems.isLoading,
    createItem: createItem.mutateAsync,
    isCreating: createItem.isPending
  };
};
