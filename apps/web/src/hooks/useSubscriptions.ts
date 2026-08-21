import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import api from '@/services/api';

// useSubscriptions: Subscription upgrade, downgrade, and contract renewal hook.
export const useSubscriptions = () => {
  const queryClient = useQueryClient();

  const getItems = useQuery({
    queryKey: ['useSubscriptions'],
    queryFn: async () => {
      const res = await api.get('/subscriptions');
      return res.data;
    }
  });

  const createItem = useMutation({
    mutationFn: async (payload: any) => {
      const res = await api.post('/subscriptions', payload);
      return res.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['useSubscriptions'] });
    }
  });

  return {
    items: getItems.data?.items || [],
    isLoading: getItems.isLoading,
    createItem: createItem.mutateAsync,
    isCreating: createItem.isPending
  };
};
