import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import api from '@/services/api';

// useWebhooks: Webhook endpoint configuration and delivery retry hook.
export const useWebhooks = () => {
  const queryClient = useQueryClient();

  const getItems = useQuery({
    queryKey: ['useWebhooks'],
    queryFn: async () => {
      const res = await api.get('/webhooks');
      return res.data;
    }
  });

  const createItem = useMutation({
    mutationFn: async (payload: any) => {
      const res = await api.post('/webhooks', payload);
      return res.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['useWebhooks'] });
    }
  });

  return {
    items: getItems.data?.items || [],
    isLoading: getItems.isLoading,
    createItem: createItem.mutateAsync,
    isCreating: createItem.isPending
  };
};
