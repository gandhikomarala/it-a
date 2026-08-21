import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import api from '@/services/api';

// useCustomerEvents: Real-time customer event streaming and timeline hook.
export const useCustomerEvents = () => {
  const queryClient = useQueryClient();

  const getItems = useQuery({
    queryKey: ['useCustomerEvents'],
    queryFn: async () => {
      const res = await api.get('/customerevents');
      return res.data;
    }
  });

  const createItem = useMutation({
    mutationFn: async (payload: any) => {
      const res = await api.post('/customerevents', payload);
      return res.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['useCustomerEvents'] });
    }
  });

  return {
    items: getItems.data?.items || [],
    isLoading: getItems.isLoading,
    createItem: createItem.mutateAsync,
    isCreating: createItem.isPending
  };
};
