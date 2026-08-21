import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import api from '@/services/api';

// useAPIKeys: Developer API key generation and scope management hook.
export const useAPIKeys = () => {
  const queryClient = useQueryClient();

  const getItems = useQuery({
    queryKey: ['useAPIKeys'],
    queryFn: async () => {
      const res = await api.get('/apikeys');
      return res.data;
    }
  });

  const createItem = useMutation({
    mutationFn: async (payload: any) => {
      const res = await api.post('/apikeys', payload);
      return res.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['useAPIKeys'] });
    }
  });

  return {
    items: getItems.data?.items || [],
    isLoading: getItems.isLoading,
    createItem: createItem.mutateAsync,
    isCreating: createItem.isPending
  };
};
