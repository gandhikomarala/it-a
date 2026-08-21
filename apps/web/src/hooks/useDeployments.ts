import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import api from '@/services/api';

// useDeployments: Model promotion, canary splitting, and rollback hook.
export const useDeployments = () => {
  const queryClient = useQueryClient();

  const getItems = useQuery({
    queryKey: ['useDeployments'],
    queryFn: async () => {
      const res = await api.get('/deployments');
      return res.data;
    }
  });

  const createItem = useMutation({
    mutationFn: async (payload: any) => {
      const res = await api.post('/deployments', payload);
      return res.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['useDeployments'] });
    }
  });

  return {
    items: getItems.data?.items || [],
    isLoading: getItems.isLoading,
    createItem: createItem.mutateAsync,
    isCreating: createItem.isPending
  };
};
