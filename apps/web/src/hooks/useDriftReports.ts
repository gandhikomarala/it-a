import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import api from '@/services/api';

// useDriftReports: Statistical PSI and KS drift report inspection hook.
export const useDriftReports = () => {
  const queryClient = useQueryClient();

  const getItems = useQuery({
    queryKey: ['useDriftReports'],
    queryFn: async () => {
      const res = await api.get('/driftreports');
      return res.data;
    }
  });

  const createItem = useMutation({
    mutationFn: async (payload: any) => {
      const res = await api.post('/driftreports', payload);
      return res.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['useDriftReports'] });
    }
  });

  return {
    items: getItems.data?.items || [],
    isLoading: getItems.isLoading,
    createItem: createItem.mutateAsync,
    isCreating: createItem.isPending
  };
};
