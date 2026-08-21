import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import api from '@/services/api';

// useSystemSettings: Platform configuration and risk threshold management hook.
export const useSystemSettings = () => {
  const queryClient = useQueryClient();

  const getItems = useQuery({
    queryKey: ['useSystemSettings'],
    queryFn: async () => {
      const res = await api.get('/systemsettings');
      return res.data;
    }
  });

  const createItem = useMutation({
    mutationFn: async (payload: any) => {
      const res = await api.post('/systemsettings', payload);
      return res.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['useSystemSettings'] });
    }
  });

  return {
    items: getItems.data?.items || [],
    isLoading: getItems.isLoading,
    createItem: createItem.mutateAsync,
    isCreating: createItem.isPending
  };
};
