import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import api from '@/services/api';

// useAuditLogs: Compliance audit trail filtering and CSV export hook.
export const useAuditLogs = () => {
  const queryClient = useQueryClient();

  const getItems = useQuery({
    queryKey: ['useAuditLogs'],
    queryFn: async () => {
      const res = await api.get('/auditlogs');
      return res.data;
    }
  });

  const createItem = useMutation({
    mutationFn: async (payload: any) => {
      const res = await api.post('/auditlogs', payload);
      return res.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['useAuditLogs'] });
    }
  });

  return {
    items: getItems.data?.items || [],
    isLoading: getItems.isLoading,
    createItem: createItem.mutateAsync,
    isCreating: createItem.isPending
  };
};
