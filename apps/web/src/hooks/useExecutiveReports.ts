import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import api from '@/services/api';

// useExecutiveReports: Executive BI dashboard and PDF/CSV report generation hook.
export const useExecutiveReports = () => {
  const queryClient = useQueryClient();

  const getItems = useQuery({
    queryKey: ['useExecutiveReports'],
    queryFn: async () => {
      const res = await api.get('/executivereports');
      return res.data;
    }
  });

  const createItem = useMutation({
    mutationFn: async (payload: any) => {
      const res = await api.post('/executivereports', payload);
      return res.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['useExecutiveReports'] });
    }
  });

  return {
    items: getItems.data?.items || [],
    isLoading: getItems.isLoading,
    createItem: createItem.mutateAsync,
    isCreating: createItem.isPending
  };
};
