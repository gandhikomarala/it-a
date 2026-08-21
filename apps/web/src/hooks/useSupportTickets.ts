import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import api from '@/services/api';

// useSupportTickets: Support ticket creation, escalation, and SLA tracking hook.
export const useSupportTickets = () => {
  const queryClient = useQueryClient();

  const getItems = useQuery({
    queryKey: ['useSupportTickets'],
    queryFn: async () => {
      const res = await api.get('/supporttickets');
      return res.data;
    }
  });

  const createItem = useMutation({
    mutationFn: async (payload: any) => {
      const res = await api.post('/supporttickets', payload);
      return res.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['useSupportTickets'] });
    }
  });

  return {
    items: getItems.data?.items || [],
    isLoading: getItems.isLoading,
    createItem: createItem.mutateAsync,
    isCreating: createItem.isPending
  };
};
