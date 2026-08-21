// Central TypeScript domain types.

export type RiskLevel = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
export type ModelStage = 'DEVELOPMENT' | 'VALIDATION' | 'STAGING' | 'PRODUCTION' | 'ARCHIVED' | 'REJECTED';
export type TaskStatus = 'PENDING' | 'QUEUED' | 'PROCESSING' | 'RUNNING' | 'COMPLETED' | 'FAILED' | 'CANCELLED';
export type DriftStatus = 'NORMAL' | 'WARNING' | 'CRITICAL';

export interface User {
  id: string;
  email: string;
  first_name: string;
  last_name: string;
  role: string;
  permissions: string[];
  is_active: boolean;
}

export interface Customer {
  id: string;
  customer_id: string;
  first_name: string;
  last_name: string;
  email: string;
  age: number;
  gender: string;
  region: string;
  city: string;
  income: number;
  signup_date: string;
  subscription_type: string;
  contract_type: string;
  payment_method: string;
  monthly_charge: number;
  tenure_months: number;
  total_spend: number;
  is_active: boolean;
  latest_churn_probability?: number;
  latest_risk_level?: RiskLevel;
  latest_prediction_date?: string;
  tags?: string[];
}

export interface CustomerSegmentationSummary {
  total_customers: number;
  active_customers: number;
  high_risk_count: number;
  medium_risk_count: number;
  low_risk_count: number;
  estimated_revenue_at_risk: number;
  average_customer_tenure: number;
  average_monthly_revenue: number;
}

export interface Dataset {
  id: string;
  name: string;
  description?: string;
  latest_version: number;
  latest_quality_score?: number;
  row_count: number;
  column_count: number;
  tags: string[];
  versions_count: number;
  created_at: string;
}

export interface MLModel {
  id: string;
  name: string;
  description?: string;
  active_production_version?: number;
  active_production_version_id?: string;
  production_roc_auc?: number;
  versions_count: number;
  created_at: string;
}

export interface SHAPContribution {
  feature_name: string;
  display_name: string;
  feature_value: any;
  shap_value: number;
  contribution_percentage: number;
  impact_direction: 'POSITIVE' | 'NEGATIVE';
}

export interface PredictionExplanation {
  customer_id: string;
  base_value: number;
  prediction_probability: number;
  top_positive_factors: SHAPContribution[];
  top_negative_factors: SHAPContribution[];
  all_contributions: SHAPContribution[];
  summary_text: string;
}

export interface PredictionResult {
  customer_id: string;
  prediction: number;
  churn_probability: number;
  risk_level: RiskLevel;
  confidence: number;
  model_id: string;
  model_version: string;
  prediction_timestamp: string;
  explanation?: PredictionExplanation;
}

export interface AnalyticsKPIs {
  total_customers: number;
  active_customers: number;
  overall_churn_rate_pct: number;
  high_risk_customers_count: number;
  estimated_revenue_at_risk_monthly: number;
  estimated_revenue_at_risk_annual: number;
  average_customer_lifetime_value: number;
  net_retention_rate_pct: number;
}

export interface ChurnTrendPoint {
  date: string;
  total_customers: number;
  predicted_churners: number;
  actual_churners?: number;
  churn_rate_pct: number;
}

export interface ChurnSegment {
  segment_name: string;
  customer_count: number;
  churn_rate_pct: number;
  revenue_at_risk: number;
}

export interface AnalyticsDashboard {
  kpis: AnalyticsKPIs;
  churn_trends: ChurnTrendPoint[];
  revenue_risk: {
    by_subscription_tier: ChurnSegment[];
    by_contract_type: ChurnSegment[];
  };
  cohort_retention: {
    cohort_months: string[];
    matrix: number[][];
  };
}
