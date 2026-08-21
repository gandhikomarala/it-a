# Enterprise Customer Churn Prediction & MLOps Platform Architecture

## System Overview
The platform is designed as a high-scale, production-grade MLOps system providing real-time churn risk scoring, explainability, statistical drift detection, automated retraining, and enterprise analytics.

```
+-------------------------------------------------------------+
|                      React + Vite UI                        |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                     FastAPI REST Gateway                    |
+--------------+---------------+--------------+---------------+
               |               |              |
               v               v              v
        +--------------+ +-----------+ +--------------+
        | PostgreSQL 16| |  Redis 7  | | Celery Worker|
        +--------------+ +-----------+ +------+-------+
                                              |
                                              v
                                       +--------------+
                                       |  ML Engine   |
                                       +--------------+
```

## Key Modules
1. **ML Engine (`ml/`)**: Custom Scikit-learn transformers, LightGBM/RandomForest wrappers, SHAP factor attribution, PSI drift detection.
2. **Backend API (`backend/`, `apps/api/`)**: SQLAlchemy 2.0 async ORM, 32 relational tables, RBAC permission guards, JWT rotation.
3. **Frontend Dashboard (`apps/web/`)**: React 18, Tailwind CSS, TanStack Query, Recharts visualizations.
