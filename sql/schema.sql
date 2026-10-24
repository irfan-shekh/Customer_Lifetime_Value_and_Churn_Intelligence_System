-- Customer Lifetime Value and Churn Intelligence System
-- Normalized PostgreSQL Database Schema

-- 1. Model Execution Tracking
CREATE TABLE IF NOT EXISTS model_runs (
    model_run_id VARCHAR(64) PRIMARY KEY,
    model_type VARCHAR(64) NOT NULL,
    model_version VARCHAR(32) NOT NULL,
    threshold NUMERIC(5, 4),
    metrics JSONB,
    hyperparameters JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. Core Customers Entity
CREATE TABLE IF NOT EXISTS customers (
    customer_id VARCHAR(64) PRIMARY KEY,
    gender VARCHAR(16),
    senior_citizen INTEGER DEFAULT 0 CHECK (senior_citizen IN (0, 1)),
    partner VARCHAR(8),
    dependents VARCHAR(8),
    phone_service VARCHAR(8),
    multiple_lines VARCHAR(32),
    internet_service VARCHAR(32),
    online_security VARCHAR(32),
    online_backup VARCHAR(32),
    device_protection VARCHAR(32),
    tech_support VARCHAR(32),
    streaming_tv VARCHAR(32),
    streaming_movies VARCHAR(32),
    contract VARCHAR(32),
    paperless_billing VARCHAR(8),
    payment_method VARCHAR(64),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. Temporal Snapshot / Current Account Status
CREATE TABLE IF NOT EXISTS customer_snapshots (
    snapshot_id BIGSERIAL PRIMARY KEY,
    customer_id VARCHAR(64) NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    tenure INTEGER NOT NULL CHECK (tenure >= 0),
    monthly_charges NUMERIC(10, 2) NOT NULL CHECK (monthly_charges >= 0),
    total_charges NUMERIC(12, 2) NOT NULL CHECK (total_charges >= 0),
    churn_status INTEGER NOT NULL CHECK (churn_status IN (0, 1)),
    churn_reason VARCHAR(255),
    snapshot_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_customer_snapshot UNIQUE (customer_id, snapshot_date)
);

-- 4. Churn Risk Predictions
CREATE TABLE IF NOT EXISTS churn_predictions (
    prediction_id BIGSERIAL PRIMARY KEY,
    customer_id VARCHAR(64) NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    model_run_id VARCHAR(64) REFERENCES model_runs(model_run_id),
    churn_probability NUMERIC(5, 4) NOT NULL CHECK (churn_probability >= 0.0 AND churn_probability <= 1.0),
    predicted_churn INTEGER NOT NULL CHECK (predicted_churn IN (0, 1)),
    risk_tier VARCHAR(16) NOT NULL CHECK (risk_tier IN ('Low', 'Medium', 'High', 'Critical')),
    top_churn_drivers JSONB,
    predicted_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 5. Survival Analysis Horizon Predictions
CREATE TABLE IF NOT EXISTS survival_predictions (
    survival_id BIGSERIAL PRIMARY KEY,
    customer_id VARCHAR(64) NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    model_run_id VARCHAR(64) REFERENCES model_runs(model_run_id),
    median_remaining_tenure NUMERIC(6, 2),
    prob_active_12m NUMERIC(5, 4),
    prob_active_24m NUMERIC(5, 4),
    prob_active_36m NUMERIC(5, 4),
    prob_active_48m NUMERIC(5, 4),
    predicted_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 6. Customer Lifetime Value (CLV)
CREATE TABLE IF NOT EXISTS customer_lifetime_values (
    clv_id BIGSERIAL PRIMARY KEY,
    customer_id VARCHAR(64) NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    model_run_id VARCHAR(64) REFERENCES model_runs(model_run_id),
    forecast_horizon_months INTEGER NOT NULL,
    contribution_margin_pct NUMERIC(5, 4) NOT NULL,
    annual_discount_rate NUMERIC(5, 4) NOT NULL,
    expected_future_revenue NUMERIC(12, 2) NOT NULL,
    expected_future_margin NUMERIC(12, 2) NOT NULL,
    discounted_clv NUMERIC(12, 2) NOT NULL,
    revenue_at_risk NUMERIC(12, 2) NOT NULL,
    calculated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 7. Customer Behavioral Segments
CREATE TABLE IF NOT EXISTS customer_segments (
    segment_id BIGSERIAL PRIMARY KEY,
    customer_id VARCHAR(64) NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    model_run_id VARCHAR(64) REFERENCES model_runs(model_run_id),
    cluster_id INTEGER NOT NULL,
    cluster_label VARCHAR(64) NOT NULL,
    cluster_description TEXT,
    assigned_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 8. Retention Campaigns & Scenario Simulations
CREATE TABLE IF NOT EXISTS retention_campaigns (
    campaign_id VARCHAR(64) PRIMARY KEY,
    campaign_name VARCHAR(128) NOT NULL,
    target_risk_tier VARCHAR(32) NOT NULL,
    discount_pct NUMERIC(5, 4) NOT NULL,
    discount_duration_months INTEGER NOT NULL,
    offer_acceptance_rate NUMERIC(5, 4) NOT NULL,
    expected_uplift_pct NUMERIC(5, 4) NOT NULL,
    budget_limit NUMERIC(12, 2),
    eligible_customers INTEGER NOT NULL,
    estimated_retained_customers INTEGER NOT NULL,
    campaign_cost NUMERIC(12, 2) NOT NULL,
    retained_contribution NUMERIC(12, 2) NOT NULL,
    net_benefit NUMERIC(12, 2) NOT NULL,
    roi_pct NUMERIC(8, 2) NOT NULL,
    simulated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 9. Campaign Customer Assignments
CREATE TABLE IF NOT EXISTS campaign_assignments (
    assignment_id BIGSERIAL PRIMARY KEY,
    campaign_id VARCHAR(64) NOT NULL REFERENCES retention_campaigns(campaign_id) ON DELETE CASCADE,
    customer_id VARCHAR(64) NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    assigned_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_campaign_assignment UNIQUE (campaign_id, customer_id)
);

-- Indexes for high-frequency queries
CREATE INDEX IF NOT EXISTS idx_customers_contract ON customers(contract);
CREATE INDEX IF NOT EXISTS idx_snapshots_tenure ON customer_snapshots(tenure);
CREATE INDEX IF NOT EXISTS idx_snapshots_churn ON customer_snapshots(churn_status);
CREATE INDEX IF NOT EXISTS idx_churn_preds_prob ON churn_predictions(churn_probability DESC);
CREATE INDEX IF NOT EXISTS idx_churn_preds_tier ON churn_predictions(risk_tier);
CREATE INDEX IF NOT EXISTS idx_clv_discounted ON customer_lifetime_values(discounted_clv DESC);
CREATE INDEX IF NOT EXISTS idx_segments_cluster ON customer_segments(cluster_label);
