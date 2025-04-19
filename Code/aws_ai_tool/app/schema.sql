-- Drop tables if they exist to ensure a clean slate on initialization
DROP TABLE IF EXISTS metrics;
DROP TABLE IF EXISTS experiments;

-- Table to store information about each training experiment/job
CREATE TABLE experiments (
    experiment_id INTEGER PRIMARY KEY AUTOINCREMENT, -- Unique ID for the experiment
    name TEXT NOT NULL,                             -- User-defined name for the experiment
    model_script TEXT NOT NULL,                     -- Filename of the uploaded model script used
    instance_type TEXT,                             -- AWS EC2 instance type used (e.g., g4dn.xlarge)
    data_source TEXT,                               -- Description or path of the local data used
    start_time TIMESTAMP,                           -- Timestamp when the job was initiated
    end_time TIMESTAMP,                             -- Timestamp when the job finished (completed or failed)
    status TEXT NOT NULL DEFAULT 'Pending',         -- Current status (e.g., Pending, Setting Up, Transferring Data, Training, Downloading Results, Completed, Failed)
    aws_instance_id TEXT,                           -- AWS EC2 instance ID
    hyperparameters TEXT,                           -- JSON string or simple text of hyperparameters used
    results_path TEXT,                              -- Local path where results (model, logs) are stored after download
    error_message TEXT,                             -- Stores specific error details if the job fails
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP -- Timestamp when the record was created
);

-- Table to store time-series metrics for each experiment (e.g., loss per epoch)
CREATE TABLE metrics (
    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,    -- Unique ID for the metric entry
    experiment_id INTEGER NOT NULL,                 -- Foreign key linking to the experiments table
    epoch INTEGER,                                  -- Training epoch number (if applicable)
    step INTEGER,                                   -- Training step number (if applicable)
    metric_name TEXT NOT NULL,                      -- Name of the metric (e.g., 'loss', 'psnr', 'ssim')
    metric_value REAL NOT NULL,                     -- The actual value of the metric
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, -- Timestamp when the metric was recorded
    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id) ON DELETE CASCADE -- Ensure metrics are deleted if the parent experiment is deleted
);

-- Optional: Indexes for faster querying
CREATE INDEX IF NOT EXISTS idx_experiment_status ON experiments (status);
CREATE INDEX IF NOT EXISTS idx_metric_experiment ON metrics (experiment_id);
CREATE INDEX IF NOT EXISTS idx_metric_name ON metrics (metric_name);
