use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use anyhow::{Result, Context, anyhow};
use serde::{Serialize, Deserialize};
use rocksdb::{DB, ColumnFamily, Options, WriteBatch, ReadOptions, WriteOptions};
use ed25519_dalek::PublicKey;
use metrics::{counter, gauge, histogram};
use chrono::{DateTime, Utc};
use bincode;
use blake3;

// Database column families
const CF_MODELS: &str = "models";
const CF_VALIDATIONS: &str = "validations";
const CF_VALIDATORS: &str = "validators";
const CF_METRICS: &str = "metrics";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredModel {
    pub id: String,
    pub hash: [u8; 32],
    pub size: u64,
    pub format: String,
    pub created_at: DateTime<Utc>,
    pub metadata: ModelMetadata,
    pub status: ModelStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub name: String,
    pub version: String,
    pub framework: String,
    pub parameters: u64,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelStatus {
    Pending,
    Validating,
    Valid,
    Invalid,
    Error(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRecord {
    pub id: String,
    pub model_id: String,
    pub validator: PublicKey,
    pub timestamp: DateTime<Utc>,
    pub results: ValidationResults,
    pub consensus: Option<ConsensusData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResults {
    pub accuracy: f64,
    pub latency_ms: f64,
    pub memory_usage_mb: f64,
    pub throughput: f64,
    pub error_rate: f64,
    pub metrics: ValidationMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationMetrics {
    pub inference_time: Vec<f64>,
    pub memory_profile: Vec<(DateTime<Utc>, f64)>,
    pub cpu_usage: Vec<(DateTime<Utc>, f64)>,
    pub gpu_usage: Option<Vec<(DateTime<Utc>, f64)>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusData {
    pub round: u64,
    pub validators: Vec<PublicKey>,
    pub signatures: Vec<(PublicKey, Vec<u8>)>,
    pub timestamp: DateTime<Utc>,
}

pub struct Database {
    db: Arc<DB>,
    path: PathBuf,
    options: Options,
}

impl Database {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut options = Options::default();
        
        options.create_if_missing(true);
        options.create_missing_column_families(true);
        options.set_max_open_files(10000);
        options.set_keep_log_file_num(10);
        options.set_max_total_wal_size(536870912); // 512MB
        options.set_compression_type(rocksdb::DBCompressionType::Lz4);
        
        // Create column families if they don't exist
        let cfs = vec![CF_MODELS, CF_VALIDATIONS, CF_VALIDATORS, CF_METRICS];
        let db = DB::open_cf(&options, &path, cfs)
            .context("Failed to open database")?;
            
        Ok(Self {
            db: Arc::new(db),
            path,
            options,
        })
    }

    pub fn store_model(&self, model: &StoredModel) -> Result<()> {
        let cf = self.get_cf(CF_MODELS)?;
        let key = model.id.as_bytes();
        let value = bincode::serialize(model)?;
        
        let mut batch = WriteBatch::default();
        batch.put_cf(cf, key, value);
        
        let mut write_opts = WriteOptions::default();
        write_opts.set_sync(true);
        
        self.db.write_opt(batch, &write_opts)
            .context("Failed to store model")?;
            
        counter!("vetras.storage.models_stored", 1);
        Ok(())
    }

    pub fn get_model(&self, id: &str) -> Result<Option<StoredModel>> {
        let cf = self.get_cf(CF_MODELS)?;
        
        if let Some(data) = self.db.get_cf(cf, id.as_bytes())
            .context("Failed to read model")? {
            let model: StoredModel = bincode::deserialize(&data)?;
            Ok(Some(model))
        } else {
            Ok(None)
        }
    }

    pub fn store_validation(&self, validation: &ValidationRecord) -> Result<()> {
        let cf = self.get_cf(CF_VALIDATIONS)?;
        let key = validation.id.as_bytes();
        let value = bincode::serialize(validation)?;
        
        let mut batch = WriteBatch::default();
        batch.put_cf(cf, key, value);
        
        // Update model status
        if let Some(mut model) = self.get_model(&validation.model_id)? {
            model.status = ModelStatus::Valid; // Or based on validation results
            let model_cf = self.get_cf(CF_MODELS)?;
            batch.put_cf(model_cf, model.id.as_bytes(), bincode::serialize(&model)?);
        }
        
        let mut write_opts = WriteOptions::default();
        write_opts.set_sync(true);
        
        self.db.write_opt(batch, &write_opts)
            .context("Failed to store validation")?;
            
        counter!("vetras.storage.validations_stored", 1);
        Ok(())
    }

    pub fn get_validation(&self, id: &str) -> Result<Option<ValidationRecord>> {
        let cf = self.get_cf(CF_VALIDATIONS)?;
        
        if let Some(data) = self.db.get_cf(cf, id.as_bytes())
            .context("Failed to read validation")? {
            let validation: ValidationRecord = bincode::deserialize(&data)?;
            Ok(Some(validation))
        } else {
            Ok(None)
        }
    }

    pub fn get_model_validations(&self, model_id: &str) -> Result<Vec<ValidationRecord>> {
        let cf = self.get_cf(CF_VALIDATIONS)?;
        let mut validations = Vec::new();
        
        let mut opts = ReadOptions::default();
        opts.set_prefix_same_as_start(true);
        
        let iter = self.db.iterator_cf_opt(cf, opts, rocksdb::IteratorMode::Start);
        
        for item in iter {
            let (_, value) = item?;
            let validation: ValidationRecord = bincode::deserialize(&value)?;
            
            if validation.model_id == model_id {
                validations.push(validation);
            }
        }
        
        Ok(validations)
    }

    pub fn update_metrics(&self, metrics: &ValidationMetrics) -> Result<()> {
        let cf = self.get_cf(CF_METRICS)?;
        let key = format!("metrics_{}", Utc::now().timestamp());
        let value = bincode::serialize(metrics)?;
        
        self.db.put_cf(cf, key.as_bytes(), value)
            .context("Failed to store metrics")?;
            
        // Update gauges
        if let Some(gpu_usage) = metrics.gpu_usage.as_ref() {
            if let Some((_, usage)) = gpu_usage.last() {
                gauge!("vetras.validation.gpu_usage", *usage);
            }
        }
        
        if let Some((_, cpu_usage)) = metrics.cpu_usage.last() {
            gauge!("vetras.validation.cpu_usage", *cpu_usage);
        }
        
        Ok(())
    }

    pub fn compute_model_hash(&self, data: &[u8]) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(data);
        *hasher.finalize().as_bytes()
    }

    fn get_cf(&self, name: &str) -> Result<&ColumnFamily> {
        self.db.cf_handle(name)
            .ok_or_else(|| anyhow!("Column family not found: {}", name))
    }

    pub fn backup(&self, backup_path: &Path) -> Result<()> {
        use rocksdb::backup::BackupEngine;
        
        let mut backup_opts = Options::default();
        backup_opts.create_if_missing(true);
        
        let mut backup_engine = BackupEngine::open(&backup_opts, backup_path)
            .context("Failed to create backup engine")?;
            
        backup_engine.create_new_backup(&self.db)
            .context("Failed to create backup")?;
            
        counter!("vetras.storage.backups_created", 1);
        Ok(())
    }
}

impl Drop for Database {
    fn drop(&mut self) {
        if let Err(e) = self.db.flush() {
            error!("Failed to flush database on drop: {}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use ed25519_dalek::Keypair;
    use rand::rngs::OsRng;

    fn create_test_model() -> StoredModel {
        StoredModel {
            id: "test_model".to_string(),
            hash: [0u8; 32],
            size: 1000,
            format: "ONNX".to_string(),
            created_at: Utc::now(),
            metadata: ModelMetadata {
                name: "Test Model".to_string(),
                version: "1.0.0".to_string(),
                framework: "PyTorch".to_string(),
                parameters: 1_000_000,
                input_shape: vec![1, 3, 224, 224],
                output_shape: vec![1, 1000],
                tags: vec!["test".to_string()],
            },
            status: ModelStatus::Pending,
        }
    }

    fn create_test_validation(model_id: &str) -> ValidationRecord {
        let keypair = Keypair::generate(&mut OsRng);
        
        ValidationRecord {
            id: "test_validation".to_string(),
            model_id: model_id.to_string(),
            validator: keypair.public,
            timestamp: Utc::now(),
            results: ValidationResults {
                accuracy: 0.95,
                latency_ms: 10.0,
                memory_usage_mb: 1000.0,
                throughput: 100.0,
                error_rate: 0.05,
                metrics: ValidationMetrics {
                    inference_time: vec![10.0, 11.0, 9.0],
                    memory_profile: vec![(Utc::now(), 1000.0)],
                    cpu_usage: vec![(Utc::now(), 50.0)],
                    gpu_usage: Some(vec![(Utc::now(), 80.0)]),
                },
            },
            consensus: None,
        }
    }

    #[test]
    fn test_database_operations() {
        let temp_dir = TempDir::new().unwrap();
        let db = Database::new(temp_dir.path()).unwrap();
        
        // Test model storage
        let model = create_test_model();
        db.store_model(&model).unwrap();
        
        let retrieved = db.get_model(&model.id).unwrap().unwrap();
        assert_eq!(retrieved.id, model.id);
        
        // Test validation storage
        let validation = create_test_validation(&model.id);
        db.store_validation(&validation).unwrap();
        
        let retrieved = db.get_validation(&validation.id).unwrap().unwrap();
        assert_eq!(retrieved.id, validation.id);
        
        // Test model validations retrieval
        let validations = db.get_model_validations(&model.id).unwrap();
        assert_eq!(validations.len(), 1);
        assert_eq!(validations[0].id, validation.id);
    }

    #[test]
    fn test_backup() {
        let temp_dir = TempDir::new().unwrap();
        let backup_dir = TempDir::new().unwrap();
        let db = Database::new(temp_dir.path()).unwrap();
        
        // Store some data
        let model = create_test_model();
        db.store_model(&model).unwrap();
        
        // Create backup
        db.backup(backup_dir.path()).unwrap();
        
        // Verify backup exists
        assert!(backup_dir.path().join("CURRENT").exists());
    }
}
