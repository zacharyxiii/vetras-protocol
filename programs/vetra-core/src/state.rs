use borsh::{BorshDeserialize, BorshSerialize};
use solana_program::{
    clock::UnixTimestamp,
    program_error::ProgramError,
    pubkey::Pubkey,
};
use std::collections::HashMap;

/// Maximum size for model metadata string
pub const MAX_METADATA_LENGTH: usize = 1024;
/// Maximum number of validators per model
pub const MAX_VALIDATORS: usize = 100;
/// Minimum number of validators required for consensus
pub const MIN_VALIDATORS_CONSENSUS: usize = 3;
/// Validation timeout in seconds
pub const VALIDATION_TIMEOUT: i64 = 3600; // 1 hour

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct ModelSubmission {
    /// Owner of the submitted model
    pub owner: Pubkey,
    /// Type of AI model
    pub model_type: ModelType,
    /// Model metadata in JSON format
    pub metadata: String,
    /// Current validation status
    pub status: ValidationStatus,
    /// Submission timestamp
    pub created_at: UnixTimestamp,
    /// Last updated timestamp
    pub updated_at: UnixTimestamp,
    /// Model's unique identifier (hash)
    pub model_hash: [u8; 32],
    /// Storage location details
    pub storage_info: StorageInfo,
    /// Current validation round
    pub validation_round: u64,
    /// Validation metrics
    pub metrics: Option<ValidationMetrics>,
    /// Access control settings
    pub access_control: AccessControl,
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct StorageInfo {
    /// Storage protocol (IPFS, Arweave, etc.)
    pub protocol: StorageProtocol,
    /// Content identifier/address
    pub identifier: String,
    /// Content size in bytes
    pub size: u64,
    /// Content checksum
    pub checksum: [u8; 32],
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub enum StorageProtocol {
    IPFS,
    Arweave,
    FileCoin,
    Custom(String),
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub enum ModelType {
    LLM {
        architecture: String,
        parameter_count: u64,
        context_window: u32,
    },
    ComputerVision {
        architecture: String,
        input_resolution: (u32, u32),
        model_family: String,
    },
    TabularPredictor {
        framework: String,
        input_features: u32,
        model_type: String,
    },
    Custom {
        category: String,
        specifications: String,
    },
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub enum ValidationStatus {
    Pending,
    InProgress {
        started_at: UnixTimestamp,
        validator_count: u32,
    },
    Completed {
        completed_at: UnixTimestamp,
        result: ValidationResult,
    },
    Failed {
        error_code: u32,
        error_message: String,
        failed_at: UnixTimestamp,
    },
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct ValidationResult {
    /// Overall validation score (0-100)
    pub score: u8,
    /// Detailed validation metrics
    pub metrics: ValidationMetrics,
    /// Consensus details
    pub consensus: ConsensusInfo,
    /// Validator signatures
    pub signatures: Vec<ValidatorSignature>,
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct ValidationMetrics {
    /// Performance metrics
    pub performance: PerformanceMetrics,
    /// Safety evaluation results
    pub safety: SafetyMetrics,
    /// Bias detection results
    pub bias: BiasMetrics,
    /// Resource usage metrics
    pub resources: ResourceMetrics,
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct PerformanceMetrics {
    /// Response latency (ms)
    pub latency: u32,
    /// Throughput (requests/second)
    pub throughput: u32,
    /// Error rate (percentage)
    pub error_rate: f32,
    /// Custom metrics
    pub custom: HashMap<String, f64>,
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct SafetyMetrics {
    /// Content safety score (0-100)
    pub safety_score: u8,
    /// Identified safety concerns
    pub concerns: Vec<SafetyConcern>,
    /// Mitigation recommendations
    pub recommendations: Vec<String>,
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct BiasMetrics {
    /// Overall bias score (0-100, lower is better)
    pub bias_score: u8,
    /// Detected bias categories
    pub detected_biases: Vec<BiasCategory>,
    /// Bias assessment confidence
    pub confidence: f32,
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct ResourceMetrics {
    /// Memory usage (MB)
    pub memory_usage: u64,
    /// CPU utilization (percentage)
    pub cpu_utilization: f32,
    /// GPU utilization (percentage)
    pub gpu_utilization: Option<f32>,
    /// Network bandwidth (MB/s)
    pub bandwidth: f32,
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub enum SafetyConcern {
    Harmful,
    Unethical,
    Biased,
    Privacy,
    Security,
    Custom(String),
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub enum BiasCategory {
    Gender,
    Racial,
    Age,
    Cultural,
    Socioeconomic,
    Geographic,
    Custom(String),
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct ConsensusInfo {
    /// Number of validators participated
    pub validator_count: u32,
    /// Consensus threshold achieved
    pub threshold_achieved: bool,
    /// Consensus round number
    pub round: u64,
    /// Timestamp of consensus achievement
    pub timestamp: UnixTimestamp,
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct ValidatorSignature {
    /// Validator's public key
    pub validator: Pubkey,
    /// Signature of the validation result
    pub signature: [u8; 64],
    /// Timestamp of signature
    pub timestamp: UnixTimestamp,
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct AccessControl {
    /// Whether results are public
    pub is_public: bool,
    /// Allowed viewers (empty if public)
    pub allowed_viewers: Vec<Pubkey>,
    /// Access expiration timestamp
    pub expires_at: Option<UnixTimestamp>,
}

impl ModelSubmission {
    pub fn new(
        owner: Pubkey,
        model_type: ModelType,
        metadata: String,
        model_hash: [u8; 32],
        storage_info: StorageInfo,
        access_control: AccessControl,
    ) -> Result<Self, ProgramError> {
        if metadata.len() > MAX_METADATA_LENGTH {
            return Err(ProgramError::InvalidArgument);
        }

        let now = solana_program::clock::Clock::get()?.unix_timestamp;

        Ok(Self {
            owner,
            model_type,
            metadata,
            status: ValidationStatus::Pending,
            created_at: now,
            updated_at: now,
            model_hash,
            storage_info,
            validation_round: 0,
            metrics: None,
            access_control,
        })
    }

    pub fn update_status(&mut self, new_status: ValidationStatus) -> Result<(), ProgramError> {
        self.status = new_status;
        self.updated_at = solana_program::clock::Clock::get()?.unix_timestamp;
        Ok(())
    }

    pub fn is_timeout(&self) -> Result<bool, ProgramError> {
        let now = solana_program::clock::Clock::get()?.unix_timestamp;
        
        if let ValidationStatus::InProgress { started_at, .. } = self.status {
            Ok(now - started_at > VALIDATION_TIMEOUT)
        } else {
            Ok(false)
        }
    }

    pub fn can_access(&self, viewer: &Pubkey) -> bool {
        if self.access_control.is_public {
            return true;
        }

        if &self.owner == viewer {
            return true;
        }

        if let Some(expires_at) = self.access_control.expires_at {
            if let Ok(now) = solana_program::clock::Clock::get() {
                if now.unix_timestamp >= expires_at {
                    return false;
                }
            }
        }

        self.access_control.allowed_viewers.contains(viewer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use solana_program::clock::Clock;
    use std::collections::HashMap;

    fn create_test_submission() -> ModelSubmission {
        let owner = Pubkey::new_unique();
        let model_type = ModelType::LLM {
            architecture: "transformer".to_string(),
            parameter_count: 1_000_000,
            context_window: 2048,
        };
        let storage_info = StorageInfo {
            protocol: StorageProtocol::IPFS,
            identifier: "QmTest123".to_string(),
            size: 1000,
            checksum: [0; 32],
        };
        let access_control = AccessControl {
            is_public: false,
            allowed_viewers: vec![],
            expires_at: None,
        };

        ModelSubmission::new(
            owner,
            model_type,
            "test metadata".to_string(),
            [0; 32],
            storage_info,
            access_control,
        )
        .unwrap()
    }

    #[test]
    fn test_model_submission_creation() {
        let submission = create_test_submission();
        assert_eq!(submission.metadata, "test metadata");
        assert!(matches!(submission.status, ValidationStatus::Pending));
    }

    #[test]
    fn test_validation_metrics() {
        let metrics = ValidationMetrics {
            performance: PerformanceMetrics {
                latency: 100,
                throughput: 1000,
                error_rate: 0.01,
                custom: HashMap::new(),
            },
            safety: SafetyMetrics {
                safety_score: 95,
                concerns: vec![],
                recommendations: vec![],
            },
            bias: BiasMetrics {
                bias_score: 10,
                detected_biases: vec![],
                confidence: 0.95,
            },
            resources: ResourceMetrics {
                memory_usage: 1024,
                cpu_utilization: 75.0,
                gpu_utilization: Some(80.0),
                bandwidth: 100.0,
            },
        };

        assert_eq!(metrics.performance.latency, 100);
        assert_eq!(metrics.safety.safety_score, 95);
        assert_eq!(metrics.bias.confidence, 0.95);
    }

    #[test]
    fn test_access_control() {
        let mut submission = create_test_submission();
        let viewer = Pubkey::new_unique();
        
        // Test public access
        submission.access_control.is_public = true;
        assert!(submission.can_access(&viewer));

        // Test private access
        submission.access_control.is_public = false;
        assert!(!submission.can_access(&viewer));

        // Test allowed viewer
        submission.access_control.allowed_viewers.push(viewer);
        assert!(submission.can_access(&viewer));
    }

    #[test]
    fn test_status_update() {
        let mut submission = create_test_submission();
        let new_status = ValidationStatus::InProgress {
            started_at: 1000,
            validator_count: 5,
        };
        
        submission.update_status(new_status.clone()).unwrap();
        assert!(matches!(
            submission.status,
            ValidationStatus::InProgress { .. }
        ));
    }
}
