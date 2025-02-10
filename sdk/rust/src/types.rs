use std::collections::HashMap;
use borsh::{BorshDeserialize, BorshSerialize};
use serde::{Deserialize, Serialize};
use solana_program::{
    clock::UnixTimestamp,
    pubkey::Pubkey,
};

/// Model submission data structure
#[derive(Debug, Clone, BorshSerialize, BorshDeserialize, Serialize, Deserialize)]
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

/// Types of AI models supported by the platform
#[derive(Debug, Clone, BorshSerialize, BorshDeserialize, Serialize, Deserialize)]
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

/// Storage information for model artifacts
#[derive(Debug, Clone, BorshSerialize, BorshDeserialize, Serialize, Deserialize)]
pub struct StorageInfo {
    /// Storage protocol used
    pub protocol: StorageProtocol,
    /// Content identifier/address
    pub identifier: String,
    /// Content size in bytes
    pub size: u64,
    /// Content checksum
    pub checksum: [u8; 32],
}

/// Supported storage protocols
#[derive(Debug, Clone, BorshSerialize, BorshDeserialize, Serialize, Deserialize)]
pub enum StorageProtocol {
    IPFS,
    Arweave,
    FileCoin,
    Custom(String),
}

/// Model validation status
#[derive(Debug, Clone, BorshSerialize, BorshDeserialize, Serialize, Deserialize)]
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

/// Validation result data
#[derive(Debug, Clone, BorshSerialize, BorshDeserialize, Serialize, Deserialize)]
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

/// Detailed validation metrics
#[derive(Debug, Clone, BorshSerialize, BorshDeserialize, Serialize, Deserialize)]
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

/// Performance-related metrics
#[derive(Debug, Clone, BorshSerialize, BorshDeserialize, Serialize, Deserialize)]
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

/// Safety evaluation metrics
#[derive(Debug, Clone, BorshSerialize, BorshDeserialize, Serialize, Deserialize)]
pub struct SafetyMetrics {
    /// Content safety score (0-100)
    pub safety_score: u8,
    /// Identified safety concerns
    pub concerns: Vec<SafetyConcern>,
    /// Mitigation recommendations
    pub recommendations: Vec<String>,
}

/// Bias detection metrics
#[derive(Debug, Clone, BorshSerialize, BorshDeserialize, Serialize, Deserialize)]
pub struct BiasMetrics {
    /// Overall bias score (0-100, lower is better)
    pub bias_score: u8,
    /// Detected bias categories
    pub detected_biases: Vec<BiasCategory>,
    /// Bias assessment confidence
    pub confidence: f32,
}

/// Resource utilization metrics
#[derive(Debug, Clone, BorshSerialize, BorshDeserialize, Serialize, Deserialize)]
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

/// Types of safety concerns
#[derive(Debug, Clone, BorshSerialize, BorshDeserialize, Serialize, Deserialize)]
pub enum SafetyConcern {
    Harmful,
    Unethical,
    Biased,
    Privacy,
    Security,
    Custom(String),
}

/// Categories of bias
#[derive(Debug, Clone, BorshSerialize, BorshDeserialize, Serialize, Deserialize)]
pub enum BiasCategory {
    Gender,
    Racial,
    Age,
    Cultural,
    Socioeconomic,
    Geographic,
    Custom(String),
}

/// Consensus information
#[derive(Debug, Clone, BorshSerialize, BorshDeserialize, Serialize, Deserialize)]
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

/// Validator signature information
#[derive(Debug, Clone, BorshSerialize, BorshDeserialize, Serialize, Deserialize)]
pub struct ValidatorSignature {
    /// Validator's public key
    pub validator: Pubkey,
    /// Signature of the validation result
    pub signature: [u8; 64],
    /// Timestamp of signature
    pub timestamp: UnixTimestamp,
}

/// Access control settings
#[derive(Debug, Clone, BorshSerialize, BorshDeserialize, Serialize, Deserialize)]
pub struct AccessControl {
    /// Whether results are public
    pub is_public: bool,
    /// Allowed viewers (empty if public)
    pub allowed_viewers: Vec<Pubkey>,
    /// Access expiration timestamp
    pub expires_at: Option<UnixTimestamp>,
}

impl Default for AccessControl {
    fn default() -> Self {
        Self {
            is_public: false,
            allowed_viewers: Vec::new(),
            expires_at: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use solana_program::clock::UnixTimestamp;

    #[test]
    fn test_model_submission_serialization() {
        let submission = ModelSubmission {
            owner: Pubkey::new_unique(),
            model_type: ModelType::LLM {
                architecture: "transformer".to_string(),
                parameter_count: 1_000_000,
                context_window: 2048,
            },
            metadata: "test".to_string(),
            status: ValidationStatus::Pending,
            created_at: 0,
            updated_at: 0,
            model_hash: [0; 32],
            storage_info: StorageInfo {
                protocol: StorageProtocol::IPFS,
                identifier: "test".to_string(),
                size: 1000,
                checksum: [0; 32],
            },
            validation_round: 0,
            metrics: None,
            access_control: AccessControl::default(),
        };

        let serialized = submission.try_to_vec().unwrap();
        let deserialized = ModelSubmission::try_from_slice(&serialized).unwrap();
        assert_eq!(submission.metadata, deserialized.metadata);
    }

    #[test]
    fn test_validation_metrics_creation() {
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
}
