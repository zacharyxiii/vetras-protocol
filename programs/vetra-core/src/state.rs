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
/// Maximum model size (10MB)
pub const MAX_MODEL_SIZE: u64 = 10 * 1024 * 1024;
/// Maximum context window for LLMs
pub const MAX_CONTEXT_WINDOW: u32 = 32768;

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
    /// Model version identifier
    pub version: String,
    /// Validation metrics
    pub metrics: Option<ValidationMetrics>,
    /// Access control settings
    pub access_control: AccessControl,
    /// Model dependencies (if any)
    pub dependencies: Vec<ModelDependency>,
    /// Hardware requirements
    pub hardware_requirements: HardwareRequirements,
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
    /// Encryption details (if encrypted)
    pub encryption: Option<EncryptionInfo>,
    /// Storage redundancy factor
    pub redundancy: u8,
    /// Geographic distribution of storage
    pub geo_distribution: Vec<String>,
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct EncryptionInfo {
    /// Encryption algorithm used
    pub algorithm: String,
    /// Public key used for encryption
    pub public_key: [u8; 32],
    /// Additional encryption parameters
    pub parameters: HashMap<String, String>,
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct ModelDependency {
    /// Dependency model hash
    pub model_hash: [u8; 32],
    /// Required version
    pub version_requirement: String,
    /// Dependency type
    pub dependency_type: DependencyType,
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub enum DependencyType {
    Required,
    Optional,
    Enhancement,
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct HardwareRequirements {
    /// Minimum memory required (MB)
    pub min_memory: u64,
    /// Minimum CPU cores
    pub min_cpu_cores: u32,
    /// GPU requirements
    pub gpu_requirements: Option<GpuRequirements>,
    /// Minimum bandwidth (MB/s)
    pub min_bandwidth: f32,
    /// Storage requirements (MB)
    pub storage_requirements: u64,
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct GpuRequirements {
    /// Minimum VRAM (MB)
    pub min_vram: u64,
    /// Required GPU architecture
    pub architecture: String,
    /// Minimum compute capability
    pub compute_capability: f32,
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
        quantization: Option<QuantizationType>,
        training_dataset: String,
        license: String,
    },
    ComputerVision {
        architecture: String,
        input_resolution: (u32, u32),
        model_family: String,
        supported_formats: Vec<String>,
        pre_trained: bool,
    },
    TabularPredictor {
        framework: String,
        input_features: u32,
        model_type: String,
        target_variables: Vec<String>,
        feature_importance: Option<HashMap<String, f32>>,
    },
    MultiModal {
        architectures: Vec<String>,
        modalities: Vec<Modality>,
        integration_method: String,
    },
    Custom {
        category: String,
        specifications: String,
    },
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub enum QuantizationType {
    Int8,
    Int4,
    Mixed,
    Custom(String),
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub enum Modality {
    Text,
    Image,
    Audio,
    Video,
    Sensor,
    Custom(String),
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub enum ValidationStatus {
    Pending,
    InProgress {
        started_at: UnixTimestamp,
        validator_count: u32,
        current_phase: ValidationPhase,
    },
    Completed {
        completed_at: UnixTimestamp,
        result: ValidationResult,
    },
    Failed {
        error_code: u32,
        error_message: String,
        failed_at: UnixTimestamp,
        retries_remaining: u8,
    },
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub enum ValidationPhase {
    Initial,
    Performance,
    Safety,
    Bias,
    Consensus,
    Final,
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
    /// Validation duration
    pub duration: u64,
    /// Resource consumption
    pub resource_usage: ResourceMetrics,
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
    /// Model-specific metrics
    pub model_specific: HashMap<String, f64>,
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
    /// Performance stability score
    pub stability_score: f32,
    /// Cold start performance
    pub cold_start_latency: u32,
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct SafetyMetrics {
    /// Content safety score (0-100)
    pub safety_score: u8,
    /// Identified safety concerns
    pub concerns: Vec<SafetyConcern>,
    /// Mitigation recommendations
    pub recommendations: Vec<String>,
    /// Security vulnerabilities
    pub vulnerabilities: Vec<SecurityVulnerability>,
    /// Compliance status
    pub compliance: ComplianceStatus,
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct SecurityVulnerability {
    pub severity: VulnerabilitySeverity,
    pub description: String,
    pub mitigation: String,
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub enum VulnerabilitySeverity {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct ComplianceStatus {
    pub gdpr_compliant: bool,
    pub hipaa_compliant: bool,
    pub custom_compliance: HashMap<String, bool>,
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct BiasMetrics {
    /// Overall bias score (0-100, lower is better)
    pub bias_score: u8,
    /// Detected bias categories
    pub detected_biases: Vec<BiasCategory>,
    /// Bias assessment confidence
    pub confidence: f32,
    /// Demographic parity metrics
    pub demographic_parity: HashMap<String, f32>,
    /// Historical bias indicators
    pub historical_bias: Vec<HistoricalBias>,
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct HistoricalBias {
    pub category: String,
    pub time_period: String,
    pub bias_level: f32,
    pub description: String,
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
    /// Energy consumption (kWh)
    pub energy_consumption: f32,
    /// Cost metrics
    pub cost_metrics: CostMetrics,
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct CostMetrics {
    pub compute_cost: f64,
    pub storage_cost: f64,
    pub network_cost: f64,
    pub total_cost: f64,
    pub currency: String,
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub enum SafetyConcern {
    Harmful,
    Unethical,
    Biased,
    Privacy,
    Security,
    Environmental,
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
    Language,
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
    /// Validator weights
    pub validator_weights: HashMap<Pubkey, u64>,
    /// Dissenting opinions
    pub dissenting_opinions: Vec<DissentingOpinion>,
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct DissentingOpinion {
    pub validator: Pubkey,
    pub reason: String,
    pub alternative_score: u8,
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct ValidatorSignature {
    /// Validator's public key
    pub validator: Pubkey,
    /// Signature of the validation result
    pub signature: [u8; 64],
    /// Timestamp of signature
    pub timestamp: UnixTimestamp,
    /// Reputation score at time of validation
    pub reputation_score: u32,
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct AccessControl {
    /// Whether results are public
    pub is_public: bool,
    /// Allowed viewers (empty if public)
    pub allowed_viewers: Vec<Pubkey>,
    /// Access expiration timestamp
    pub expires_at: Option<UnixTimestamp>,
    /// Access level
    pub access_level: AccessLevel,
    /// Data usage policies
    pub usage_policies: Vec<UsagePolicy>,
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub enum AccessLevel {
    Public,
    Private,
    Restricted(Vec<Pubkey>),
    TimeLimited {
        duration: i64,
        allowed_users: Vec<Pubkey>,
    },
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub enum UsagePolicy {
    NoCommercial,
    AttributionRequired,
    ShareAlike,
    NonDerivative,
    CustomPolicy(String),
}

impl ModelSubmission {
    pub fn new(
        owner: Pubkey,
        model_type: ModelType,
        metadata: String,
        model_hash: [u8; 32],
        storage_info: StorageInfo,
        access_control: AccessControl,
        hardware_requirements: HardwareRequirements,
    ) -> Result<Self, ProgramError> {
        if metadata.len() > MAX_METADATA_LENGTH {
            return Err(ProgramError::InvalidArgument);
        }

        if storage_info.size > MAX_MODEL_SIZE {
            return Err(ProgramError::InvalidArgument);
        }

        // Validate model type specific requirements
        Self::validate_model_type_requirements(&model_type)?;

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
            version: "1.0.0".to_string(),
            metrics: None,
            access_control,
            dependencies: Vec::new(),
            hardware_requirements,
        })
    }

    fn validate_model_type_requirements(model_type: &ModelType) -> Result<(), ProgramError> {
        match model_type {
            ModelType::LLM { context_window, parameter_count, .. } => {
                if *context_window > MAX_CONTEXT_WINDOW {
                    return Err(ProgramError::InvalidArgument);
                }
                if *parameter_count == 0 {
                    return Err(ProgramError::InvalidArgument);
                }
            }
            ModelType::ComputerVision { input_resolution, .. } => {
                if input_resolution.0 == 0 || input_resolution.1 == 0 {
                    return Err(ProgramError::InvalidArgument);
                }
            }
            ModelType::TabularPredictor { input_features, .. } => {
                if *input_features == 0 {
                    return Err(ProgramError::InvalidArgument);
                }
            }
            ModelType::MultiModal { modalities, .. } => {
                if modalities.is_empty() {
                    return Err(ProgramError::InvalidArgument);
                }
            }
            ModelType::Custom { .. } => {}
        }
        Ok(())
    }

    pub fn update_status(&mut self, new_status: ValidationStatus) -> Result<(), ProgramError> {
        // Validate state transition
        match (&self.status, &new_status) {
            (ValidationStatus::Pending, ValidationStatus::InProgress { .. }) => {}
            (ValidationStatus::InProgress { .. }, ValidationStatus::Completed { .. }) => {}
            (ValidationStatus::InProgress { .. }, ValidationStatus::Failed { .. }) => {}
            _ => return Err(ProgramError::InvalidArgument),
        }

        self.status = new_status;
        self.updated_at = solana_program::clock::Clock::get()?.unix_timestamp;
        Ok(())
    }

    pub fn advance_validation_phase(&mut self) -> Result<(), ProgramError> {
        if let ValidationStatus::InProgress { ref mut current_phase, .. } = self.status {
            *current_phase = match current_phase {
                ValidationPhase::Initial => ValidationPhase::Performance,
                ValidationPhase::Performance => ValidationPhase::Safety,
                ValidationPhase::Safety => ValidationPhase::Bias,
                ValidationPhase::Bias => ValidationPhase::Consensus,
                ValidationPhase::Consensus => ValidationPhase::Final,
                ValidationPhase::Final => return Err(ProgramError::InvalidArgument),
            };
            self.updated_at = solana_program::clock::Clock::get()?.unix_timestamp;
            Ok(())
        } else {
            Err(ProgramError::InvalidArgument)
        }
    }

    pub fn update_metrics(&mut self, metrics: ValidationMetrics) -> Result<(), ProgramError> {
        self.validate_metrics(&metrics)?;
        self.metrics = Some(metrics);
        self.updated_at = solana_program::clock::Clock::get()?.unix_timestamp;
        Ok(())
    }

    fn validate_metrics(&self, metrics: &ValidationMetrics) -> Result<(), ProgramError> {
        // Validate performance metrics
        if metrics.performance.error_rate > 1.0 || 
           metrics.performance.latency == 0 || 
           metrics.performance.throughput == 0 {
            return Err(ProgramError::InvalidArgument);
        }

        // Validate safety metrics
        if metrics.safety.safety_score > 100 {
            return Err(ProgramError::InvalidArgument);
        }

        // Validate resource metrics
        if metrics.resources.cpu_utilization > 100.0 || 
           metrics.resources.gpu_utilization.map_or(false, |v| v > 100.0) {
            return Err(ProgramError::InvalidArgument);
        }

        Ok(())
    }

    pub fn add_dependency(&mut self, dependency: ModelDependency) -> Result<(), ProgramError> {
        if self.dependencies.len() >= 10 {  // Arbitrary limit
            return Err(ProgramError::InvalidArgument);
        }
        if self.dependencies.iter().any(|d| d.model_hash == dependency.model_hash) {
            return Err(ProgramError::InvalidArgument);
        }
        self.dependencies.push(dependency);
        self.updated_at = solana_program::clock::Clock::get()?.unix_timestamp;
        Ok(())
    }

    pub fn update_access_control(&mut self, new_access_control: AccessControl) -> Result<(), ProgramError> {
        self.access_control = new_access_control;
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
        match &self.access_control.access_level {
            AccessLevel::Public => true,
            AccessLevel::Private => &self.owner == viewer,
            AccessLevel::Restricted(allowed_users) => {
                &self.owner == viewer || allowed_users.contains(viewer)
            }
            AccessLevel::TimeLimited { allowed_users, duration } => {
                if let Ok(now) = solana_program::clock::Clock::get() {
                    if now.unix_timestamp - self.created_at > *duration {
                        return false;
                    }
                    &self.owner == viewer || allowed_users.contains(viewer)
                } else {
                    false
                }
            }
        }
    }

    pub fn verify_storage(&self) -> Result<bool, ProgramError> {
        // Verify storage size matches
        if self.storage_info.size > MAX_MODEL_SIZE {
            return Ok(false);
        }

        // Verify checksum if encryption is not used
        if self.storage_info.encryption.is_none() {
            if self.storage_info.checksum != self.model_hash {
                return Ok(false);
            }
        }

        // Additional storage protocol specific checks
        match self.storage_info.protocol {
            StorageProtocol::IPFS => {
                // Verify IPFS CID format
                if !self.storage_info.identifier.starts_with("Qm") {
                    return Ok(false);
                }
            }
            StorageProtocol::Arweave => {
                // Verify Arweave transaction ID format
                if self.storage_info.identifier.len() != 43 {
                    return Ok(false);
                }
            }
            StorageProtocol::FileCoin => {
                // Add FileCoin specific checks
            }
            StorageProtocol::Custom(_) => {
                // Custom protocol validation
            }
        }

        Ok(true)
    }

    pub fn estimate_cost(&self) -> Result<f64, ProgramError> {
        let base_cost = match &self.model_type {
            ModelType::LLM { parameter_count, .. } => {
                // Cost scales with model size
                (*parameter_count as f64 * 0.001) + 10.0
            }
            ModelType::ComputerVision { .. } => 5.0,
            ModelType::TabularPredictor { .. } => 3.0,
            ModelType::MultiModal { modalities, .. } => {
                // Cost scales with number of modalities
                modalities.len() as f64 * 2.0 + 5.0
            }
            ModelType::Custom { .. } => 5.0,
        };

        // Add storage costs
        let storage_cost = (self.storage_info.size as f64 * 0.001) * 
            match self.storage_info.redundancy {
                r if r > 1 => r as f64 * 0.8, // Discount for redundancy
                _ => 1.0,
            };

        Ok(base_cost + storage_cost)
    }
