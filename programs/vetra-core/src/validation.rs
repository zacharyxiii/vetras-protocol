use crate::{
    errors::{VetrasError, VetrasResult},
    state::{
        BiasCategory, BiasMetrics, ModelSubmission, ModelType, PerformanceMetrics,
        ResourceMetrics, SafetyConcern, SafetyMetrics, ValidationMetrics, ValidationResult,
        ValidationStatus, ConsensusInfo, ValidatorSignature
    },
};
use solana_program::{
    account_info::AccountInfo,
    clock::Clock,
    entrypoint::ProgramResult,
    msg,
    program_error::ProgramError,
    pubkey::Pubkey,
    sysvar::Sysvar,
};
use std::collections::HashMap;

/// Minimum required validation score
pub const MIN_VALIDATION_SCORE: u8 = 70;
/// Maximum validation attempts per model
pub const MAX_VALIDATION_ATTEMPTS: u8 = 3;
/// Consensus threshold percentage
pub const CONSENSUS_THRESHOLD: f32 = 0.75;
/// Maximum time window for validation (seconds)
pub const VALIDATION_WINDOW: i64 = 3600;

pub struct ValidationEngine<'a, 'b> {
    program_id: &'a Pubkey,
    submission: &'b mut ModelSubmission,
    validators: Vec<&'a AccountInfo<'b>>,
    clock: &'a Clock,
}

impl<'a, 'b> ValidationEngine<'a, 'b> {
    pub fn new(
        program_id: &'a Pubkey,
        submission: &'b mut ModelSubmission,
        validators: Vec<&'a AccountInfo<'b>>,
        clock: &'a Clock,
    ) -> Self {
        Self {
            program_id,
            submission,
            validators,
            clock,
        }
    }

    pub fn start_validation(&mut self) -> ProgramResult {
        if !self.is_valid_validator_set() {
            return Err(VetrasError::InsufficientValidators.into());
        }

        let started_at = self.clock.unix_timestamp;
        self.submission.update_status(ValidationStatus::InProgress {
            started_at,
            validator_count: self.validators.len() as u32,
        })?;

        msg!("Started validation for model: {:?}", self.submission.model_hash);
        Ok(())
    }

    pub fn submit_validation_result(
        &mut self,
        validator: &Pubkey,
        metrics: ValidationMetrics,
        signature: [u8; 64],
    ) -> VetrasResult<()> {
        // Verify validator is authorized
        if !self.is_authorized_validator(validator) {
            return Err(VetrasError::UnauthorizedValidator);
        }

        // Check validation is in progress
        match self.submission.status {
            ValidationStatus::InProgress { .. } => {}
            _ => return Err(VetrasError::InvalidStateTransition),
        }

        // Verify validation window
        if self.is_validation_timeout() {
            return Err(VetrasError::ValidationTimeout);
        }

        // Process validator's result
        self.process_validator_result(validator, metrics, signature)?;

        // Check if consensus is reached
        if self.check_consensus()? {
            self.finalize_validation()?;
        }

        Ok(())
    }

    fn process_validator_result(
        &mut self,
        validator: &Pubkey,
        metrics: ValidationMetrics,
        signature: [u8; 64],
    ) -> VetrasResult<()> {
        // Validate metrics
        self.validate_metrics(&metrics)?;

        // Create validator signature
        let validator_signature = ValidatorSignature {
            validator: *validator,
            signature,
            timestamp: self.clock.unix_timestamp,
        };

        // Store result
        if let ValidationStatus::InProgress { ref mut validator_count, .. } = self.submission.status {
            *validator_count += 1;
        }

        // Update submission metrics
        self.submission.metrics = Some(metrics);

        Ok(())
    }

    fn validate_metrics(&self, metrics: &ValidationMetrics) -> VetrasResult<()> {
        // Validate performance metrics
        if metrics.performance.error_rate > 1.0 
            || metrics.performance.latency == 0 
            || metrics.performance.throughput == 0 {
            return Err(VetrasError::InvalidValidationResult);
        }

        // Validate safety metrics
        if metrics.safety.safety_score > 100 {
            return Err(VetrasError::InvalidValidationResult);
        }

        // Validate resource metrics
        if metrics.resources.cpu_utilization > 100.0 
            || metrics.resources.gpu_utilization.map_or(false, |v| v > 100.0) {
            return Err(VetrasError::InvalidValidationResult);
        }

        Ok(())
    }

    pub fn validate_llm(&self, metrics: &mut ValidationMetrics) -> VetrasResult<()> {
        if let ModelType::LLM { parameter_count, context_window, .. } = &self.submission.model_type {
            // Adjust performance expectations based on model size
            let expected_latency = (*parameter_count / 1_000_000) * 10; // 10ms per million parameters
            if metrics.performance.latency as u64 > expected_latency * 2 {
                return Err(VetrasError::SubparLLMPerformance);
            }

            // Check context window utilization
            let memory_per_token = 64; // bytes
            let expected_memory = (*context_window as u64 * memory_per_token * 2) / (1024 * 1024); // MB
            if metrics.resources.memory_usage < expected_memory {
                return Err(VetrasError::MetricsComputationFailed);
            }

            // Update LLM-specific metrics
            metrics.performance.custom.insert(
                "tokens_per_second".to_string(),
                (*context_window as f64 / metrics.performance.latency as f64) * 1000.0,
            );
        }

        Ok(())
    }

    fn check_consensus(&self) -> VetrasResult<bool> {
        if let ValidationStatus::InProgress { validator_count, .. } = self.submission.status {
            let threshold = (validator_count as f32 * CONSENSUS_THRESHOLD).ceil() as u32;
            Ok(validator_count >= threshold)
        } else {
            Err(VetrasError::InvalidStateTransition)
        }
    }

    fn finalize_validation(&mut self) -> VetrasResult<()> {
        let metrics = self.submission.metrics.clone()
            .ok_or(VetrasError::MetricsComputationFailed)?;

        let consensus = ConsensusInfo {
            validator_count: self.validators.len() as u32,
            threshold_achieved: true,
            round: self.submission.validation_round,
            timestamp: self.clock.unix_timestamp,
        };

        let result = ValidationResult {
            score: self.compute_final_score(&metrics),
            metrics,
            consensus,
            signatures: Vec::new(), // To be filled with validator signatures
        };

        self.submission.update_status(ValidationStatus::Completed {
            completed_at: self.clock.unix_timestamp,
            result,
        })?;

        msg!("Validation completed successfully");
        Ok(())
    }

    fn compute_final_score(&self, metrics: &ValidationMetrics) -> u8 {
        let performance_score = self.compute_performance_score(&metrics.performance);
        let safety_score = metrics.safety.safety_score;
        let bias_score = 100 - metrics.bias.bias_score; // Inverse bias score (lower is better)

        // Weighted average of scores
        let final_score = (
            performance_score as f32 * 0.4 + // 40% weight to performance
            safety_score as f32 * 0.4 + // 40% weight to safety
            bias_score as f32 * 0.2 // 20% weight to bias
        ) as u8;

        final_score.min(100)
    }

    fn compute_performance_score(&self, metrics: &PerformanceMetrics) -> u8 {
        let latency_score = (1.0 - (metrics.latency as f32 / 1000.0).min(1.0)) * 100.0;
        let throughput_score = (metrics.throughput as f32 / 100.0).min(1.0) * 100.0;
        let error_score = (1.0 - metrics.error_rate) * 100.0;

        ((latency_score + throughput_score + error_score) / 3.0) as u8
    }

    fn is_valid_validator_set(&self) -> bool {
        self.validators.len() >= 3 && // Minimum 3 validators
        self.validators.iter().all(|v| self.is_authorized_validator(v.key))
    }

    fn is_authorized_validator(&self, validator: &Pubkey) -> bool {
        // In a real implementation, this would check validator stakes and reputation
        // For now, we just check if they're in our validator set
        self.validators.iter().any(|v| v.key == validator)
    }

    fn is_validation_timeout(&self) -> bool {
        if let ValidationStatus::InProgress { started_at, .. } = self.submission.status {
            self.clock.unix_timestamp - started_at > VALIDATION_WINDOW
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use solana_program::clock::Clock;

    fn create_test_submission() -> ModelSubmission {
        ModelSubmission {
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
            storage_info: crate::state::StorageInfo {
                protocol: crate::state::StorageProtocol::IPFS,
                identifier: "test".to_string(),
                size: 1000,
                checksum: [0; 32],
            },
            validation_round: 0,
            metrics: None,
            access_control: crate::state::AccessControl {
                is_public: false,
                allowed_viewers: vec![],
                expires_at: None,
            },
        }
    }

    fn create_test_metrics() -> ValidationMetrics {
        ValidationMetrics {
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
        }
    }

    #[test]
    fn test_validation_engine_creation() {
        let program_id = Pubkey::new_unique();
        let mut submission = create_test_submission();
        let validators = vec![];
        let clock = Clock::default();

        let engine = ValidationEngine::new(&program_id, &mut submission, validators, &clock);
        assert_eq!(engine.program_id, &program_id);
    }

    #[test]
    fn test_compute_final_score() {
        let program_id = Pubkey::new_unique();
        let mut submission = create_test_submission();
        let validators = vec![];
        let clock = Clock::default();
        let engine = ValidationEngine::new(&program_id, &mut submission, validators, &clock);

        let metrics = create_test_metrics();
        let score = engine.compute_final_score(&metrics);
        assert!(score > 0 && score <= 100);
    }

    #[test]
    fn test_validation_timeout() {
        let program_id = Pubkey::new_unique();
        let mut submission = create_test_submission();
        submission.status = ValidationStatus::InProgress {
            started_at: 0,
            validator_count: 3,
        };
        let validators = vec![];
        let mut clock = Clock::default();
        clock.unix_timestamp = VALIDATION_WINDOW + 1;

        let engine = ValidationEngine::new(&program_id, &mut submission, validators, &clock);
        assert!(engine.is_validation_timeout());
    }

    #[test]
    fn test_metrics_validation() {
        let program_id = Pubkey::new_unique();
        let mut submission = create_test_submission();
        let validators = vec![];
        let clock = Clock::default();
        let engine = ValidationEngine::new(&program_id, &mut submission, validators, &clock);

        let metrics = create_test_metrics();
        assert!(engine.validate_metrics(&metrics).is_ok());

        // Test invalid metrics
        let mut invalid_metrics = metrics;
        invalid_metrics.performance.error_rate = 2.0;
        assert!(engine.validate_metrics(&invalid_metrics).is_err());
    }

    #[test]
    fn test_llm_validation() {
        let program_id = Pubkey::new_unique();
        let mut submission = create_test_submission();
        let validators = vec![];
        let clock = Clock::default();
        let engine = ValidationEngine::new(&program_id, &mut submission, validators, &clock);

        let mut metrics = create_test_metrics();
        assert!(engine.validate_llm(&mut metrics).is_ok());
    }
}
