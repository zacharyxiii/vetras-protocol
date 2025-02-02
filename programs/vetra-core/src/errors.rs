use num_derive::FromPrimitive;
use solana_program::{
    decode_error::DecodeError,
    msg,
    program_error::ProgramError,
};
use thiserror::Error;

#[derive(Error, Debug, Copy, Clone, FromPrimitive, PartialEq)]
pub enum VetraError {
    // Model Submission Errors
    #[error("Model data exceeds maximum size limit of 10MB")]
    ModelTooLarge = 0,
    #[error("Invalid model format or corrupt data")]
    InvalidModelFormat = 1,
    #[error("Model metadata is missing required fields")]
    IncompleteMetadata = 2,
    #[error("Model type is not supported by current validators")]
    UnsupportedModelType = 3,
    #[error("Model hash verification failed")]
    HashMismatch = 4,

    // Validation Process Errors
    #[error("Validation session already in progress for this model")]
    ValidationInProgress = 10,
    #[error("Validation timeout - exceeded maximum time limit")]
    ValidationTimeout = 11,
    #[error("Insufficient validator participation")]
    InsufficientValidators = 12,
    #[error("Validation result format mismatch")]
    ResultFormatMismatch = 13,
    #[error("Invalid validation proof submitted")]
    InvalidValidationProof = 14,
    #[error("Validation metrics computation failed")]
    MetricsComputationFailed = 15,

    // Consensus Errors
    #[error("Failed to reach consensus threshold")]
    ConsensusFailure = 20,
    #[error("Invalid consensus signature")]
    InvalidConsensusSignature = 21,
    #[error("Consensus round mismatch")]
    ConsensusRoundMismatch = 22,
    #[error("Duplicate validation submission")]
    DuplicateSubmission = 23,

    // Access Control Errors
    #[error("Account not authorized to submit models")]
    UnauthorizedSubmission = 30,
    #[error("Account not authorized to perform validation")]
    UnauthorizedValidator = 31,
    #[error("Account not authorized to access results")]
    UnauthorizedAccess = 32,
    #[error("Account has exceeded submission rate limit")]
    RateLimitExceeded = 33,

    // LLM-Specific Validation Errors
    #[error("LLM response format validation failed")]
    InvalidLLMResponse = 40,
    #[error("LLM performance metrics below threshold")]
    SubparLLMPerformance = 41,
    #[error("LLM bias detection check failed")]
    BiasDetected = 42,
    #[error("LLM safety check failed")]
    SafetyCheckFailed = 43,
    #[error("LLM hallucination detection threshold exceeded")]
    ExcessiveHallucination = 44,

    // Storage Errors
    #[error("Failed to store validation result on-chain")]
    StorageError = 50,
    #[error("Failed to retrieve stored validation data")]
    RetrievalError = 51,
    #[error("Storage capacity exceeded for account")]
    StorageCapacityExceeded = 52,
    #[error("Invalid data compression format")]
    CompressionError = 53,

    // System State Errors
    #[error("Program upgrade required - version mismatch")]
    VersionMismatch = 60,
    #[error("System pause in effect")]
    SystemPaused = 61,
    #[error("Emergency shutdown active")]
    EmergencyShutdown = 62,

    // Account State Errors
    #[error("Invalid account state transition")]
    InvalidStateTransition = 70,
    #[error("Account data verification failed")]
    AccountVerificationFailed = 71,
    #[error("Required account not initialized")]
    UninitializedAccount = 72,
}

impl From<VetraError> for ProgramError {
    fn from(e: VetraError) -> Self {
        ProgramError::Custom(e as u32)
    }
}

impl<T> DecodeError<T> for VetraError {
    fn type_of() -> &'static str {
        "VetraError"
    }
}

pub type VetraResult<T> = Result<T, VetraError>;

pub trait ErrorHandler {
    fn log_and_map<T>(self, error_context: &str) -> VetraResult<T>
    where
        Self: Sized + std::fmt::Debug,
    {
        msg!("Error occurred in {}: {:?}", error_context, self);
        Err(match self {
            // Map specific error types to VetraError variants
            _ => VetraError::StorageError, // Default mapping for unknown errors
        })
    }
}

// Implement ErrorHandler for common error types
impl ErrorHandler for std::io::Error {}
impl ErrorHandler for borsh::maybestd::io::Error {}
impl ErrorHandler for solana_program::program_error::ProgramError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_conversion() {
        let error = VetraError::ModelTooLarge;
        let program_error: ProgramError = error.into();
        assert_eq!(program_error, ProgramError::Custom(0));
    }

    #[test]
    fn test_error_messages() {
        assert_eq!(
            VetraError::ModelTooLarge.to_string(),
            "Model data exceeds maximum size limit of 10MB"
        );
        assert_eq!(
            VetraError::ValidationInProgress.to_string(),
            "Validation session already in progress for this model"
        );
    }

    #[test]
    fn test_error_handling_trait() {
        let io_error = std::io::Error::new(std::io::ErrorKind::Other, "test error");
        let result: VetraResult<()> = io_error.log_and_map("IO Operation");
        assert!(matches!(result, Err(VetraError::StorageError)));
    }

    #[test]
    fn test_all_error_variants() {
        // Test that all error variants can be converted to ProgramError
        let errors = [
            VetraError::ModelTooLarge,
            VetraError::InvalidModelFormat,
            VetraError::ValidationInProgress,
            VetraError::ConsensusFailure,
            VetraError::UnauthorizedSubmission,
            VetraError::InvalidLLMResponse,
            VetraError::StorageError,
            VetraError::VersionMismatch,
            VetraError::InvalidStateTransition,
        ];

        for error in errors.iter() {
            let program_error: ProgramError = (*error).into();
            assert_eq!(program_error, ProgramError::Custom(*error as u32));
        }
    }
}
