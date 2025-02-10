use solana_program::{
    instruction::{AccountMeta, Instruction},
    program_error::ProgramError,
    pubkey::Pubkey,
    system_program,
    sysvar,
};
use solana_sdk::hash::hash;
use std::convert::TryInto;

use crate::types::{
    ModelType, StorageInfo, ValidationMetrics,
    AccessControl, ModelSubmission,
};

/// Module for creating program instructions
pub mod create_instruction {
    use super::*;

    /// Create instruction for initializing a model submission
    pub fn init_model_submission(
        program_id: &Pubkey,
        payer: &Pubkey,
        submission_address: &Pubkey,
        model_type: ModelType,
        metadata: String,
        storage_info: StorageInfo,
        access_control: AccessControl,
    ) -> Result<Instruction, ProgramError> {
        let accounts = vec![
            AccountMeta::new(*payer, true),
            AccountMeta::new(*submission_address, false),
            AccountMeta::new_readonly(system_program::id(), false),
            AccountMeta::new_readonly(sysvar::rent::id(), false),
        ];

        let data = VetrasInstruction::InitModelSubmission {
            model_type,
            metadata,
            storage_info,
            access_control,
        }
        .try_to_vec()?;

        Ok(Instruction {
            program_id: *program_id,
            accounts,
            data,
        })
    }

    /// Create instruction for starting validation process
    pub fn start_validation(
        program_id: &Pubkey,
        validator: &Pubkey,
        submission_address: &Pubkey,
        validators: &[Pubkey],
    ) -> Result<Instruction, ProgramError> {
        let mut accounts = vec![
            AccountMeta::new(*validator, true),
            AccountMeta::new(*submission_address, false),
            AccountMeta::new_readonly(sysvar::clock::id(), false),
        ];

        // Add validator accounts
        for validator_key in validators {
            accounts.push(AccountMeta::new_readonly(*validator_key, false));
        }

        let data = VetrasInstruction::StartValidation {}.try_to_vec()?;

        Ok(Instruction {
            program_id: *program_id,
            accounts,
            data,
        })
    }

    /// Create instruction for submitting validation results
    pub fn submit_validation(
        program_id: &Pubkey,
        validator: &Pubkey,
        submission_address: &Pubkey,
        metrics: ValidationMetrics,
        signature: [u8; 64],
    ) -> Result<Instruction, ProgramError> {
        let accounts = vec![
            AccountMeta::new(*validator, true),
            AccountMeta::new(*submission_address, false),
            AccountMeta::new_readonly(sysvar::clock::id(), false),
        ];

        let data = VetrasInstruction::SubmitValidation {
            metrics,
            signature,
        }
        .try_to_vec()?;

        Ok(Instruction {
            program_id: *program_id,
            accounts,
            data,
        })
    }

    /// Create instruction for updating access control
    pub fn update_access_control(
        program_id: &Pubkey,
        owner: &Pubkey,
        submission_address: &Pubkey,
        new_access_control: AccessControl,
    ) -> Result<Instruction, ProgramError> {
        let accounts = vec![
            AccountMeta::new(*owner, true),
            AccountMeta::new(*submission_address, false),
        ];

        let data = VetrasInstruction::UpdateAccessControl {
            new_access_control,
        }
        .try_to_vec()?;

        Ok(Instruction {
            program_id: *program_id,
            accounts,
            data,
        })
    }

    /// Create instruction for canceling validation
    pub fn cancel_validation(
        program_id: &Pubkey,
        authority: &Pubkey,
        submission_address: &Pubkey,
    ) -> Result<Instruction, ProgramError> {
        let accounts = vec![
            AccountMeta::new(*authority, true),
            AccountMeta::new(*submission_address, false),
            AccountMeta::new_readonly(sysvar::clock::id(), false),
        ];

        let data = VetrasInstruction::CancelValidation {}.try_to_vec()?;

        Ok(Instruction {
            program_id: *program_id,
            accounts,
            data,
        })
    }
}

/// Derive program address for model submission
pub fn derive_submission_address(
    owner: &Pubkey,
    identifier: &str,
    program_id: &Pubkey,
) -> (Pubkey, u8) {
    Pubkey::find_program_address(
        &[
            owner.as_ref(),
            identifier.as_bytes(),
        ],
        program_id,
    )
}

/// Hash model data for unique identification
pub fn hash_model_data(data: &[u8]) -> [u8; 32] {
    let hash_result = hash(data);
    hash_result.to_bytes()
}

/// Calculate required account space for model submission
pub fn get_required_space(model_type: &ModelType, metadata: &str) -> usize {
    let base_size = std::mem::size_of::<ModelSubmission>();
    let metadata_size = metadata.len();
    
    // Add space based on model type
    let model_type_size = match model_type {
        ModelType::LLM { .. } => 1024, // Additional space for LLM data
        ModelType::ComputerVision { .. } => 512,
        ModelType::TabularPredictor { .. } => 256,
        ModelType::Custom { .. } => 2048, // More space for custom specifications
    };

    base_size + metadata_size + model_type_size
}

/// Validate model metadata format
pub fn validate_metadata(metadata: &str) -> Result<(), ProgramError> {
    // Check size limits
    if metadata.len() > 1024 * 10 { // 10KB limit
        return Err(ProgramError::InvalidArgument);
    }

    // Attempt to parse as JSON
    if serde_json::from_str::<serde_json::Value>(metadata).is_err() {
        return Err(ProgramError::InvalidArgument);
    }

    Ok(())
}

/// Validate storage information
pub fn validate_storage_info(info: &StorageInfo) -> Result<(), ProgramError> {
    // Check size limits
    if info.size > 1024 * 1024 * 1024 { // 1GB limit
        return Err(ProgramError::InvalidArgument);
    }

    // Validate identifier format based on protocol
    match info.protocol {
        StorageProtocol::IPFS => {
            if !info.identifier.starts_with("Qm") {
                return Err(ProgramError::InvalidArgument);
            }
        }
        StorageProtocol::Arweave => {
            if info.identifier.len() != 43 {
                return Err(ProgramError::InvalidArgument);
            }
        }
        StorageProtocol::FileCoin => {
            // Add FileCoin-specific validation
        }
        StorageProtocol::Custom(_) => {
            // Custom protocol validation could be added here
        }
    }

    Ok(())
}

/// Format pubkey for display
pub fn format_pubkey(pubkey: &Pubkey) -> String {
    let s = pubkey.to_string();
    format!("{}...{}", &s[..4], &s[s.len()-4..])
}

#[cfg(test)]
mod tests {
    use super::*;
    use solana_program::pubkey::Pubkey;

    #[test]
    fn test_derive_submission_address() {
        let owner = Pubkey::new_unique();
        let program_id = Pubkey::new_unique();
        let identifier = "test_model";

        let (address, bump) = derive_submission_address(&owner, identifier, &program_id);
        assert!(bump < 256);
        assert_ne!(address, Pubkey::default());
    }

    #[test]
    fn test_hash_model_data() {
        let data = b"test model data";
        let hash = hash_model_data(data);
        assert_eq!(hash.len(), 32);
        assert_ne!(hash, [0; 32]);
    }

    #[test]
    fn test_get_required_space() {
        let model_type = ModelType::LLM {
            architecture: "transformer".to_string(),
            parameter_count: 1_000_000,
            context_window: 2048,
        };
        let metadata = "test metadata";

        let space = get_required_space(&model_type, metadata);
        assert!(space > 0);
    }

    #[test]
    fn test_validate_metadata() {
        // Valid JSON metadata
        let valid_metadata = r#"{"name": "test", "version": "1.0"}"#;
        assert!(validate_metadata(valid_metadata).is_ok());

        // Invalid JSON metadata
        let invalid_metadata = r#"{"name": "test", version: 1.0"#;
        assert!(validate_metadata(invalid_metadata).is_err());

        // Too large metadata
        let large_metadata = "x".repeat(1024 * 11); // 11KB
        assert!(validate_metadata(&large_metadata).is_err());
    }

    #[test]
    fn test_validate_storage_info() {
        let valid_info = StorageInfo {
            protocol: StorageProtocol::IPFS,
            identifier: "QmTest123".to_string(),
            size: 1000,
            checksum: [0; 32],
        };
        assert!(validate_storage_info(&valid_info).is_ok());

        let invalid_info = StorageInfo {
            protocol: StorageProtocol::IPFS,
            identifier: "invalid".to_string(),
            size: 1000,
            checksum: [0; 32],
        };
        assert!(validate_storage_info(&invalid_info).is_err());
    }

    #[test]
    fn test_format_pubkey() {
        let pubkey = Pubkey::new_unique();
        let formatted = format_pubkey(&pubkey);
        assert_eq!(formatted.len(), 9); // 4 + 3 + 4
        assert!(formatted.contains("..."));
    }
}
