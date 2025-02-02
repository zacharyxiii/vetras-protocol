use solana_program::{
    account_info::AccountInfo,
    entrypoint,
    entrypoint::ProgramResult,
    program_error::ProgramError,
    pubkey::Pubkey,
    msg,
};

// Declare modules
pub mod errors;
pub mod state;
pub mod processor;
pub mod validation;
pub mod utils;

// Re-export commonly used types
pub use crate::{
    errors::{VetraError, VetraResult},
    processor::Processor,
    state::{ModelSubmission, ModelType, ValidationStatus},
};

// Program version
pub const PROGRAM_VERSION: u8 = 1;

// Constants for program limits
pub const MAX_MODEL_SIZE: u64 = 10 * 1024 * 1024; // 10MB
pub const MAX_VALIDATORS: u32 = 100;
pub const MIN_VALIDATORS: u32 = 3;
pub const VALIDATION_TIMEOUT_SECONDS: i64 = 3600; // 1 hour

/// Program entrypoint
#[cfg(not(feature = "no-entrypoint"))]
entrypoint!(process_instruction);

/// The main program entry point
/// Receives the raw instruction data and passes it to the processor
pub fn process_instruction(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    instruction_data: &[u8],
) -> ProgramResult {
    msg!("VETRA: Processing instruction");
    
    // Log basic transaction information
    if cfg!(debug_assertions) {
        msg!("Program ID: {}", program_id);
        msg!("Number of accounts: {}", accounts.len());
        msg!("Instruction data length: {}", instruction_data.len());
    }

    // Version check (for future upgrades)
    if instruction_data.is_empty() {
        msg!("Error: Empty instruction data");
        return Err(ProgramError::InvalidInstructionData);
    }

    // Transaction time limit check
    if let Some(first_account) = accounts.first() {
        if let Ok(clock) = solana_program::clock::Clock::get() {
            let transaction_start = clock.unix_timestamp;
            // Ensure we have enough time to process the instruction
            if transaction_start % solana_program::clock::DEFAULT_TICKS_PER_SLOT as i64 
                > VALIDATION_TIMEOUT_SECONDS - 1 {
                msg!("Error: Transaction would exceed time limit");
                return Err(VetraError::ValidationTimeout.into());
            }
        }
    }

    // Route the instruction to the processor
    let result = Processor::process(program_id, accounts, instruction_data);

    // Log the result status
    match &result {
        Ok(_) => msg!("Instruction executed successfully"),
        Err(e) => msg!("Failed to execute instruction: {:?}", e),
    }

    result
}

/// Helper function to verify program ownership of an account
pub fn verify_program_account(account: &AccountInfo, program_id: &Pubkey) -> ProgramResult {
    if account.owner != program_id {
        msg!("Error: Account not owned by program");
        return Err(ProgramError::IncorrectProgramId);
    }
    Ok(())
}

/// Helper function to verify account is writable
pub fn verify_writable(account: &AccountInfo) -> ProgramResult {
    if !account.is_writable {
        msg!("Error: Account must be writable");
        return Err(ProgramError::InvalidAccountData);
    }
    Ok(())
}

/// Helper function to verify account is signer
pub fn verify_signer(account: &AccountInfo) -> ProgramResult {
    if !account.is_signer {
        msg!("Error: Account must be signer");
        return Err(ProgramError::MissingRequiredSignature);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use solana_program::clock::Epoch;

    // Helper function to create test AccountInfo
    fn create_test_account(
        key: &Pubkey,
        owner: &Pubkey,
        is_signer: bool,
        is_writable: bool,
    ) -> AccountInfo {
        AccountInfo::new(
            key,
            is_signer,
            is_writable,
            &mut 0,
            &mut [],
            owner,
            false,
            Epoch::default(),
        )
    }

    #[test]
    fn test_verify_program_account() {
        let program_id = Pubkey::new_unique();
        let account_key = Pubkey::new_unique();
        
        // Test valid program account
        let valid_account = create_test_account(&account_key, &program_id, false, false);
        assert!(verify_program_account(&valid_account, &program_id).is_ok());

        // Test invalid program account
        let wrong_owner = Pubkey::new_unique();
        let invalid_account = create_test_account(&account_key, &wrong_owner, false, false);
        assert!(verify_program_account(&invalid_account, &program_id).is_err());
    }

    #[test]
    fn test_verify_writable() {
        let key = Pubkey::new_unique();
        let owner = Pubkey::new_unique();

        // Test writable account
        let writable_account = create_test_account(&key, &owner, false, true);
        assert!(verify_writable(&writable_account).is_ok());

        // Test non-writable account
        let non_writable_account = create_test_account(&key, &owner, false, false);
        assert!(verify_writable(&non_writable_account).is_err());
    }

    #[test]
    fn test_verify_signer() {
        let key = Pubkey::new_unique();
        let owner = Pubkey::new_unique();

        // Test signer account
        let signer_account = create_test_account(&key, &owner, true, false);
        assert!(verify_signer(&signer_account).is_ok());

        // Test non-signer account
        let non_signer_account = create_test_account(&key, &owner, false, false);
        assert!(verify_signer(&non_signer_account).is_err());
    }

    #[test]
    fn test_empty_instruction() {
        let program_id = Pubkey::new_unique();
        let account = create_test_account(&Pubkey::new_unique(), &program_id, false, false);
        
        let result = process_instruction(
            &program_id,
            &[account],
            &[], // Empty instruction data
        );
        
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), ProgramError::InvalidInstructionData);
    }
}

// Additional utility module for common program functions
pub mod utils {
    use super::*;
    use solana_program::system_program;

    pub fn validate_system_program(program_id: &Pubkey) -> ProgramResult {
        if program_id != &system_program::ID {
            msg!("Error: Invalid system program ID");
            return Err(ProgramError::IncorrectProgramId);
        }
        Ok(())
    }

    pub fn validate_account_size(current_size: usize, required_size: usize) -> ProgramResult {
        if current_size < required_size {
            msg!("Error: Account size too small");
            return Err(ProgramError::AccountDataTooSmall);
        }
        Ok(())
    }

    pub fn get_required_account_space(model_type: &ModelType) -> usize {
        let base_size = std::mem::size_of::<ModelSubmission>();
        match model_type {
            ModelType::LLM { .. } => base_size + 1024, // Additional space for LLM metadata
            ModelType::ComputerVision { .. } => base_size + 512,
            _ => base_size + 256,
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_validate_system_program() {
            assert!(validate_system_program(&system_program::ID).is_ok());
            assert!(validate_system_program(&Pubkey::new_unique()).is_err());
        }

        #[test]
        fn test_validate_account_size() {
            assert!(validate_account_size(100, 50).is_ok());
            assert!(validate_account_size(50, 100).is_err());
        }

        #[test]
        fn test_get_required_account_space() {
            let llm_type = ModelType::LLM {
                architecture: "transformer".to_string(),
                parameter_count: 1_000_000,
                context_window: 2048,
            };
            let cv_type = ModelType::ComputerVision {
                architecture: "cnn".to_string(),
                input_resolution: (224, 224),
                model_family: "resnet".to_string(),
            };

            assert!(get_required_account_space(&llm_type) > get_required_account_space(&cv_type));
        }
    }
}