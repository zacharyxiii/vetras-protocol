use borsh::{BorshDeserialize, BorshSerialize};
use solana_program::{
    account_info::{next_account_info, AccountInfo},
    entrypoint::ProgramResult,
    msg,
    program_error::ProgramError,
    pubkey::Pubkey,
    system_instruction,
    program::{invoke, invoke_signed},
    sysvar::{clock::Clock, rent::Rent, Sysvar},
};

use crate::{
    errors::{VetrasError, VetrasResult},
    state::{
        ModelSubmission, ModelType, StorageInfo, ValidationMetrics,
        ValidationStatus, AccessControl,
    },
    validation::ValidationEngine,
};

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone)]
pub enum VetrasInstruction {
    /// Initialize a new model submission
    /// 
    /// Accounts expected:
    /// 0. `[signer]` Submitter account
    /// 1. `[writable]` Model submission account (PDA)
    /// 2. `[]` System program
    /// 3. `[]` Rent sysvar
    InitModelSubmission {
        model_type: ModelType,
        metadata: String,
        storage_info: StorageInfo,
        access_control: AccessControl,
    },

    /// Start validation process for a model
    /// 
    /// Accounts expected:
    /// 0. `[signer]` Validator account initiating the process
    /// 1. `[writable]` Model submission account
    /// 2. `[]` Clock sysvar
    /// 3..N. `[]` Validator accounts
    StartValidation {},

    /// Submit validation results
    /// 
    /// Accounts expected:
    /// 0. `[signer]` Validator account
    /// 1. `[writable]` Model submission account
    /// 2. `[]` Clock sysvar
    SubmitValidation {
        metrics: ValidationMetrics,
        signature: [u8; 64],
    },

    /// Update model access control
    /// 
    /// Accounts expected:
    /// 0. `[signer]` Model owner
    /// 1. `[writable]` Model submission account
    UpdateAccessControl {
        new_access_control: AccessControl,
    },

    /// Cancel validation process
    /// 
    /// Accounts expected:
    /// 0. `[signer]` Model owner or validator
    /// 1. `[writable]` Model submission account
    /// 2. `[]` Clock sysvar
    CancelValidation {},
}

pub struct Processor {}

impl Processor {
    pub fn process(
        program_id: &Pubkey,
        accounts: &[AccountInfo],
        instruction_data: &[u8],
    ) -> ProgramResult {
        let instruction = sInstruction::try_from_slice(instruction_data)
            .map_err(|_| ProgramError::InvalidInstructionData)?;

        match instruction {
            VetrasInstruction::InitModelSubmission {
                model_type,
                metadata,
                storage_info,
                access_control,
            } => {
                msg!("Instruction: InitModelSubmission");
                Self::process_init_model_submission(
                    program_id,
                    accounts,
                    model_type,
                    metadata,
                    storage_info,
                    access_control,
                )
            }
            VetrasInstruction::StartValidation {} => {
                msg!("Instruction: StartValidation");
                Self::process_start_validation(program_id, accounts)
            }
            VetrasInstruction::SubmitValidation { metrics, signature } => {
                msg!("Instruction: SubmitValidation");
                Self::process_submit_validation(program_id, accounts, metrics, signature)
            }
            VetrasInstruction::UpdateAccessControl { new_access_control } => {
                msg!("Instruction: UpdateAccessControl");
                Self::process_update_access_control(program_id, accounts, new_access_control)
            }
            VetrasInstruction::CancelValidation {} => {
                msg!("Instruction: CancelValidation");
                Self::process_cancel_validation(program_id, accounts)
            }
        }
    }

    fn process_init_model_submission(
        program_id: &Pubkey,
        accounts: &[AccountInfo],
        model_type: ModelType,
        metadata: String,
        storage_info: StorageInfo,
        access_control: AccessControl,
    ) -> ProgramResult {
        let account_info_iter = &mut accounts.iter();
        let submitter_info = next_account_info(account_info_iter)?;
        let submission_account_info = next_account_info(account_info_iter)?;
        let system_program_info = next_account_info(account_info_iter)?;
        let rent_info = next_account_info(account_info_iter)?;

        if !submitter_info.is_signer {
            return Err(ProgramError::MissingRequiredSignature);
        }

        // Verify account address derivation
        let (expected_address, bump_seed) = Self::find_submission_address(
            submitter_info.key,
            &storage_info.identifier,
            program_id,
        );
        if expected_address != *submission_account_info.key {
            return Err(ProgramError::InvalidSeeds);
        }

        // Calculate required space and rent
        let rent = Rent::from_account_info(rent_info)?;
        let model_submission = ModelSubmission::new(
            *submitter_info.key,
            model_type,
            metadata,
            storage_info.checksum,
            storage_info,
            access_control,
        )?;
        
        let space = model_submission.try_to_vec()?.len();
        let rent_lamports = rent.minimum_balance(space);

        // Create submission account
        let create_account_ix = system_instruction::create_account(
            submitter_info.key,
            submission_account_info.key,
            rent_lamports,
            space as u64,
            program_id,
        );

        invoke_signed(
            &create_account_ix,
            &[
                submitter_info.clone(),
                submission_account_info.clone(),
                system_program_info.clone(),
            ],
            &[&[
                submitter_info.key.as_ref(),
                storage_info.identifier.as_bytes(),
                &[bump_seed],
            ]],
        )?;

        // Initialize submission data
        model_submission.serialize(&mut *submission_account_info.try_borrow_mut_data()?)?;

        msg!("Model submission initialized successfully");
        Ok(())
    }

    fn process_start_validation(
        program_id: &Pubkey,
        accounts: &[AccountInfo],
    ) -> ProgramResult {
        let account_info_iter = &mut accounts.iter();
        let validator_info = next_account_info(account_info_iter)?;
        let submission_account_info = next_account_info(account_info_iter)?;
        let clock_info = next_account_info(account_info_iter)?;

        if !validator_info.is_signer {
            return Err(ProgramError::MissingRequiredSignature);
        }

        // Collect validator accounts
        let mut validators = Vec::new();
        for validator in account_info_iter {
            validators.push(validator);
        }

        // Deserialize submission account
        let mut submission = ModelSubmission::try_from_slice(
            &submission_account_info.try_borrow_data()?
        )?;

        // Initialize validation engine
        let clock = Clock::from_account_info(clock_info)?;
        let mut validation_engine = ValidationEngine::new(
            program_id,
            &mut submission,
            validators,
            &clock,
        );

        // Start validation process
        validation_engine.start_validation()?;

        // Save updated submission state
        submission.serialize(&mut *submission_account_info.try_borrow_mut_data()?)?;

        msg!("Validation process started successfully");
        Ok(())
    }

    fn process_submit_validation(
        program_id: &Pubkey,
        accounts: &[AccountInfo],
        metrics: ValidationMetrics,
        signature: [u8; 64],
    ) -> ProgramResult {
        let account_info_iter = &mut accounts.iter();
        let validator_info = next_account_info(account_info_iter)?;
        let submission_account_info = next_account_info(account_info_iter)?;
        let clock_info = next_account_info(account_info_iter)?;

        if !validator_info.is_signer {
            return Err(ProgramError::MissingRequiredSignature);
        }

        // Deserialize submission
        let mut submission = ModelSubmission::try_from_slice(
            &submission_account_info.try_borrow_data()?
        )?;

        // Initialize validation engine
        let clock = Clock::from_account_info(clock_info)?;
        let mut validation_engine = ValidationEngine::new(
            program_id,
            &mut submission,
            vec![],  // Not needed for result submission
            &clock,
        );

        // Submit validation result
        validation_engine.submit_validation_result(
            validator_info.key,
            metrics,
            signature,
        )?;

        // Save updated submission state
        submission.serialize(&mut *submission_account_info.try_borrow_mut_data()?)?;

        msg!("Validation result submitted successfully");
        Ok(())
    }

    fn process_update_access_control(
        program_id: &Pubkey,
        accounts: &[AccountInfo],
        new_access_control: AccessControl,
    ) -> ProgramResult {
        let account_info_iter = &mut accounts.iter();
        let owner_info = next_account_info(account_info_iter)?;
        let submission_account_info = next_account_info(account_info_iter)?;

        if !owner_info.is_signer {
            return Err(ProgramError::MissingRequiredSignature);
        }

        // Deserialize submission
        let mut submission = ModelSubmission::try_from_slice(
            &submission_account_info.try_borrow_data()?
        )?;

        // Verify owner
        if submission.owner != *owner_info.key {
            return Err(VetrasError::UnauthorizedAccess.into());
        }

        // Update access control
        submission.access_control = new_access_control;
        submission.update_status(submission.status)?;

        // Save updated submission state
        submission.serialize(&mut *submission_account_info.try_borrow_mut_data()?)?;

        msg!("Access control updated successfully");
        Ok(())
    }

    fn process_cancel_validation(
        program_id: &Pubkey,
        accounts: &[AccountInfo],
    ) -> ProgramResult {
        let account_info_iter = &mut accounts.iter();
        let authority_info = next_account_info(account_info_iter)?;
        let submission_account_info = next_account_info(account_info_iter)?;
        let clock_info = next_account_info(account_info_iter)?;

        if !authority_info.is_signer {
            return Err(ProgramError::MissingRequiredSignature);
        }

        // Deserialize submission
        let mut submission = ModelSubmission::try_from_slice(
            &submission_account_info.try_borrow_data()?
        )?;

        // Verify authority (owner or active validator)
        if submission.owner != *authority_info.key {
            // Check if signer is an active validator
            if let ValidationStatus::InProgress { .. } = submission.status {
                // Additional validator verification logic would go here
            } else {
                return Err(VetrasError::UnauthorizedAccess.into());
            }
        }

        // Update status to failed
        let clock = Clock::from_account_info(clock_info)?;
        submission.update_status(ValidationStatus::Failed {
            error_code: 0,
            error_message: "Validation cancelled by authority".to_string(),
            failed_at: clock.unix_timestamp,
        })?;

        // Save updated submission state
        submission.serialize(&mut *submission_account_info.try_borrow_mut_data()?)?;

        msg!("Validation cancelled successfully");
        Ok(())
    }

    fn find_submission_address(
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use solana_program::clock::Epoch;
    use solana_program::system_program;

    // Helper function to create test accounts
    fn create_test_account_info(
        key: Pubkey,
        is_signer: bool,
        lamports: u64,
        data: Vec<u8>,
        owner: Pubkey,
    ) -> AccountInfo {
        AccountInfo::new(
            &key,
            is_signer,
            false,
            &mut lamports,
            &mut data,
            &owner,
            false,
            Epoch::default(),
        )
    }

    #[test]
    fn test_init_model_submission() {
        let program_id = Pubkey::new_unique();
        let submitter = Pubkey::new_unique();
        let submission_key = Pubkey::new_unique();

        // Create test accounts
        let submitter_info = create_test_account_info(
            submitter,
            true,
            1000000000,
            vec![0; 0],
            system_program::id(),
        );
        let submission_info = create_test_account_info(
            submission_key,
            false,
            0,
            vec![0; 1000],
            program_id,
        );
        let system_program_info = create_test_account_info(
            system_program::id(),
            false,
            0,
            vec![],
            system_program::id(),
        );
        let rent_info = create_test_account_info(
            solana_program::sysvar::rent::id(),
            false,
            0,
            vec![],
            solana_program::sysvar::id(),
        );

        let accounts = vec![
            submitter_info,
            submission_info,
            system_program_info,
            rent_info,
        ];

        let model_type = ModelType::LLM {
            architecture: "transformer".to_string(),
            parameter_count: 1_000_000,
            context_window: 2048,
        };
        let storage_info = StorageInfo {
            protocol: crate::state::StorageProtocol::IPFS,
            identifier: "test".to_string(),
            size: 1000,
            checksum: [0; 32],
        };
        let access_control = AccessControl {
            is_public: false,
            allowed_viewers: vec![],
            expires_at: None,
        };

        // Create test instruction
        let instruction_data = VetrasInstruction::InitModelSubmission {
            model_type,
            metadata: "test metadata".to_string(),
            storage_info,
            access_control: access_control.clone(),
        };

        // Test with invalid signer
        let mut bad_accounts = accounts.clone();
        bad_accounts[0] = create_test_account_info(
            submitter,
            false, // not a signer
            1000000000,
            vec![0; 0],
            system_program::id(),
        );

        let result = Processor::process_init_model_submission(
            &program_id,
            &bad_accounts,
            model_type.clone(),
            "test metadata".to_string(),
            storage_info.clone(),
            access_control.clone(),
        );

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), ProgramError::MissingRequiredSignature);
    }

    #[test]
    fn test_start_validation() {
        let program_id = Pubkey::new_unique();
        let validator = Pubkey::new_unique();
        let submission_key = Pubkey::new_unique();

        // Create initial submission state
        let mut submission = ModelSubmission::new(
            Pubkey::new_unique(),
            ModelType::LLM {
                architecture: "transformer".to_string(),
                parameter_count: 1_000_000,
                context_window: 2048,
            },
            "test metadata".to_string(),
            [0; 32],
            StorageInfo {
                protocol: crate::state::StorageProtocol::IPFS,
                identifier: "test_id".to_string(),
                size: 1000,
                checksum: [0; 32],
            },
            AccessControl {
                is_public: false,
                allowed_viewers: vec![],
                expires_at: None,
            },
        ).unwrap();

        let mut submission_data = submission.try_to_vec().unwrap();

        let accounts = vec![
            // Validator
            create_test_account_info(
                validator,
                true,
                1000000000,
                vec![],
                system_program::id(),
            ),
            // Submission account
            create_test_account_info(
                submission_key,
                false,
                1000000000,
                submission_data.clone(),
                program_id,
            ),
            // Clock sysvar
            create_test_account_info(
                solana_program::sysvar::clock::id(),
                false,
                0,
                vec![],
                solana_program::sysvar::id(),
            ),
            // Additional validator accounts
            create_test_account_info(
                Pubkey::new_unique(),
                false,
                1000000000,
                vec![],
                program_id,
            ),
            create_test_account_info(
                Pubkey::new_unique(),
                false,
                1000000000,
                vec![],
                program_id,
            ),
        ];

        // Test valid start validation
        let result = Processor::process_start_validation(
            &program_id,
            &accounts,
        );
        assert!(result.is_ok());

        // Test with invalid signer
        let mut bad_accounts = accounts.clone();
        bad_accounts[0] = create_test_account_info(
            validator,
            false, // not a signer
            1000000000,
            vec![],
            system_program::id(),
        );

        let result = Processor::process_start_validation(
            &program_id,
            &bad_accounts,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_submit_validation() {
        let program_id = Pubkey::new_unique();
        let validator = Pubkey::new_unique();
        let submission_key = Pubkey::new_unique();

        // Create test validation metrics
        let metrics = ValidationMetrics {
            performance: crate::state::PerformanceMetrics {
                latency: 100,
                throughput: 1000,
                error_rate: 0.01,
                custom: std::collections::HashMap::new(),
            },
            safety: crate::state::SafetyMetrics {
                safety_score: 95,
                concerns: vec![],
                recommendations: vec![],
            },
            bias: crate::state::BiasMetrics {
                bias_score: 10,
                detected_biases: vec![],
                confidence: 0.95,
            },
            resources: crate::state::ResourceMetrics {
                memory_usage: 1024,
                cpu_utilization: 75.0,
                gpu_utilization: Some(80.0),
                bandwidth: 100.0,
            },
        };

        let signature = [0; 64];

        // Create initial submission state
        let mut submission = ModelSubmission::new(
            Pubkey::new_unique(),
            ModelType::LLM {
                architecture: "transformer".to_string(),
                parameter_count: 1_000_000,
                context_window: 2048,
            },
            "test metadata".to_string(),
            [0; 32],
            StorageInfo {
                protocol: crate::state::StorageProtocol::IPFS,
                identifier: "test_id".to_string(),
                size: 1000,
                checksum: [0; 32],
            },
            AccessControl {
                is_public: false,
                allowed_viewers: vec![],
                expires_at: None,
            },
        ).unwrap();

        // Set status to InProgress
        submission.update_status(ValidationStatus::InProgress {
            started_at: 0,
            validator_count: 0,
        }).unwrap();

        let mut submission_data = submission.try_to_vec().unwrap();

        let accounts = vec![
            create_test_account_info(
                validator,
                true,
                1000000000,
                vec![],
                system_program::id(),
            ),
            create_test_account_info(
                submission_key,
                false,
                1000000000,
                submission_data.clone(),
                program_id,
            ),
            create_test_account_info(
                solana_program::sysvar::clock::id(),
                false,
                0,
                vec![],
                solana_program::sysvar::id(),
            ),
        ];

        // Test valid validation submission
        let result = Processor::process_submit_validation(
            &program_id,
            &accounts,
            metrics.clone(),
            signature,
        );
        assert!(result.is_ok());

        // Test with invalid signer
        let mut bad_accounts = accounts.clone();
        bad_accounts[0] = create_test_account_info(
            validator,
            false, // not a signer
            1000000000,
            vec![],
            system_program::id(),
        );

        let result = Processor::process_submit_validation(
            &program_id,
            &bad_accounts,
            metrics,
            signature,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_update_access_control() {
        let program_id = Pubkey::new_unique();
        let owner = Pubkey::new_unique();
        let submission_key = Pubkey::new_unique();

        // Create initial submission
        let mut submission = ModelSubmission::new(
            owner,
            ModelType::LLM {
                architecture: "transformer".to_string(),
                parameter_count: 1_000_000,
                context_window: 2048,
            },
            "test metadata".to_string(),
            [0; 32],
            StorageInfo {
                protocol: crate::state::StorageProtocol::IPFS,
                identifier: "test_id".to_string(),
                size: 1000,
                checksum: [0; 32],
            },
            AccessControl {
                is_public: false,
                allowed_viewers: vec![],
                expires_at: None,
            },
        ).unwrap();

        let mut submission_data = submission.try_to_vec().unwrap();

        let accounts = vec![
            create_test_account_info(
                owner,
                true,
                1000000000,
                vec![],
                system_program::id(),
            ),
            create_test_account_info(
                submission_key,
                false,
                1000000000,
                submission_data.clone(),
                program_id,
            ),
        ];

        let new_access_control = AccessControl {
            is_public: true,
            allowed_viewers: vec![],
            expires_at: None,
        };

        // Test valid access control update
        let result = Processor::process_update_access_control(
            &program_id,
            &accounts,
            new_access_control.clone(),
        );
        assert!(result.is_ok());

        // Test with non-owner
        let mut bad_accounts = accounts.clone();
        bad_accounts[0] = create_test_account_info(
            Pubkey::new_unique(), // different pubkey
            true,
            1000000000,
            vec![],
            system_program::id(),
        );

        let result = Processor::process_update_access_control(
            &program_id,
            &bad_accounts,
            new_access_control,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_cancel_validation() {
        let program_id = Pubkey::new_unique();
        let owner = Pubkey::new_unique();
        let submission_key = Pubkey::new_unique();

        // Create initial submission state
        let mut submission = ModelSubmission::new(
            owner,
            ModelType::LLM {
                architecture: "transformer".to_string(),
                parameter_count: 1_000_000,
                context_window: 2048,
            },
            "test metadata".to_string(),
            [0; 32],
            StorageInfo {
                protocol: crate::state::StorageProtocol::IPFS,
                identifier: "test_id".to_string(),
                size: 1000,
                checksum: [0; 32],
            },
            AccessControl {
                is_public: false,
                allowed_viewers: vec![],
                expires_at: None,
            },
        ).unwrap();

        submission.update_status(ValidationStatus::InProgress {
            started_at: 0,
            validator_count: 1,
        }).unwrap();

        let mut submission_data = submission.try_to_vec().unwrap();

        let accounts = vec![
            create_test_account_info(
                owner,
                true,
                1000000000,
                vec![],
                system_program::id(),
            ),
            create_test_account_info(
                submission_key,
                false,
                1000000000,
                submission_data.clone(),
                program_id,
            ),
            create_test_account_info(
                solana_program::sysvar::clock::id(),
                false,
                0,
                vec![],
                solana_program::sysvar::id(),
            ),
        ];

        // Test valid cancellation
        let result = Processor::process_cancel_validation(
            &program_id,
            &accounts,
        );
        assert!(result.is_ok());

        // Test with non-owner and non-validator
        let mut bad_accounts = accounts.clone();
        bad_accounts[0] = create_test_account_info(
            Pubkey::new_unique(), // different pubkey
            true,
            1000000000,
            vec![],
            system_program::id(),
        );

        let result = Processor::process_cancel_validation(
            &program_id,
            &bad_accounts,
        );
        assert!(result.is_err());
    }
}