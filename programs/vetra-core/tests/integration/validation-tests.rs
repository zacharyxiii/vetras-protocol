use solana_program::{
    instruction::AccountMeta,
    pubkey::Pubkey,
    system_instruction,
};
use solana_program_test::*;
use solana_sdk::{signature::Keypair, signer::Signer, transaction::Transaction};

use vetra_core::{
    errors::VetraError,
    processor::VetraInstruction,
    state::{
        ModelType, StorageProtocol, ValidationStatus,
        StorageInfo, AccessControl, ValidationMetrics,
    },
};

use super::setup::TestContext;

#[tokio::test]
async fn test_full_validation_lifecycle() {
    // Initialize test context
    let mut context = TestContext::new().await;
    
    // Create test accounts
    let owner = Keypair::new();
    let validator1 = Keypair::new();
    let validator2 = Keypair::new();
    let validator3 = Keypair::new();

    // Create model submission
    let model_type = ModelType::LLM {
        architecture: "transformer".to_string(),
        parameter_count: 1_000_000,
        context_window: 2048,
    };

    let submission_pubkey = context
        .create_test_model_submission(&owner, model_type)
        .await
        .expect("Failed to create model submission");

    // Start validation process
    let start_validation_ix = VetraInstruction::StartValidation {};
    let accounts = vec![
        AccountMeta::new(validator1.pubkey(), true),
        AccountMeta::new(submission_pubkey, false),
        AccountMeta::new_readonly(solana_program::sysvar::clock::id(), false),
        AccountMeta::new_readonly(validator1.pubkey(), false),
        AccountMeta::new_readonly(validator2.pubkey(), false),
        AccountMeta::new_readonly(validator3.pubkey(), false),
    ];

    let tx = Transaction::new_signed_with_payer(
        &[context.create_instruction(start_validation_ix, accounts)],
        Some(&context.payer.pubkey()),
        &[&context.payer, &validator1],
        context.recent_blockhash,
    );

    context.banks_client.process_transaction(tx).await.unwrap();

    // Submit validation results from multiple validators
    let metrics = context.create_test_validation_metrics().await;
    
    // Validator 1 submission
    context
        .submit_validation_result(&validator1, &submission_pubkey, metrics.clone())
        .await
        .unwrap();

    // Validator 2 submission
    context
        .submit_validation_result(&validator2, &submission_pubkey, metrics.clone())
        .await
        .unwrap();

    // Validator 3 submission
    context
        .submit_validation_result(&validator3, &submission_pubkey, metrics)
        .await
        .unwrap();

    // Verify final state
    let account = context.get_account(&submission_pubkey).await.unwrap();
    let submission = vetra_core::state::ModelSubmission::unpack(&account.data).unwrap();
    
    match submission.status {
        ValidationStatus::Completed { .. } => assert!(true),
        _ => panic!("Validation did not complete successfully"),
    }
}

#[tokio::test]
async fn test_validation_timeout() {
    let mut context = TestContext::new().await;
    let owner = Keypair::new();
    let validator = Keypair::new();

    let model_type = ModelType::LLM {
        architecture: "transformer".to_string(),
        parameter_count: 1_000_000,
        context_window: 2048,
    };

    let submission_pubkey = context
        .create_test_model_submission(&owner, model_type)
        .await
        .unwrap();

    // Start validation
    let start_validation_ix = VetraInstruction::StartValidation {};
    let accounts = vec![
        AccountMeta::new(validator.pubkey(), true),
        AccountMeta::new(submission_pubkey, false),
        AccountMeta::new_readonly(solana_program::sysvar::clock::id(), false),
        AccountMeta::new_readonly(validator.pubkey(), false),
    ];

    let tx = Transaction::new_signed_with_payer(
        &[context.create_instruction(start_validation_ix, accounts)],
        Some(&context.payer.pubkey()),
        &[&context.payer, &validator],
        context.recent_blockhash,
    );

    context.banks_client.process_transaction(tx).await.unwrap();

    // Advance clock beyond timeout
    context.advance_clock(vetra_core::VALIDATION_TIMEOUT_SLOTS).await;

    // Attempt to submit validation after timeout
    let metrics = context.create_test_validation_metrics().await;
    let result = context
        .submit_validation_result(&validator, &submission_pubkey, metrics)
        .await;

    assert!(matches!(
        result.unwrap_err(),
        BanksClientError::TransactionError(TransactionError::InstructionError(
            0,
            InstructionError::Custom(error_code)
        )) if error_code == VetraError::ValidationTimeout as u32
    ));
}

#[tokio::test]
async fn test_unauthorized_validator() {
    let mut context = TestContext::new().await;
    let owner = Keypair::new();
    let authorized_validator = Keypair::new();
    let unauthorized_validator = Keypair::new();

    let model_type = ModelType::LLM {
        architecture: "transformer".to_string(),
        parameter_count: 1_000_000,
        context_window: 2048,
    };

    let submission_pubkey = context
        .create_test_model_submission(&owner, model_type)
        .await
        .unwrap();

    // Start validation with authorized validator
    let start_validation_ix = VetraInstruction::StartValidation {};
    let accounts = vec![
        AccountMeta::new(authorized_validator.pubkey(), true),
        AccountMeta::new(submission_pubkey, false),
        AccountMeta::new_readonly(solana_program::sysvar::clock::id(), false),
        AccountMeta::new_readonly(authorized_validator.pubkey(), false),
    ];

    let tx = Transaction::new_signed_with_payer(
        &[context.create_instruction(start_validation_ix, accounts)],
        Some(&context.payer.pubkey()),
        &[&context.payer, &authorized_validator],
        context.recent_blockhash,
    );

    context.banks_client.process_transaction(tx).await.unwrap();

    // Attempt submission with unauthorized validator
    let metrics = context.create_test_validation_metrics().await;
    let result = context
        .submit_validation_result(&unauthorized_validator, &submission_pubkey, metrics)
        .await;

    assert!(matches!(
        result.unwrap_err(),
        BanksClientError::TransactionError(TransactionError::InstructionError(
            0,
            InstructionError::Custom(error_code)
        )) if error_code == VetraError::UnauthorizedValidator as u32
    ));
}

#[tokio::test]
async fn test_duplicate_validation_submission() {
    let mut context = TestContext::new().await;
    let owner = Keypair::new();
    let validator = Keypair::new();

    let model_type = ModelType::LLM {
        architecture: "transformer".to_string(),
        parameter_count: 1_000_000,
        context_window: 2048,
    };

    let submission_pubkey = context
        .create_test_model_submission(&owner, model_type)
        .await
        .unwrap();

    // Start validation
    let start_validation_ix = VetraInstruction::StartValidation {};
    let accounts = vec![
        AccountMeta::new(validator.pubkey(), true),
        AccountMeta::new(submission_pubkey, false),
        AccountMeta::new_readonly(solana_program::sysvar::clock::id(), false),
        AccountMeta::new_readonly(validator.pubkey(), false),
    ];

    let tx = Transaction::new_signed_with_payer(
        &[context.create_instruction(start_validation_ix, accounts)],
        Some(&context.payer.pubkey()),
        &[&context.payer, &validator],
        context.recent_blockhash,
    );

    context.banks_client.process_transaction(tx).await.unwrap();

    // Submit first validation
    let metrics = context.create_test_validation_metrics().await;
    context
        .submit_validation_result(&validator, &submission_pubkey, metrics.clone())
        .await
        .unwrap();

    // Attempt duplicate submission
    let result = context
        .submit_validation_result(&validator, &submission_pubkey, metrics)
        .await;

    assert!(matches!(
        result.unwrap_err(),
        BanksClientError::TransactionError(TransactionError::InstructionError(
            0,
            InstructionError::Custom(error_code)
        )) if error_code == VetraError::DuplicateSubmission as u32
    ));
}

#[tokio::test]
async fn test_validation_cancellation() {
    let mut context = TestContext::new().await;
    let owner = Keypair::new();
    let validator = Keypair::new();

    let model_type = ModelType::LLM {
        architecture: "transformer".to_string(),
        parameter_count: 1_000_000,
        context_window: 2048,
    };

    let submission_pubkey = context
        .create_test_model_submission(&owner, model_type)
        .await
        .unwrap();

    // Start validation
    let start_validation_ix = VetraInstruction::StartValidation {};
    let accounts = vec![
        AccountMeta::new(validator.pubkey(), true),
        AccountMeta::new(submission_pubkey, false),
        AccountMeta::new_readonly(solana_program::sysvar::clock::id(), false),
        AccountMeta::new_readonly(validator.pubkey(), false),
    ];

    let tx = Transaction::new_signed_with_payer(
        &[context.create_instruction(start_validation_ix, accounts)],
        Some(&context.payer.pubkey()),
        &[&context.payer, &validator],
        context.recent_blockhash,
    );

    context.banks_client.process_transaction(tx).await.unwrap();

    // Cancel validation
    let cancel_ix = VetraInstruction::CancelValidation {};
    let cancel_accounts = vec![
        AccountMeta::new(owner.pubkey(), true),
        AccountMeta::new(submission_pubkey, false),
        AccountMeta::new_readonly(solana_program::sysvar::clock::id(), false),
    ];

    let cancel_tx = Transaction::new_signed_with_payer(
        &[context.create_instruction(cancel_ix, cancel_accounts)],
        Some(&context.payer.pubkey()),
        &[&context.payer, &owner],
        context.recent_blockhash,
    );

    context.banks_client.process_transaction(cancel_tx).await.unwrap();

    // Verify cancellation
    let account = context.get_account(&submission_pubkey).await.unwrap();
    let submission = vetra_core::state::ModelSubmission::unpack(&account.data).unwrap();
    
    match submission.status {
        ValidationStatus::Failed { .. } => assert!(true),
        _ => panic!("Validation was not cancelled properly"),
    }
}

#[tokio::test]
async fn test_consensus_threshold() {
    let mut context = TestContext::new().await;
    let owner = Keypair::new();
    let validators: Vec<Keypair> = (0..5).map(|_| Keypair::new()).collect();

    let model_type = ModelType::LLM {
        architecture: "transformer".to_string(),
        parameter_count: 1_000_000,
        context_window: 2048,
    };

    let submission_pubkey = context
        .create_test_model_submission(&owner, model_type)
        .await
        .unwrap();

    // Start validation with all validators
    let start_validation_ix = VetraInstruction::StartValidation {};
    let mut accounts = vec![
        AccountMeta::new(validators[0].pubkey(), true),
        AccountMeta::new(submission_pubkey, false),
        AccountMeta::new_readonly(solana_program::sysvar::clock::id(), false),
    ];
    accounts.extend(validators.iter().map(|v| AccountMeta::new_readonly(v.pubkey(), false)));

    let tx = Transaction::new_signed_with_payer(
        &[context.create_instruction(start_validation_ix, accounts)],
        Some(&context.payer.pubkey()),
        &[&context.payer, &validators[0]],
        context.recent_blockhash,
    );

    context.banks_client.process_transaction(tx).await.unwrap();

    // Submit validation results from 4 out of 5 validators
    let metrics = context.create_test_validation_metrics().await;
    for validator in validators.iter().take(4) {
        context
            .submit_validation_result(validator, &submission_pubkey, metrics.clone())
            .await
            .unwrap();
    }

    // Verify consensus reached
    let account = context.get_account(&submission_pubkey).await.unwrap();
    let submission = vetra_core::state::ModelSubmission::unpack(&account.data).unwrap();
    
    match submission.status {
        ValidationStatus::Completed { .. } => assert!(true),
        _ => panic!("Consensus was not reached properly"),
    }
}