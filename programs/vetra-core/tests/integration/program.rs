use solana_program::{
    instruction::{AccountMeta, Instruction},
    program_pack::Pack,
    pubkey::Pubkey,
    system_instruction,
    sysvar,
};
use solana_program_test::*;
use solana_sdk::{
    signature::{Keypair, Signer},
    transaction::Transaction,
};

use vetras_core::{
    processor::VetrasInstruction,
    state::{
        ModelSubmission, ModelType, StorageInfo, StorageProtocol,
        ValidationMetrics, ValidationStatus, AccessControl,
    },
    validation::{ValidationEngine, CONSENSUS_THRESHOLD},
};

pub struct TestContext {
    pub banks_client: BanksClient,
    pub payer: Keypair,
    pub last_blockhash: Hash,
    pub program_id: Pubkey,
}

impl TestContext {
    pub async fn new() -> Self {
        let program_id = Pubkey::new_unique();
        let (banks_client, payer, last_blockhash) = ProgramTest::new(
            "vetras_core",
            program_id,
            processor!(vetras_core::processor::process_instruction),
        )
        .start()
        .await;

        TestContext {
            banks_client,
            payer,
            last_blockhash,
            program_id,
        }
    }

    pub async fn submit_transaction(
        &mut self,
        instructions: Vec<Instruction>,
        signers: Vec<&Keypair>,
    ) -> Result<(), BanksClientError> {
        let mut transaction = Transaction::new_with_payer(
            &instructions,
            Some(&self.payer.pubkey()),
        );
        transaction.sign(&signers, self.last_blockhash);
        self.banks_client.process_transaction(transaction).await
    }

    pub async fn get_account(&mut self, address: &Pubkey) -> Option<Account> {
        self.banks_client.get_account(*address).await.unwrap()
    }
}

async fn create_test_submission(
    context: &mut TestContext,
    owner: &Keypair,
) -> Result<Pubkey, BanksClientError> {
    let model_type = ModelType::LLM {
        architecture: "transformer".to_string(),
        parameter_count: 1_000_000,
        context_window: 2048,
    };

    let storage_info = StorageInfo {
        protocol: StorageProtocol::IPFS,
        identifier: "test_id".to_string(),
        size: 1000,
        checksum: [0; 32],
    };

    let access_control = AccessControl {
        is_public: false,
        allowed_viewers: vec![],
        expires_at: None,
    };

    let submission_size = ModelSubmission::get_packed_len();
    let rent = context.banks_client.get_rent().await.unwrap();
    let rent_lamports = rent.minimum_balance(submission_size);

    let (submission_address, _) = Pubkey::find_program_address(
        &[
            owner.pubkey().as_ref(),
            storage_info.identifier.as_bytes(),
        ],
        &context.program_id,
    );

    let init_instruction = VetrasInstruction::InitModelSubmission {
        model_type,
        metadata: "test metadata".to_string(),
        storage_info,
        access_control,
    };

    let accounts = vec![
        AccountMeta::new(owner.pubkey(), true),
        AccountMeta::new(submission_address, false),
        AccountMeta::new_readonly(solana_program::system_program::id(), false),
        AccountMeta::new_readonly(sysvar::rent::id(), false),
    ];

    let instruction = Instruction::new_with_borsh(
        context.program_id,
        &init_instruction,
        accounts,
    );

    context.submit_transaction(vec![instruction], vec![owner]).await?;
    Ok(submission_address)
}

#[tokio::test]
async fn test_full_validation_flow() {
    let mut context = TestContext::new().await;
    let owner = Keypair::new();
    let validators: Vec<Keypair> = (0..3).map(|_| Keypair::new()).collect();

    // Fund accounts
    for validator in validators.iter() {
        let ix = system_instruction::transfer(
            &context.payer.pubkey(),
            &validator.pubkey(),
            1_000_000_000,
        );
        context.submit_transaction(vec![ix], vec![&context.payer]).await.unwrap();
    }

    // Create submission
    let submission_address = create_test_submission(&mut context, &owner)
        .await
        .unwrap();

    // Start validation
    let start_validation_ix = VetrasInstruction::StartValidation {};
    let mut accounts = vec![
        AccountMeta::new(validators[0].pubkey(), true),
        AccountMeta::new(submission_address, false),
        AccountMeta::new_readonly(sysvar::clock::id(), false),
    ];

    // Add validator accounts
    for validator in validators.iter() {
        accounts.push(AccountMeta::new_readonly(validator.pubkey(), false));
    }

    let instruction = Instruction::new_with_borsh(
        context.program_id,
        &start_validation_ix,
        accounts,
    );

    context.submit_transaction(vec![instruction], vec![&validators[0]])
        .await
        .unwrap();

    // Submit validation results
    let metrics = ValidationMetrics {
        performance: PerformanceMetrics {
            latency: 100,
            throughput: 1000,
            error_rate: 0.01,
            custom: std::collections::HashMap::new(),
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

    for validator in validators.iter() {
        let submit_validation_ix = VetrasInstruction::SubmitValidation {
            metrics: metrics.clone(),
            signature: [0; 64],
        };

        let accounts = vec![
            AccountMeta::new(validator.pubkey(), true),
            AccountMeta::new(submission_address, false),
            AccountMeta::new_readonly(sysvar::clock::id(), false),
        ];

        let instruction = Instruction::new_with_borsh(
            context.program_id,
            &submit_validation_ix,
            accounts,
        );

        context.submit_transaction(vec![instruction], vec![validator])
            .await
            .unwrap();
    }

    // Verify final state
    let account = context.get_account(&submission_address).await.unwrap();
    let submission = ModelSubmission::unpack(&account.data).unwrap();

    match submission.status {
        ValidationStatus::Completed { result, .. } => {
            assert!(result.score >= MIN_VALIDATION_SCORE);
            assert_eq!(result.consensus.validator_count, 3);
            assert!(result.consensus.threshold_achieved);
        }
        _ => panic!("Expected completed validation status"),
    }
}

#[tokio::test]
async fn test_validation_cancellation() {
    let mut context = TestContext::new().await;
    let owner = Keypair::new();
    
    // Create submission
    let submission_address = create_test_submission(&mut context, &owner)
        .await
        .unwrap();

    // Start validation
    let validator = Keypair::new();
    let start_validation_ix = VetrasInstruction::StartValidation {};
    let accounts = vec![
        AccountMeta::new(validator.pubkey(), true),
        AccountMeta::new(submission_address, false),
        AccountMeta::new_readonly(sysvar::clock::id(), false),
        AccountMeta::new_readonly(validator.pubkey(), false),
    ];

    let instruction = Instruction::new_with_borsh(
        context.program_id,
        &start_validation_ix,
        accounts,
    );

    context.submit_transaction(vec![instruction], vec![&validator])
        .await
        .unwrap();

    // Cancel validation
    let cancel_validation_ix = VetrasInstruction::CancelValidation {};
    let accounts = vec![
        AccountMeta::new(owner.pubkey(), true),
        AccountMeta::new(submission_address, false),
        AccountMeta::new_readonly(sysvar::clock::id(), false),
    ];

    let instruction = Instruction::new_with_borsh(
        context.program_id,
        &cancel_validation_ix,
        accounts,
    );

    context.submit_transaction(vec![instruction], vec![&owner])
        .await
        .unwrap();

    // Verify cancelled state
    let account = context.get_account(&submission_address).await.unwrap();
    let submission = ModelSubmission::unpack(&account.data).unwrap();

    match submission.status {
        ValidationStatus::Failed { .. } => (),
        _ => panic!("Expected failed validation status"),
    }
}

#[tokio::test]
async fn test_access_control_update() {
    let mut context = TestContext::new().await;
    let owner = Keypair::new();
    
    // Create submission
    let submission_address = create_test_submission(&mut context, &owner)
        .await
        .unwrap();

    // Update access control
    let new_access_control = AccessControl {
        is_public: true,
        allowed_viewers: vec![Pubkey::new_unique()],
        expires_at: Some(1000),
    };

    let update_access_ix = VetrasInstruction::UpdateAccessControl {
        new_access_control: new_access_control.clone(),
    };

    let accounts = vec![
        AccountMeta::new(owner.pubkey(), true),
        AccountMeta::new(submission_address, false),
    ];

    let instruction = Instruction::new_with_borsh(
        context.program_id,
        &update_access_ix,
        accounts,
    );

    context.submit_transaction(vec![instruction], vec![&owner])
        .await
        .unwrap();

    // Verify updated access control
    let account = context.get_account(&submission_address).await.unwrap();
    let submission = ModelSubmission::unpack(&account.data).unwrap();
    assert_eq!(submission.access_control, new_access_control);
}

#[tokio::test]
async fn test_unauthorized_access() {
    let mut context = TestContext::new().await;
    let owner = Keypair::new();
    let unauthorized = Keypair::new();
    
    // Create submission
    let submission_address = create_test_submission(&mut context, &owner)
        .await
        .unwrap();

    // Attempt unauthorized access control update
    let new_access_control = AccessControl {
        is_public: true,
        allowed_viewers: vec![],
        expires_at: None,
    };

    let update_access_ix = VetrasInstruction::UpdateAccessControl {
        new_access_control,
    };

    let accounts = vec![
        AccountMeta::new(unauthorized.pubkey(), true),
        AccountMeta::new(submission_address, false),
    ];

    let instruction = Instruction::new_with_borsh(
        context.program_id,
        &update_access_ix,
        accounts,
    );

    let result = context.submit_transaction(vec![instruction], vec![&unauthorized]).await;
    assert!(result.is_err());
}