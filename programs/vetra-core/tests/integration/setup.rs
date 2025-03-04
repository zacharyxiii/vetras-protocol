use solana_program::{
    account_info::AccountInfo,
    clock::Clock,
    instruction::{AccountMeta, Instruction},
    program_pack::Pack,
    pubkey::Pubkey,
    rent::Rent,
    system_instruction,
};
use solana_program_test::*;
use solana_sdk::{
    account::Account,
    signature::{Keypair, Signer},
    transaction::Transaction,
};
use std::str::FromStr;

use vetras_core::{
    processor::VetrasInstruction,
    state::{
        AccessControl, BiasMetrics, ModelSubmission, ModelType, PerformanceMetrics,
        ResourceMetrics, SafetyMetrics, StorageInfo, StorageProtocol, ValidationMetrics,
        ValidationStatus,
    },
};

pub struct TestContext {
    pub program_id: Pubkey,
    pub banks_client: BanksClient,
    pub payer: Keypair,
    pub recent_blockhash: Hash,
    pub rent: Rent,
    pub clock: Clock,
}

pub async fn setup_program_test() -> (ProgramTest, Keypair) {
    let program_id = Pubkey::from_str("Vetras111111111111111111111111111111111111111").unwrap();
    let mut program_test = ProgramTest::new(
        "vetras_core",
        program_id,
        processor!(vetras_core::processor::process_instruction),
    );

    // Create test validator account with initial balance
    let validator = Keypair::new();
    program_test.add_account(
        validator.pubkey(),
        Account {
            lamports: 1_000_000_000,
            data: vec![],
            owner: solana_program::system_program::id(),
            executable: false,
            rent_epoch: 0,
        },
    );

    (program_test, validator)
}

impl TestContext {
    pub async fn new() -> Self {
        let (mut program_test, payer) = setup_program_test().await;
        let (banks_client, payer, recent_blockhash) = program_test.start().await;

        let rent = banks_client
            .get_rent()
            .await
            .expect("Failed to get rent");

        let clock = Clock::default();

        Self {
            program_id: Pubkey::from_str("Vetras111111111111111111111111111111111111111").unwrap(),
            banks_client,
            payer,
            recent_blockhash,
            rent,
            clock,
        }
    }

    pub async fn create_test_model_submission(
        &mut self,
        owner: &Keypair,
        model_type: ModelType,
    ) -> Result<Pubkey, BanksClientError> {
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

        let space = ModelSubmission::get_packed_len();
        let rent = self.rent.minimum_balance(space);

        let submission_keypair = Keypair::new();
        let create_account_ix = system_instruction::create_account(
            &self.payer.pubkey(),
            &submission_keypair.pubkey(),
            rent,
            space as u64,
            &self.program_id,
        );

        let init_ix = Instruction {
            program_id: self.program_id,
            accounts: vec![
                AccountMeta::new(owner.pubkey(), true),
                AccountMeta::new(submission_keypair.pubkey(), false),
                AccountMeta::new_readonly(solana_program::system_program::id(), false),
                AccountMeta::new_readonly(solana_program::sysvar::rent::id(), false),
            ],
            data: VetrasInstruction::InitModelSubmission {
                model_type,
                metadata: "test".to_string(),
                storage_info,
                access_control,
            }
            .try_to_vec()
            .unwrap(),
        };

        let mut transaction = Transaction::new_with_payer(
            &[create_account_ix, init_ix],
            Some(&self.payer.pubkey()),
        );
        transaction.sign(
            &[&self.payer, &submission_keypair, owner],
            self.recent_blockhash,
        );

        self.banks_client
            .process_transaction(transaction)
            .await
            .map(|_| submission_keypair.pubkey())
    }

    pub async fn create_test_validation_metrics(&self) -> ValidationMetrics {
        ValidationMetrics {
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
        }
    }

    pub async fn submit_validation_result(
        &mut self,
        validator: &Keypair,
        submission_pubkey: &Pubkey,
        metrics: ValidationMetrics,
    ) -> Result<(), BanksClientError> {
        let submit_ix = Instruction {
            program_id: self.program_id,
            accounts: vec![
                AccountMeta::new(validator.pubkey(), true),
                AccountMeta::new(*submission_pubkey, false),
                AccountMeta::new_readonly(solana_program::sysvar::clock::id(), false),
            ],
            data: VetrasInstruction::SubmitValidation {
                metrics,
                signature: [0; 64], // Test signature
            }
            .try_to_vec()
            .unwrap(),
        };

        let mut transaction =
            Transaction::new_with_payer(&[submit_ix], Some(&self.payer.pubkey()));
        transaction.sign(&[&self.payer, validator], self.recent_blockhash);

        self.banks_client.process_transaction(transaction).await
    }

    pub async fn get_account(&mut self, pubkey: &Pubkey) -> Result<Account, BanksClientError> {
        self.banks_client
            .get_account(*pubkey)
            .await?
            .ok_or(BanksClientError::ClientError("Account not found".into()))
    }

    pub async fn advance_clock(&mut self, slots: u64) {
        // Simulated clock advancement for testing timeouts and delays
        self.clock.slot += slots;
        self.clock.unix_timestamp += (slots * solana_program::clock::DEFAULT_MS_PER_SLOT) as i64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_context_creation() {
        let context = TestContext::new().await;
        assert_eq!(
            context.program_id,
            Pubkey::from_str("Vetras111111111111111111111111111111111111111").unwrap()
        );
    }

    #[tokio::test]
    async fn test_model_submission_creation() {
        let mut context = TestContext::new().await;
        let owner = Keypair::new();

        let model_type = ModelType::LLM {
            architecture: "transformer".to_string(),
            parameter_count: 1_000_000,
            context_window: 2048,
        };

        let submission_pubkey = context
            .create_test_model_submission(&owner, model_type)
            .await
            .expect("Failed to create model submission");

        let account = context
            .get_account(&submission_pubkey)
            .await
            .expect("Failed to get account");
        assert!(account.data.len() > 0);
    }

    #[tokio::test]
    async fn test_validation_metrics_creation() {
        let context = TestContext::new().await;
        let metrics = context.create_test_validation_metrics().await;
        assert_eq!(metrics.performance.latency, 100);
        assert_eq!(metrics.safety.safety_score, 95);
    }
}