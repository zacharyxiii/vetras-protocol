use std::sync::Arc;
use tokio::sync::RwLock;
use solana_client::rpc_client::RpcClient;
use solana_sdk::{
    commitment_config::CommitmentConfig,
    pubkey::Pubkey,
    signature::{Keypair, Signature},
    signer::Signer,
    transaction::Transaction,
};
use thiserror::Error;

use crate::{
    types::{
        ModelSubmission, ModelType, ValidationStatus,
        ValidationMetrics, StorageInfo, AccessControl,
    },
    utils::{derive_submission_address, create_instruction},
};

#[derive(Error, Debug)]
pub enum ClientError {
    #[error("RPC error: {0}")]
    RpcError(#[from] solana_client::client_error::ClientError),
    
    #[error("Program error: {0}")]
    ProgramError(#[from] solana_sdk::program_error::ProgramError),
    
    #[error("Validation error: {0}")]
    ValidationError(String),
    
    #[error("Storage error: {0}")]
    StorageError(String),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
}

pub type Result<T> = std::result::Result<T, ClientError>;

/// Configuration for the VETRAS client
#[derive(Clone, Debug)]
pub struct ClientConfig {
    /// RPC endpoint URL
    pub rpc_url: String,
    /// Program ID
    pub program_id: Pubkey,
    /// Commitment level
    pub commitment: CommitmentConfig,
    /// Validation timeout in seconds
    pub validation_timeout: u64,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            rpc_url: "https://api.mainnet-beta.solana.com".to_string(),
            program_id: Pubkey::new_unique(), // Replace with actual program ID
            commitment: CommitmentConfig::confirmed(),
            validation_timeout: 3600,
        }
    }
}

/// VETRAS client for interacting with the validation network
pub struct VetrasClient {
    config: ClientConfig,
    rpc_client: Arc<RpcClient>,
    payer: Arc<Keypair>,
    cache: Arc<RwLock<lru::LruCache<Pubkey, ModelSubmission>>>,
}

impl VetrasClient {
    /// Create a new VETRAS client instance
    pub fn new(config: ClientConfig, payer: Keypair) -> Self {
        Self {
            rpc_client: Arc::new(RpcClient::new_with_commitment(
                config.rpc_url.clone(),
                config.commitment,
            )),
            config,
            payer: Arc::new(payer),
            cache: Arc::new(RwLock::new(lru::LruCache::new(100))),
        }
    }

    /// Submit a new model for validation
    pub async fn submit_model(
        &self,
        model_type: ModelType,
        metadata: String,
        storage_info: StorageInfo,
        access_control: Option<AccessControl>,
    ) -> Result<Pubkey> {
        // Derive submission address
        let (submission_address, _) = derive_submission_address(
            &self.payer.pubkey(),
            &storage_info.identifier,
            &self.config.program_id,
        );

        // Create submission instruction
        let ix = create_instruction::init_model_submission(
            &self.config.program_id,
            &self.payer.pubkey(),
            &submission_address,
            model_type,
            metadata,
            storage_info,
            access_control.unwrap_or_default(),
        )?;

        // Create and send transaction
        let recent_blockhash = self.rpc_client.get_latest_blockhash()?;
        let tx = Transaction::new_signed_with_payer(
            &[ix],
            Some(&self.payer.pubkey()),
            &[&*self.payer],
            recent_blockhash,
        );

        self.rpc_client.send_and_confirm_transaction(&tx)?;
        Ok(submission_address)
    }

    /// Start validation process for a model
    pub async fn start_validation(
        &self,
        submission_address: &Pubkey,
        validators: Vec<Pubkey>,
    ) -> Result<Signature> {
        // Create start validation instruction
        let ix = create_instruction::start_validation(
            &self.config.program_id,
            &self.payer.pubkey(),
            submission_address,
            &validators,
        )?;

        // Create and send transaction
        let recent_blockhash = self.rpc_client.get_latest_blockhash()?;
        let tx = Transaction::new_signed_with_payer(
            &[ix],
            Some(&self.payer.pubkey()),
            &[&*self.payer],
            recent_blockhash,
        );

        let signature = self.rpc_client.send_and_confirm_transaction(&tx)?;
        Ok(signature)
    }

    /// Submit validation results
    pub async fn submit_validation_result(
        &self,
        submission_address: &Pubkey,
        metrics: ValidationMetrics,
        signature: [u8; 64],
    ) -> Result<Signature> {
        let ix = create_instruction::submit_validation(
            &self.config.program_id,
            &self.payer.pubkey(),
            submission_address,
            metrics,
            signature,
        )?;

        let recent_blockhash = self.rpc_client.get_latest_blockhash()?;
        let tx = Transaction::new_signed_with_payer(
            &[ix],
            Some(&self.payer.pubkey()),
            &[&*self.payer],
            recent_blockhash,
        );

        let signature = self.rpc_client.send_and_confirm_transaction(&tx)?;
        Ok(signature)
    }

    /// Get model submission details
    pub async fn get_submission(&self, address: &Pubkey) -> Result<ModelSubmission> {
        // Check cache first
        if let Some(cached) = self.cache.read().await.get(address) {
            return Ok(cached.clone());
        }

        // Fetch from chain
        let account = self.rpc_client.get_account(address)?;
        let submission = ModelSubmission::try_from_slice(&account.data)?;

        // Update cache
        self.cache.write().await.put(*address, submission.clone());
        Ok(submission)
    }

    /// Update access control settings
    pub async fn update_access_control(
        &self,
        submission_address: &Pubkey,
        new_access_control: AccessControl,
    ) -> Result<Signature> {
        let ix = create_instruction::update_access_control(
            &self.config.program_id,
            &self.payer.pubkey(),
            submission_address,
            new_access_control,
        )?;

        let recent_blockhash = self.rpc_client.get_latest_blockhash()?;
        let tx = Transaction::new_signed_with_payer(
            &[ix],
            Some(&self.payer.pubkey()),
            &[&*self.payer],
            recent_blockhash,
        );

        let signature = self.rpc_client.send_and_confirm_transaction(&tx)?;
        Ok(signature)
    }

    /// Cancel ongoing validation
    pub async fn cancel_validation(&self, submission_address: &Pubkey) -> Result<Signature> {
        let ix = create_instruction::cancel_validation(
            &self.config.program_id,
            &self.payer.pubkey(),
            submission_address,
        )?;

        let recent_blockhash = self.rpc_client.get_latest_blockhash()?;
        let tx = Transaction::new_signed_with_payer(
            &[ix],
            Some(&self.payer.pubkey()),
            &[&*self.payer],
            recent_blockhash,
        );

        let signature = self.rpc_client.send_and_confirm_transaction(&tx)?;
        Ok(signature)
    }

    /// Wait for validation completion
    pub async fn wait_for_validation(
        &self,
        submission_address: &Pubkey,
        timeout: Option<u64>,
    ) -> Result<ValidationStatus> {
        let timeout = timeout.unwrap_or(self.config.validation_timeout);
        let start = std::time::Instant::now();

        loop {
            let submission = self.get_submission(submission_address).await?;
            match submission.status {
                ValidationStatus::Completed { .. } => return Ok(submission.status),
                ValidationStatus::Failed { .. } => return Ok(submission.status),
                _ => {
                    if start.elapsed().as_secs() > timeout {
                        return Err(ClientError::ValidationError("Validation timeout".to_string()));
                    }
                    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use solana_sdk::signer::keypair::Keypair;

    fn setup_test_client() -> VetrasClient {
        let config = ClientConfig {
            rpc_url: "http://localhost:8899".to_string(),
            program_id: Pubkey::new_unique(),
            commitment: CommitmentConfig::processed(),
            validation_timeout: 60,
        };
        let payer = Keypair::new();
        VetrasClient::new(config, payer)
    }

    #[tokio::test]
    async fn test_submit_model() {
        let client = setup_test_client();
        let model_type = ModelType::LLM {
            architecture: "transformer".to_string(),
            parameter_count: 1_000_000,
            context_window: 2048,
        };
        let storage_info = StorageInfo {
            protocol: crate::types::StorageProtocol::IPFS,
            identifier: "test".to_string(),
            size: 1000,
            checksum: [0; 32],
        };

        let result = client.submit_model(
            model_type,
            "test metadata".to_string(),
            storage_info,
            None,
        ).await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_get_submission() {
        let client = setup_test_client();
        let address = Pubkey::new_unique();
        
        // This test would need a local validator setup
        let result = client.get_submission(&address).await;
        assert!(result.is_err()); // Should fail without local validator
    }

    // Add more tests...
}
