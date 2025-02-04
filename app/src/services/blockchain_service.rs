use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc, NaiveDateTime};
use thiserror::Error;
use web3::types::{H256, U256, Address, TransactionReceipt};
use web3::contract::{Contract, Options};
use ethabi::{Token, encode};
use log::{debug, error, info, warn};

use crate::models::node::{
    ValidatorNode, NodeStatus, ValidationCapabilities,
    HardwareSpecs, StakingInfo, Delegation,
};
use crate::models::stats::{EconomicStats, NodeStats};
use crate::models::validation::ValidationTask;
use crate::models::consensus::ConsensusData;
use crate::config::BlockchainConfig;
use crate::utils::contracts::{load_contract_abi, verify_signature};

#[derive(Error, Debug)]
pub enum BlockchainError {
    #[error("Web3 error: {0}")]
    Web3Error(#[from] web3::Error),
    
    #[error("Contract error: {0}")]
    ContractError(String),
    
    #[error("Transaction error: {0}")]
    TransactionError(String),
    
    #[error("Invalid address: {0}")]
    InvalidAddress(String),
    
    #[error("Insufficient funds: required {0}, available {1}")]
    InsufficientFunds(U256, U256),
    
    #[error("Invalid stake amount: {0}")]
    InvalidStakeAmount(String),
    
    #[error("Node not found: {0}")]
    NodeNotFound(String),
}

pub struct BlockchainService {
    web3: web3::Web3<web3::transports::Http>,
    validator_contract: Contract<web3::transports::Http>,
    staking_contract: Contract<web3::transports::Http>,
    config: BlockchainConfig,
    state_cache: Arc<RwLock<StateCache>>,
}

struct StateCache {
    validators: Vec<ValidatorNode>,
    total_stake: U256,
    last_update: DateTime<Utc>,
}

#[derive(Debug)]
enum EconomicEvent {
    StakeDeposited {
        validator: Address,
        amount: U256,
        timestamp: u64,
    },
    StakeWithdrawn {
        validator: Address,
        amount: U256,
        timestamp: u64,
    },
    RewardClaimed {
        validator: Address,
        amount: U256,
        period: u64,
        timestamp: u64,
    },
    ValidatorSlashed {
        validator: Address,
        amount: U256,
        reason: String,
        timestamp: u64,
    },
    DelegatorRewardClaimed {
        delegator: Address,
        validator: Address,
        amount: U256,
        timestamp: u64,
    },
}

#[derive(Debug)]
struct SlashingEvent {
    validator: Address,
    amount: U256,
    reason: String,
    timestamp: DateTime<Utc>,
}

#[derive(Debug)]
pub struct NodeRegistrationResponse {
    pub node_id: String,
    pub registration_tx: String,
    pub stake_tx: String,
    pub status: NodeStatus,
}

impl BlockchainService {
    pub async fn new(config: BlockchainConfig) -> Result<Self, BlockchainError> {
        let transport = web3::transports::Http::new(&config.rpc_url)?;
        let web3 = web3::Web3::new(transport);

        let validator_abi = load_contract_abi("validator.json")?;
        let staking_abi = load_contract_abi("staking.json")?;

        let validator_contract = Contract::new(
            web3.eth(),
            config.validator_contract_address.parse()?,
            validator_abi,
        );

        let staking_contract = Contract::new(
            web3.eth(),
            config.staking_contract_address.parse()?,
            staking_abi,
        );

        Ok(Self {
            web3,
            validator_contract,
            staking_contract,
            config,
            state_cache: Arc::new(RwLock::new(StateCache {
                validators: Vec::new(),
                total_stake: U256::zero(),
                last_update: Utc::now(),
            })),
        })
    }

    pub async fn register_validator_node(
        &self,
        owner_address: &str,
        node_name: &str,
        hardware_specs: &HardwareSpecs,
        capabilities: &ValidationCapabilities,
        stake_amount: u64,
        delegation_enabled: bool,
        max_delegations: Option<u32>,
    ) -> Result<NodeRegistrationResponse, BlockchainError> {
        let owner = owner_address.parse::<Address>()
            .map_err(|_| BlockchainError::InvalidAddress(owner_address.to_string()))?;

        let min_stake = self.get_min_validator_stake().await?;
        if U256::from(stake_amount) < min_stake {
            return Err(BlockchainError::InvalidStakeAmount(
                format!("Stake amount below minimum requirement: {}", min_stake)
            ));
        }

        let params = encode(&[
            Token::String(node_name.to_string()),
            Token::String(serde_json::to_string(hardware_specs).unwrap()),
            Token::String(serde_json::to_string(capabilities).unwrap()),
            Token::Bool(delegation_enabled),
            Token::Uint(U256::from(max_delegations.unwrap_or(0))),
        ]);

        let tx = self.validator_contract
            .call("registerValidator", params, owner, Options::with(|opt| {
                opt.value = Some(U256::from(stake_amount));
            }))
            .await?;

        let receipt = self.web3.eth()
            .transaction_receipt(tx)
            .await?
            .ok_or_else(|| BlockchainError::TransactionError(
                "Transaction receipt not found".to_string()
            ))?;

        let node_id = self.parse_registration_event(&receipt)?;

        Ok(NodeRegistrationResponse {
            node_id,
            registration_tx: tx.to_string(),
            stake_tx: receipt.transaction_hash.to_string(),
            status: NodeStatus::Pending,
        })
    }

    pub async fn update_validator_node(
        &self,
        node_id: &str,
        hardware_specs: Option<&HardwareSpecs>,
        capabilities: Option<&ValidationCapabilities>,
        delegation_enabled: Option<bool>,
        max_delegations: Option<u32>,
    ) -> Result<ValidatorNode, BlockchainError> {
        let node = self.get_validator_node(node_id).await?
            .ok_or_else(|| BlockchainError::NodeNotFound(node_id.to_string()))?;

        let params = encode(&[
            Token::String(node_id.to_string()),
            Token::String(serde_json::to_string(&hardware_specs.unwrap_or(&node.hardware_specs)).unwrap()),
            Token::String(serde_json::to_string(&capabilities.unwrap_or(&node.capabilities)).unwrap()),
            Token::Bool(delegation_enabled.unwrap_or(node.delegation_enabled)),
            Token::Uint(U256::from(max_delegations.unwrap_or(node.max_delegations.unwrap_or(0)))),
        ]);

        let tx = self.validator_contract
            .call("updateValidator", params, node.owner, Options::default())
            .await?;

        self.web3.eth()
            .transaction_receipt(tx)
            .await?
            .ok_or_else(|| BlockchainError::TransactionError(
                "Transaction receipt not found".to_string()
            ))?;

        self.get_validator_node(node_id).await?
            .ok_or_else(|| BlockchainError::NodeNotFound(node_id.to_string()))
    }

    pub async fn get_validator_node(
        &self,
        node_id: &str,
    ) -> Result<Option<ValidatorNode>, BlockchainError> {
        {
            let cache = self.state_cache.read().await;
            if let Some(node) = cache.validators.iter().find(|n| n.id == node_id) {
                return Ok(Some(node.clone()));
            }
        }

        let result = self.validator_contract
            .query("getValidator", node_id, None, Options::default(), None)
            .await?;

        self.parse_validator_data(result)
    }

    pub async fn list_validator_nodes(
        &self,
        status: Option<&NodeStatus>,
        min_stake: Option<u64>,
        capabilities: Option<&Vec<String>>,
        delegation_enabled: Option<bool>,
        page: Option<u32>,
        per_page: Option<u32>,
    ) -> Result<Vec<ValidatorNode>, BlockchainError> {
        self.refresh_state_cache().await?;

        let cache = self.state_cache.read().await;
        let mut nodes = cache.validators.clone();

        if let Some(status) = status {
            nodes.retain(|n| &n.status == status);
        }

        if let Some(min_stake) = min_stake {
            nodes.retain(|n| n.total_stake >= U256::from(min_stake));
        }

        if let Some(caps) = capabilities {
            nodes.retain(|n| caps.iter().all(|c| n.capabilities.supported_frameworks.contains(c)));
        }

        if let Some(delegation) = delegation_enabled {
            nodes.retain(|n| n.delegation_enabled == delegation);
        }

        let page = page.unwrap_or(1) as usize;
        let per_page = per_page.unwrap_or(10) as usize;
        let start = (page - 1) * per_page;
        let end = start + per_page;

        Ok(nodes[start..end.min(nodes.len())].to_vec())
    }

    pub async fn get_validator_stake_info(
        &self,
        node_id: &str,
    ) -> Result<Option<StakingInfo>, BlockchainError> {
        let result = self.staking_contract
            .query("getValidatorStake", node_id, None, Options::default(), None)
            .await?;

        self.parse_staking_info(result)
    }

    pub async fn get_min_validator_stake(&self) -> Result<U256, BlockchainError> {
        self.validator_contract
            .query("minValidatorStake", (), None, Options::default(), None)
            .await
            .map_err(BlockchainError::Web3Error)
    }

    pub async fn submit_validation_task(
        &self,
        validator: &str,
        task: ValidationTask,
    ) -> Result<H256, BlockchainError> {
        let params = encode(&[
            Token::String(validator.to_string()),
            Token::String(serde_json::to_string(&task).unwrap()),
        ]);

        self.validator_contract
            .call("submitValidationTask", params, None, Options::default())
            .await
            .map_err(BlockchainError::Web3Error)
    }

    pub async fn submit_validation_result(
        &self,
        task_id: &str,
        result: ConsensusData,
        signature: &[u8],
    ) -> Result<H256, BlockchainError> {
        let params = encode(&[
            Token::String(task_id.to_string()),
            Token::String(serde_json::to_string(&result).unwrap()),
            Token::Bytes(signature.to_vec()),
        ]);

        self.validator_contract
            .call("submitValidationResult", params, None, Options::default())
            .await
            .map_err(BlockchainError::Web3Error)
    }

    async fn refresh_state_cache(&self) -> Result<(), BlockchainError> {
        let mut cache = self.state_cache.write().await;
        if (Utc::now() - cache.last_update).num_seconds() < 60 {
            return Ok(());
        }

        let total_validators: U256 = self.validator_contract
            .query("getTotalValidators", (), None, Options::default(), None)
            .await?;

        let mut validators = Vec::with_capacity(total_validators.as_usize());
        let mut total_stake = U256::zero();

        for i in 0..total_validators.as_u64() {
            if let Ok(Some(validator)) = self.get_validator_by_index(i).await {
                total_stake += validator.total_stake;
                validators.push(validator);
            }
        }

        cache.validators = validators;
        cache.total_stake = total_stake;
        cache.last_update = Utc::now();

        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use tokio::test;
        use std::str::FromStr;
        use hex_literal::hex;

        #[test]
        async fn test_register_validator_node() {
            let mock_transport = web3::transports::Http::new("http://localhost:8545").unwrap();
            let web3 = web3::Web3::new(mock_transport);

            let validator_address = Address::from(hex!("1234567890123456789012345678901234567890"));
            let staking_address = Address::from(hex!("0987654321098765432109876543210987654321"));

            let config = BlockchainConfig {
                rpc_url: "http://localhost:8545".to_string(),
                validator_contract_address: format!("{:?}", validator_address),
                staking_contract_address: format!("{:?}", staking_address),
                min_stake: 1000,
            };

            let service = BlockchainService::new(config).await.unwrap();
            let owner = Address::from(hex!("5555555555555555555555555555555555555555"));

            let hardware_specs = HardwareSpecs {
                cpu_cores: 8,
                memory_gb: 32,
                storage_gb: 1000,
                gpu_model: Some("NVIDIA A100".to_string()),
            };

            let capabilities = ValidationCapabilities {
                max_model_size: 1000000000,
                supported_frameworks: vec!["tensorflow".to_string(), "pytorch".to_string()],
                concurrent_validations: 5,
            };

            let result = service.register_validator_node(
                &format!("{:?}", owner),
                "test_node",
                &hardware_specs,
                &capabilities,
                5000,
                true,
                Some(10),
            ).await;

            assert!(result.is_ok());
            let response = result.unwrap();
            assert_eq!(response.status, NodeStatus::Pending);
            assert!(response.registration_tx.starts_with("0x"));
        }

        #[test]
        async fn test_get_validator_node() {
            let mock_transport = web3::transports::Http::new("http://localhost:8545").unwrap();
            let web3 = web3::Web3::new(mock_transport);

            let config = BlockchainConfig {
                rpc_url: "http://localhost:8545".to_string(),
                validator_contract_address: "0x1234567890123456789012345678901234567890".to_string(),
                staking_contract_address: "0x0987654321098765432109876543210987654321".to_string(),
                min_stake: 1000,
            };

            let service = BlockchainService::new(config).await.unwrap();
            let node_id = "test_node";

            let result = service.get_validator_node(node_id).await;
            assert!(result.is_ok());
        }

        #[test]
        async fn test_list_validator_nodes() {
            let mock_transport = web3::transports::Http::new("http://localhost:8545").unwrap();
            let web3 = web3::Web3::new(mock_transport);

            let config = BlockchainConfig {
                rpc_url: "http://localhost:8545".to_string(),
                validator_contract_address: "0x1234567890123456789012345678901234567890".to_string(),
                staking_contract_address: "0x0987654321098765432109876543210987654321".to_string(),
                min_stake: 1000,
            };

            let service = BlockchainService::new(config).await.unwrap();

            let result = service.list_validator_nodes(
                Some(&NodeStatus::Active),
                Some(1000),
                Some(&vec!["tensorflow".to_string()]),
                Some(true),
                Some(1),
                Some(10),
            ).await;

            assert!(result.is_ok());
            let nodes = result.unwrap();
            assert!(nodes.len() <= 10);
        }

        #[test]
        async fn test_get_validator_stake_info() {
            let mock_transport = web3::transports::Http::new("http://localhost:8545").unwrap();
            let web3 = web3::Web3::new(mock_transport);

            let config = BlockchainConfig {
                rpc_url: "http://localhost:8545".to_string(),
                validator_contract_address: "0x1234567890123456789012345678901234567890".to_string(),
                staking_contract_address: "0x0987654321098765432109876543210987654321".to_string(),
                min_stake: 1000,
            };

            let service = BlockchainService::new(config).await.unwrap();
            let node_id = "test_node";

            let result = service.get_validator_stake_info(node_id).await;
            assert!(result.is_ok());
        }

        #[test]
        async fn test_update_validator_node() {
            let mock_transport = web3::transports::Http::new("http://localhost:8545").unwrap();
            let web3 = web3::Web3::new(mock_transport);

            let config = BlockchainConfig {
                rpc_url: "http://localhost:8545".to_string(),
                validator_contract_address: "0x1234567890123456789012345678901234567890".to_string(),
                staking_contract_address: "0x0987654321098765432109876543210987654321".to_string(),
                min_stake: 1000,
            };

            let service = BlockchainService::new(config).await.unwrap();
            let node_id = "test_node";

            let capabilities = ValidationCapabilities {
                max_model_size: 2000000000,
                supported_frameworks: vec!["tensorflow".to_string(), "pytorch".to_string(), "onnx".to_string()],
                concurrent_validations: 10,
            };

            let result = service.update_validator_node(
                node_id,
                None,
                Some(&capabilities),
                Some(true),
                Some(20),
            ).await;

            assert!(result.is_ok());
        }

        #[test]
        async fn test_submit_validation_task() {
            let mock_transport = web3::transports::Http::new("http://localhost:8545").unwrap();
            let web3 = web3::Web3::new(mock_transport);

            let config = BlockchainConfig {
                rpc_url: "http://localhost:8545".to_string(),
                validator_contract_address: "0x1234567890123456789012345678901234567890".to_string(),
                staking_contract_address: "0x0987654321098765432109876543210987654321".to_string(),
                min_stake: 1000,
            };

            let service = BlockchainService::new(config).await.unwrap();

            let task = ValidationTask {
                model_id: "model1".to_string(),
                model_hash: "0x1234...".to_string(),
                validation_type: "comprehensive".to_string(),
                parameters: serde_json::json!({
                    "batch_size": 32,
                    "iterations": 1000,
                }),
                deadline: Utc::now() + chrono::Duration::hours(1),
            };

            let result = service.submit_validation_task("validator1", task).await;
            assert!(result.is_ok());
        }

        #[test]
        async fn test_submit_validation_result() {
            let mock_transport = web3::transports::Http::new("http://localhost:8545").unwrap();
            let web3 = web3::Web3::new(mock_transport);

            let config = BlockchainConfig {
                rpc_url: "http://localhost:8545".to_string(),
                validator_contract_address: "0x1234567890123456789012345678901234567890".to_string(),
                staking_contract_address: "0x0987654321098765432109876543210987654321".to_string(),
                min_stake: 1000,
            };

            let service = BlockchainService::new(config).await.unwrap();

            let consensus_data = ConsensusData {
                task_id: "task1".to_string(),
                model_id: "model1".to_string(),
                result: serde_json::json!({
                    "accuracy": 0.95,
                    "loss": 0.05,
                    "convergence_time": 1200,
                }),
                validators: vec!["validator1".to_string(), "validator2".to_string()],
                signatures: vec![vec![1, 2, 3], vec![4, 5, 6]],
                timestamp: Utc::now(),
            };

            let signature = vec![1, 2, 3, 4];
            let result = service.submit_validation_result("task1", consensus_data, &signature).await;
            assert!(result.is_ok());
        }
    }
}