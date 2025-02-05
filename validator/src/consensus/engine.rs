use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, RwLock, mpsc};
use anyhow::{Result, Context, anyhow};
use serde::{Serialize, Deserialize};
use ed25519_dalek::{Keypair, PublicKey, Signature, Signer, Verifier};
use rand::rngs::OsRng;
use metrics::{counter, gauge, histogram};
use tracing::{info, warn, error};

const CONSENSUS_THRESHOLD: f64 = 0.67; // 2/3 majority required
const VALIDATION_TIMEOUT: Duration = Duration::from_secs(300); // 5 minutes
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);
const MAX_VALIDATION_BATCH_SIZE: usize = 10;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationProposal {
    pub model_id: String,
    pub validator: PublicKey,
    pub timestamp: i64,
    pub results: ValidationResults,
    pub signature: Signature,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResults {
    pub metrics: ModelMetrics,
    pub status: ValidationStatus,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    pub accuracy: f64,
    pub latency_ms: f64,
    pub memory_usage_mb: f64,
    pub throughput_qps: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ValidationStatus {
    Valid,
    Invalid,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusMessage {
    pub message_type: MessageType,
    pub sender: PublicKey,
    pub signature: Signature,
    pub timestamp: i64,
    pub payload: MessagePayload,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageType {
    Proposal,
    Vote,
    Commit,
    Heartbeat,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessagePayload {
    ValidationProposal(ValidationProposal),
    ValidationVote {
        proposal_id: String,
        approve: bool,
    },
    ValidationCommit {
        proposal_id: String,
        final_result: ValidationResults,
    },
    HeartbeatPing {
        node_status: NodeStatus,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeStatus {
    pub uptime: Duration,
    pub active_validations: usize,
    pub completed_validations: u64,
    pub memory_usage: f64,
}

pub struct ConsensusEngine {
    keypair: Keypair,
    active_validators: Arc<RwLock<HashSet<PublicKey>>>,
    active_proposals: Arc<RwLock<HashMap<String, ProposalState>>>,
    completed_validations: Arc<RwLock<HashMap<String, ValidationResults>>>,
    message_tx: broadcast::Sender<ConsensusMessage>,
    message_rx: broadcast::Receiver<ConsensusMessage>,
    validation_tx: mpsc::Sender<ValidationResults>,
    node_status: Arc<RwLock<NodeStatus>>,
    start_time: Instant,
}

struct ProposalState {
    proposal: ValidationProposal,
    votes: HashMap<PublicKey, bool>,
    created_at: Instant,
}

impl ConsensusEngine {
    pub fn new(validation_tx: mpsc::Sender<ValidationResults>) -> Result<Self> {
        let keypair = Keypair::generate(&mut OsRng);
        let (message_tx, message_rx) = broadcast::channel(1000);
        
        Ok(Self {
            keypair,
            active_validators: Arc::new(RwLock::new(HashSet::new())),
            active_proposals: Arc::new(RwLock::new(HashMap::new())),
            completed_validations: Arc::new(RwLock::new(HashMap::new())),
            message_tx,
            message_rx,
            validation_tx,
            node_status: Arc::new(RwLock::new(NodeStatus {
                uptime: Duration::from_secs(0),
                active_validations: 0,
                completed_validations: 0,
                memory_usage: 0.0,
            })),
            start_time: Instant::now(),
        })
    }

    pub async fn start(&self) -> Result<()> {
        let message_tx = self.message_tx.clone();
        let message_rx = self.message_rx.resubscribe();
        let node_status = self.node_status.clone();
        let start_time = self.start_time;

        // Start heartbeat mechanism
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(HEARTBEAT_INTERVAL).await;
                let uptime = start_time.elapsed();
                
                if let Ok(mut status) = node_status.write().await {
                    status.uptime = uptime;
                    status.memory_usage = Self::get_memory_usage();
                    
                    let heartbeat = ConsensusMessage {
                        message_type: MessageType::Heartbeat,
                        sender: self.keypair.public,
                        signature: self.sign_message(&status),
                        timestamp: chrono::Utc::now().timestamp(),
                        payload: MessagePayload::HeartbeatPing {
                            node_status: status.clone(),
                        },
                    };
                    
                    if let Err(e) = message_tx.send(heartbeat) {
                        error!("Failed to send heartbeat: {}", e);
                    }
                }
            }
        });

        // Handle incoming messages
        loop {
            match message_rx.recv().await {
                Ok(message) => {
                    if !self.verify_message(&message) {
                        warn!("Received message with invalid signature");
                        continue;
                    }

                    match message.message_type {
                        MessageType::Proposal => {
                            self.handle_proposal(message).await?;
                        }
                        MessageType::Vote => {
                            self.handle_vote(message).await?;
                        }
                        MessageType::Commit => {
                            self.handle_commit(message).await?;
                        }
                        MessageType::Heartbeat => {
                            self.handle_heartbeat(message).await?;
                        }
                    }
                }
                Err(e) => {
                    error!("Error receiving message: {}", e);
                }
            }
        }
    }

    pub async fn propose_validation(&self, proposal: ValidationProposal) -> Result<()> {
        let message = ConsensusMessage {
            message_type: MessageType::Proposal,
            sender: self.keypair.public,
            signature: self.sign_message(&proposal),
            timestamp: chrono::Utc::now().timestamp(),
            payload: MessagePayload::ValidationProposal(proposal.clone()),
        };

        counter!("vetras.consensus.proposals", 1);
        self.message_tx.send(message)
            .context("Failed to broadcast proposal")?;

        Ok(())
    }

    async fn handle_proposal(&self, message: ConsensusMessage) -> Result<()> {
        if let MessagePayload::ValidationProposal(proposal) = message.payload {
            let proposal_id = Self::compute_proposal_id(&proposal);
            
            // Check if we're already processing this proposal
            let mut proposals = self.active_proposals.write().await;
            if proposals.contains_key(&proposal_id) {
                return Ok(());
            }

            // Initialize proposal state
            proposals.insert(proposal_id.clone(), ProposalState {
                proposal: proposal.clone(),
                votes: HashMap::new(),
                created_at: Instant::now(),
            });

            // Vote on the proposal
            let vote_message = ConsensusMessage {
                message_type: MessageType::Vote,
                sender: self.keypair.public,
                signature: self.sign_message(&proposal_id),
                timestamp: chrono::Utc::now().timestamp(),
                payload: MessagePayload::ValidationVote {
                    proposal_id,
                    approve: self.validate_proposal(&proposal),
                },
            };

            counter!("vetras.consensus.votes", 1);
            self.message_tx.send(vote_message)
                .context("Failed to broadcast vote")?;
        }

        Ok(())
    }

    async fn handle_vote(&self, message: ConsensusMessage) -> Result<()> {
        if let MessagePayload::ValidationVote { proposal_id, approve } = message.payload {
            let mut proposals = self.active_proposals.write().await;
            
            if let Some(proposal_state) = proposals.get_mut(&proposal_id) {
                // Record the vote
                proposal_state.votes.insert(message.sender, approve);
                
                // Check if we have enough votes for consensus
                let total_validators = self.active_validators.read().await.len();
                let total_votes = proposal_state.votes.len();
                let positive_votes = proposal_state.votes.values().filter(|&&v| v).count();
                
                if total_votes >= (total_validators as f64 * CONSENSUS_THRESHOLD) as usize {
                    if positive_votes as f64 / total_votes as f64 >= CONSENSUS_THRESHOLD {
                        // Proposal accepted
                        let commit_message = ConsensusMessage {
                            message_type: MessageType::Commit,
                            sender: self.keypair.public,
                            signature: self.sign_message(&proposal_id),
                            timestamp: chrono::Utc::now().timestamp(),
                            payload: MessagePayload::ValidationCommit {
                                proposal_id: proposal_id.clone(),
                                final_result: proposal_state.proposal.results.clone(),
                            },
                        };

                        counter!("vetras.consensus.commits", 1);
                        self.message_tx.send(commit_message)
                            .context("Failed to broadcast commit")?;
                    }
                }
            }
        }

        Ok(())
    }

    async fn handle_commit(&self, message: ConsensusMessage) -> Result<()> {
        if let MessagePayload::ValidationCommit { proposal_id, final_result } = message.payload {
            let mut proposals = self.active_proposals.write().await;
            
            if let Some(_) = proposals.remove(&proposal_id) {
                let mut completed = self.completed_validations.write().await;
                completed.insert(proposal_id.clone(), final_result.clone());
                
                // Notify about completed validation
                if let Err(e) = self.validation_tx.send(final_result).await {
                    error!("Failed to send validation result: {}", e);
                }

                counter!("vetras.consensus.completed_validations", 1);
            }
        }

        Ok(())
    }

    async fn handle_heartbeat(&self, message: ConsensusMessage) -> Result<()> {
        if let MessagePayload::HeartbeatPing { node_status } = message.payload {
            let mut validators = self.active_validators.write().await;
            validators.insert(message.sender);
            
            gauge!("vetras.consensus.active_validators", validators.len() as f64);
        }

        Ok(())
    }

    fn validate_proposal(&self, proposal: &ValidationProposal) -> bool {
        // Implement validation logic here
        // For now, just check basic requirements
        proposal.results.metrics.accuracy >= 0.0 && 
        proposal.results.metrics.accuracy <= 1.0 &&
        proposal.results.metrics.latency_ms > 0.0
    }

    fn compute_proposal_id(proposal: &ValidationProposal) -> String {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        
        hasher.update(proposal.model_id.as_bytes());
        hasher.update(proposal.validator.as_bytes());
        hasher.update(proposal.timestamp.to_le_bytes());
        
        hex::encode(hasher.finalize())
    }

    fn sign_message<T: Serialize>(&self, message: &T) -> Signature {
        let message_bytes = bincode::serialize(message).unwrap();
        self.keypair.sign(&message_bytes)
    }

    fn verify_message(&self, message: &ConsensusMessage) -> bool {
        let message_bytes = bincode::serialize(&message).unwrap();
        message.sender.verify(&message_bytes, &message.signature).is_ok()
    }

    fn get_memory_usage() -> f64 {
        // Implementation would depend on platform
        // For now, return dummy value
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    async fn setup_test_engine() -> (ConsensusEngine, mpsc::Receiver<ValidationResults>) {
        let (tx, rx) = mpsc::channel(100);
        let engine = ConsensusEngine::new(tx).unwrap();
        (engine, rx)
    }

    #[tokio::test]
    async fn test_proposal_flow() {
        let (engine, mut rx) = setup_test_engine().await;
        
        // Create test proposal
        let proposal = ValidationProposal {
            model_id: "test_model".to_string(),
            validator: engine.keypair.public,
            timestamp: chrono::Utc::now().timestamp(),
            results: ValidationResults {
                metrics: ModelMetrics {
                    accuracy: 0.95,
                    latency_ms: 10.0,
                    memory_usage_mb: 100.0,
                    throughput_qps: 1000.0,
                },
                status: ValidationStatus::Valid,
                errors: vec![],
            },
            signature: engine.keypair.sign(&[0u8; 32]),
        };

        // Submit proposal
        engine.propose_validation(proposal.clone()).await.unwrap();

        // Verify proposal is tracked
        let proposals = engine.active_proposals.read().await;
        assert!(proposals.contains_key(&ConsensusEngine::compute_proposal_id(&proposal)));
    }

    #[tokio::test]
    async fn test_consensus_threshold() {
        let (engine, _) = setup_test_engine().await;
        
        // Add test validators
        let mut validators = engine.active_validators.write().await;
        for _ in 0..5 {
            validators.insert(Keypair::generate(&mut OsRng).public);
        }
        
        // Verify consensus threshold
        let total = validators.len();
        assert_eq!((total as f64 * CONSENSUS_THRESHOLD) as usize, 4);
    }
}
