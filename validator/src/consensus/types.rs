use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use ed25519_dalek::{PublicKey, Signature};
use chrono::{DateTime, Utc};

/// Constants for consensus parameters
pub const MIN_VALIDATORS: usize = 4;
pub const MAX_VALIDATORS: usize = 100;
pub const CONSENSUS_TIMEOUT_MS: u64 = 30_000; // 30 seconds
pub const MAX_ROUND_NUMBER: u64 = 1000;
pub const PROPOSAL_BATCH_SIZE: usize = 10;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConsensusState {
    Idle,
    PrePrepare,
    Prepare,
    Commit,
    Finalized,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusRound {
    pub round_number: u64,
    pub state: ConsensusState,
    pub leader: PublicKey,
    pub proposals: Vec<ValidationProposal>,
    pub prepare_votes: HashMap<PublicKey, PrepareVote>,
    pub commit_votes: HashMap<PublicKey, CommitVote>,
    pub started_at: DateTime<Utc>,
    pub timeout_at: DateTime<Utc>,
    pub finalized: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationProposal {
    pub id: String,
    pub model_id: String,
    pub validator: PublicKey,
    pub timestamp: DateTime<Utc>,
    pub validation_results: ValidationResults,
    pub hash: [u8; 32],
    pub signature: Signature,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResults {
    pub metrics: ModelMetrics,
    pub status: ValidationStatus,
    pub confidence: f64,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    pub accuracy: f64,
    pub latency_ms: f64,
    pub memory_usage_mb: f64,
    pub throughput: f64,
    pub gpu_utilization: Option<f64>,
    pub inference_time_ms: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ValidationStatus {
    Success,
    Failed,
    Error,
    Timeout,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrepareVote {
    pub round_number: u64,
    pub proposal_hash: [u8; 32],
    pub validator: PublicKey,
    pub timestamp: DateTime<Utc>,
    pub vote: bool,
    pub signature: Signature,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitVote {
    pub round_number: u64,
    pub proposal_hash: [u8; 32],
    pub validator: PublicKey,
    pub timestamp: DateTime<Utc>,
    pub vote: bool,
    pub signature: Signature,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusResult {
    pub round_number: u64,
    pub proposal_hash: [u8; 32],
    pub validation_results: ValidationResults,
    pub prepare_votes: Vec<PrepareVote>,
    pub commit_votes: Vec<CommitVote>,
    pub finalized_at: DateTime<Utc>,
    pub leader_signature: Signature,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorInfo {
    pub public_key: PublicKey,
    pub stake: u64,
    pub last_seen: DateTime<Utc>,
    pub performance: ValidatorPerformance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorPerformance {
    pub total_proposals: u64,
    pub successful_proposals: u64,
    pub total_votes: u64,
    pub vote_participation_rate: f64,
    pub average_response_time_ms: f64,
    pub uptime_percentage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusMetrics {
    pub total_rounds: u64,
    pub successful_rounds: u64,
    pub failed_rounds: u64,
    pub average_round_time_ms: f64,
    pub total_proposals: u64,
    pub total_validations: u64,
    pub validator_count: usize,
}

impl ConsensusRound {
    pub fn new(round_number: u64, leader: PublicKey) -> Self {
        let now = Utc::now();
        Self {
            round_number,
            state: ConsensusState::Idle,
            leader,
            proposals: Vec::new(),
            prepare_votes: HashMap::new(),
            commit_votes: HashMap::new(),
            started_at: now,
            timeout_at: now + chrono::Duration::milliseconds(CONSENSUS_TIMEOUT_MS as i64),
            finalized: false,
        }
    }

    pub fn is_timed_out(&self) -> bool {
        Utc::now() > self.timeout_at
    }

    pub fn has_consensus(&self, validator_count: usize) -> bool {
        let threshold = (validator_count * 2 / 3) + 1;
        
        let prepare_votes = self.prepare_votes.values()
            .filter(|v| v.vote)
            .count();
            
        let commit_votes = self.commit_votes.values()
            .filter(|v| v.vote)
            .count();
            
        prepare_votes >= threshold && commit_votes >= threshold
    }

    pub fn can_finalize(&self, validator_count: usize) -> bool {
        self.state == ConsensusState::Commit && 
        self.has_consensus(validator_count) &&
        !self.is_timed_out()
    }
}

impl ValidationProposal {
    pub fn compute_hash(&self) -> [u8; 32] {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        
        hasher.update(self.id.as_bytes());
        hasher.update(self.model_id.as_bytes());
        hasher.update(self.validator.as_bytes());
        hasher.update(self.timestamp.timestamp().to_be_bytes());
        
        // Hash validation results
        if let Ok(results_bytes) = bincode::serialize(&self.validation_results) {
            hasher.update(&results_bytes);
        }
        
        hasher.finalize().into()
    }

    pub fn verify_signature(&self) -> bool {
        let message = self.compute_hash();
        self.validator.verify(&message, &self.signature).is_ok()
    }
}

impl PrepareVote {
    pub fn verify(&self) -> bool {
        let message = self.proposal_hash;
        self.validator.verify(&message, &self.signature).is_ok()
    }
}

impl CommitVote {
    pub fn verify(&self) -> bool {
        let message = self.proposal_hash;
        self.validator.verify(&message, &self.signature).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::Keypair;
    use rand::rngs::OsRng;

    fn create_test_proposal() -> ValidationProposal {
        let keypair = Keypair::generate(&mut OsRng);
        
        ValidationProposal {
            id: "test_id".to_string(),
            model_id: "model_id".to_string(),
            validator: keypair.public,
            timestamp: Utc::now(),
            validation_results: ValidationResults {
                metrics: ModelMetrics {
                    accuracy: 0.95,
                    latency_ms: 10.0,
                    memory_usage_mb: 1000.0,
                    throughput: 100.0,
                    gpu_utilization: Some(80.0),
                    inference_time_ms: vec![10.0, 11.0, 9.0],
                },
                status: ValidationStatus::Success,
                confidence: 0.9,
                errors: vec![],
            },
            hash: [0u8; 32],
            signature: Signature::new([0u8; 64]),
        }
    }

    #[test]
    fn test_consensus_round() {
        let keypair = Keypair::generate(&mut OsRng);
        let round = ConsensusRound::new(1, keypair.public);
        
        assert_eq!(round.state, ConsensusState::Idle);
        assert!(!round.finalized);
        assert!(!round.is_timed_out());
    }

    #[test]
    fn test_consensus_threshold() {
        let keypair = Keypair::generate(&mut OsRng);
        let mut round = ConsensusRound::new(1, keypair.public);
        
        // Add votes
        for i in 0..7 {
            let voter = Keypair::generate(&mut OsRng);
            let prepare_vote = PrepareVote {
                round_number: 1,
                proposal_hash: [0u8; 32],
                validator: voter.public,
                timestamp: Utc::now(),
                vote: true,
                signature: Signature::new([0u8; 64]),
            };
            let commit_vote = CommitVote {
                round_number: 1,
                proposal_hash: [0u8; 32],
                validator: voter.public,
                timestamp: Utc::now(),
                vote: true,
                signature: Signature::new([0u8; 64]),
            };
            
            round.prepare_votes.insert(voter.public, prepare_vote);
            round.commit_votes.insert(voter.public, commit_vote);
        }
        
        // Test with 10 validators (7 votes > 2/3 * 10)
        assert!(round.has_consensus(10));
    }

    #[test]
    fn test_proposal_verification() {
        let proposal = create_test_proposal();
        let hash = proposal.compute_hash();
        assert_ne!(hash, [0u8; 32]);
    }
}
