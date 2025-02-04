use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{DateTime, Duration, Utc};
use uuid::Uuid;
use thiserror::Error;
use futures::StreamExt;
use log::{debug, error, info, warn};

use crate::models::validation::{
    ValidationRequest, ValidationStatus, ValidationResult,
    ValidationMetrics, DisputeRequest, Dispute,
    ValidationLevel, ValidatorAssignment,
};
use crate::models::node::ValidatorNode;
use crate::services::blockchain_service::BlockchainService;
use crate::services::ai_service::AIService;
use crate::utils::ipfs::IpfsClient;
use crate::utils::metrics::MetricsCollector;

#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("Validation request not found: {0}")]
    NotFound(Uuid),
    #[error("Unauthorized access to validation {0}")]
    Unauthorized(Uuid),
    #[error("Invalid validation state transition from {0} to {1}")]
    InvalidStateTransition(ValidationStatus, ValidationStatus),
    #[error("Blockchain error: {0}")]
    BlockchainError(#[from] crate::services::blockchain_service::BlockchainError),
    #[error("AI service error: {0}")]
    AIServiceError(#[from] crate::services::ai_service::AIServiceError),
    #[error("IPFS error: {0}")]
    IpfsError(String),
    #[error("Database error: {0}")]
    DatabaseError(String),
    #[error("Consensus error: {0}")]
    ConsensusError(String),
}

pub struct ValidationService {
    db: Arc<mongodb::Database>,
    blockchain: Arc<BlockchainService>,
    ai_service: Arc<AIService>,
    ipfs: Arc<IpfsClient>,
    metrics: Arc<MetricsCollector>,
    active_validations: Arc<RwLock<hashmap::HashMap<Uuid, ValidationState>>>,
}

struct ValidationState {
    request: ValidationRequest,
    validators: Vec<ValidatorAssignment>,
    partial_results: Vec<ValidationResult>,
    consensus_deadline: DateTime<Utc>,
}

impl ValidationService {
    pub fn new(
        db: Arc<mongodb::Database>,
        blockchain: Arc<BlockchainService>,
        ai_service: Arc<AIService>,
        ipfs: Arc<IpfsClient>,
        metrics: Arc<MetricsCollector>,
    ) -> Self {
        Self {
            db,
            blockchain,
            ai_service,
            ipfs,
            metrics,
            active_validations: Arc::new(RwLock::new(hashmap::HashMap::new())),
        }
    }

    pub async fn submit_request(
        &self,
        request: ValidationRequest,
    ) -> Result<ValidationRequest, ValidationError> {
        let request_id = request.id;
        info!("Submitting validation request {}", request_id);

        // Store request in database
        self.db
            .collection("validations")
            .insert_one(&request, None)
            .await
            .map_err(|e| ValidationError::DatabaseError(e.to_string()))?;

        // Select validators based on requirements and preferences
        let validators = self.select_validators(&request).await?;

        // Initialize validation state
        let state = ValidationState {
            request: request.clone(),
            validators,
            partial_results: Vec::new(),
            consensus_deadline: Utc::now() + Duration::hours(2),
        };

        // Store state in memory
        self.active_validations
            .write()
            .await
            .insert(request_id, state);

        // Notify validators and start validation process
        self.initiate_validation_process(request_id).await?;

        // Update metrics
        self.metrics.validation_submitted(&request);

        Ok(request)
    }

    async fn select_validators(
        &self,
        request: &ValidationRequest,
    ) -> Result<Vec<ValidatorAssignment>, ValidationError> {
        let required_validators = match request.validation_level {
            ValidationLevel::Basic => 3,
            ValidationLevel::Standard => 5,
            ValidationLevel::Comprehensive => 7,
            ValidationLevel::Enhanced => 9,
        };

        // Query blockchain for available validators
        let available_validators = self.blockchain
            .get_available_validators()
            .await
            .map_err(ValidationError::BlockchainError)?;

        // Filter validators based on preferences and capabilities
        let mut suitable_validators = self.filter_validators(
            available_validators,
            &request.validator_preferences,
        );

        // Sort by reputation and stake
        suitable_validators.sort_by(|a, b| {
            b.reputation_score.partial_cmp(&a.reputation_score)
                .unwrap()
                .then(b.total_stake.cmp(&a.total_stake))
        });

        // Select top validators
        let selected = suitable_validators
            .into_iter()
            .take(required_validators)
            .map(|v| ValidatorAssignment {
                validator: v,
                assigned_at: Utc::now(),
                status: ValidationStatus::Pending,
            })
            .collect();

        Ok(selected)
    }

    async fn initiate_validation_process(
        &self,
        request_id: Uuid,
    ) -> Result<(), ValidationError> {
        let state = self.active_validations
            .read()
            .await
            .get(&request_id)
            .cloned()
            .ok_or(ValidationError::NotFound(request_id))?;

        // Upload model to IPFS for validator access
        let model_cid = self.ipfs
            .add(&state.request.model_data)
            .await
            .map_err(|e| ValidationError::IpfsError(e.to_string()))?;

        // Create validation tasks for each validator
        for assignment in state.validators.iter() {
            let task = self.create_validation_task(
                request_id,
                &assignment.validator,
                &model_cid,
            ).await?;

            // Submit task to validator's queue
            self.blockchain
                .submit_validation_task(&assignment.validator.address, task)
                .await
                .map_err(ValidationError::BlockchainError)?;
        }

        // Update request status
        self.update_validation_status(
            request_id,
            ValidationStatus::Processing,
        ).await?;

        Ok(())
    }

    pub async fn get_validation_status(
        &self,
        request_id: Uuid,
        user_address: &str,
    ) -> Result<ValidationStatus, ValidationError> {
        // Check active validations first
        if let Some(state) = self.active_validations.read().await.get(&request_id) {
            if state.request.submitter != user_address {
                return Err(ValidationError::Unauthorized(request_id));
            }
            return Ok(state.request.status.clone());
        }

        // Check database for completed validations
        let validation = self.db
            .collection("validations")
            .find_one(
                doc! { "id": request_id.to_string() },
                None,
            )
            .await
            .map_err(|e| ValidationError::DatabaseError(e.to_string()))?
            .ok_or(ValidationError::NotFound(request_id))?;

        if validation.submitter != user_address {
            return Err(ValidationError::Unauthorized(request_id));
        }

        Ok(validation.status)
    }

    pub async fn get_validation_result(
        &self,
        request_id: Uuid,
        user_address: &str,
    ) -> Result<Option<ValidationResult>, ValidationError> {
        // Check database for result
        let result = self.db
            .collection("validation_results")
            .find_one(
                doc! { "request_id": request_id.to_string() },
                None,
            )
            .await
            .map_err(|e| ValidationError::DatabaseError(e.to_string()))?;

        // Verify access
        if let Some(ref result) = result {
            let validation = self.db
                .collection("validations")
                .find_one(
                    doc! { "id": request_id.to_string() },
                    None,
                )
                .await
                .map_err(|e| ValidationError::DatabaseError(e.to_string()))?
                .ok_or(ValidationError::NotFound(request_id))?;

            if validation.submitter != user_address {
                return Err(ValidationError::Unauthorized(request_id));
            }
        }

        Ok(result)
    }

    pub async fn get_validation_metrics(
        &self,
        request_id: Uuid,
        user_address: &str,
    ) -> Result<ValidationMetrics, ValidationError> {
        // Get validation result first
        let result = self.get_validation_result(request_id, user_address).await?
            .ok_or(ValidationError::NotFound(request_id))?;

        // Fetch additional metrics from validators
        let validator_metrics = futures::stream::iter(
            result.validator_results.iter()
        )
        .then(|v| self.blockchain.get_validator_metrics(&v.validator_address))
        .collect::<Vec<_>>()
        .await;

        // Aggregate metrics
        let metrics = ValidationMetrics {
            accuracy: result.aggregate_metrics.accuracy,
            performance_score: result.aggregate_metrics.performance_score,
            validation_time: result.validation_time,
            validator_agreement: self.calculate_validator_agreement(&result),
            detailed_metrics: result.detailed_metrics,
            validator_metrics: validator_metrics
                .into_iter()
                .filter_map(Result::ok)
                .collect(),
        };

        Ok(metrics)
    }

    pub async fn submit_dispute(
        &self,
        request_id: Uuid,
        user_address: &str,
        dispute_data: DisputeRequest,
    ) -> Result<Dispute, ValidationError> {
        // Verify validation exists and is completed
        let validation = self.db
            .collection("validations")
            .find_one(
                doc! { "id": request_id.to_string() },
                None,
            )
            .await
            .map_err(|e| ValidationError::DatabaseError(e.to_string()))?
            .ok_or(ValidationError::NotFound(request_id))?;

        if validation.status != ValidationStatus::Completed {
            return Err(ValidationError::InvalidStateTransition(
                validation.status,
                ValidationStatus::Disputed,
            ));
        }

        // Create dispute record
        let dispute = Dispute {
            id: Uuid::new_v4(),
            request_id,
            submitter: user_address.to_string(),
            reason: dispute_data.reason,
            evidence: dispute_data.evidence,
            stake_amount: dispute_data.stake_amount,
            status: ValidationStatus::Pending,
            created_at: Utc::now(),
        };

        // Store dispute
        self.db
            .collection("disputes")
            .insert_one(&dispute, None)
            .await
            .map_err(|e| ValidationError::DatabaseError(e.to_string()))?;

        // Update validation status
        self.update_validation_status(
            request_id,
            ValidationStatus::Disputed,
        ).await?;

        // Notify validators about dispute
        self.blockchain
            .notify_dispute(request_id, &dispute)
            .await
            .map_err(ValidationError::BlockchainError)?;

        Ok(dispute)
    }

    async fn update_validation_status(
        &self,
        request_id: Uuid,
        new_status: ValidationStatus,
    ) -> Result<(), ValidationError> {
        // Update in-memory state if active
        if let Some(mut state) = self.active_validations.write().await.get_mut(&request_id) {
            state.request.status = new_status.clone();
        }

        // Update database
        self.db
            .collection("validations")
            .update_one(
                doc! { "id": request_id.to_string() },
                doc! { "$set": { "status": new_status.to_string() } },
                None,
            )
            .await
            .map_err(|e| ValidationError::DatabaseError(e.to_string()))?;

        Ok(())
    }

    fn calculate_validator_agreement(&self, result: &ValidationResult) -> f64 {
        let total_validators = result.validator_results.len() as f64;
        let agreeing_validators = result.validator_results
            .iter()
            .filter(|v| v.agreed_with_consensus)
            .count() as f64;

        agreeing_validators / total_validators
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;
    use mockall::predicate::*;
    use crate::test_utils::mocks::{
        MockBlockchainService,
        MockAIService,
        MockIpfsClient,
    };

    #[test]
    async fn test_submit_validation_request() {
        // Setup mocks
        let mut blockchain_mock = MockBlockchainService::new();
        blockchain_mock
            .expect_get_available_validators()
            .returning(|| Ok(vec![/* mock validators */]));

        let service = ValidationService::new(
            Arc::new(create_test_db().await),
            Arc::new(blockchain_mock),
            Arc::new(MockAIService::new()),
            Arc::new(MockIpfsClient::new()),
            Arc::new(MetricsCollector::new()),
        );

        // Create test request
        let request = ValidationRequest {
            id: Uuid::new_v4(),
            /* fill other fields */
        };

        // Submit request
        let result = service.submit_request(request.clone()).await;

        // Assert
        assert!(result.is_ok());
        let submitted = result.unwrap();
        assert_eq!(submitted.id, request.id);
        assert_eq!(submitted.status, ValidationStatus::Pending);
    }

    // Add more tests...
}