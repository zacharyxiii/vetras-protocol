use actix_web::{web, HttpResponse, Scope};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use std::sync::Arc;
use validator::Validate;

use crate::services::validation_service::ValidationService;
use crate::services::blockchain_service::BlockchainService;
use crate::services::ai_service::AIService;
use crate::models::validation::{
    ValidationRequest, ValidationStatus, ValidationResult,
    ValidationLevel, ValidatorPreferences,
};
use crate::utils::error::{ApiError, ApiResult};
use crate::utils::auth::AuthenticatedUser;

#[derive(Debug, Deserialize, Validate)]
pub struct SubmitValidationRequest {
    #[validate(length(min = 1, max = 256))]
    pub model_name: String,
    #[validate]
    pub model_data: ModelData,
    pub validation_level: ValidationLevel,
    #[validate]
    pub validator_preferences: Option<ValidatorPreferences>,
    #[validate(range(min = 0))]
    pub stake_amount: Option<u64>,
}

#[derive(Debug, Deserialize, Validate)]
pub struct ModelData {
    #[validate(length(min = 1))]
    pub framework: String,
    #[validate(length(min = 1))]
    pub version: String,
    pub weights_uri: Option<String>,
    pub architecture: serde_json::Value,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct ValidationResponse {
    pub request_id: Uuid,
    pub status: ValidationStatus,
    pub estimated_completion_time: Option<DateTime<Utc>>,
}

pub fn validation_routes(
    validation_service: Arc<ValidationService>,
    blockchain_service: Arc<BlockchainService>,
    ai_service: Arc<AIService>,
) -> Scope {
    web::scope("/validations")
        .app_data(web::JsonConfig::default().limit(50 * 1024 * 1024)) // 50MB limit
        .service(web::resource("")
            .route(web::post().to(submit_validation))
            .route(web::get().to(list_validations)))
        .service(web::resource("/{request_id}")
            .route(web::get().to(get_validation))
            .route(web::delete().to(cancel_validation)))
        .service(web::resource("/{request_id}/status")
            .route(web::get().to(get_validation_status)))
        .service(web::resource("/{request_id}/result")
            .route(web::get().to(get_validation_result)))
        .service(web::resource("/{request_id}/metrics")
            .route(web::get().to(get_validation_metrics)))
        .service(web::resource("/{request_id}/dispute")
            .route(web::post().to(submit_dispute)))
}

async fn submit_validation(
    req: web::Json<SubmitValidationRequest>,
    user: AuthenticatedUser,
    validation_service: web::Data<Arc<ValidationService>>,
    blockchain_service: web::Data<Arc<BlockchainService>>,
    ai_service: web::Data<Arc<AIService>>,
) -> ApiResult<HttpResponse> {
    // Validate request
    req.validate().map_err(|e| ApiError::ValidationError(e))?;

    // Verify stake amount and lock tokens if provided
    if let Some(stake_amount) = req.stake_amount {
        blockchain_service
            .lock_stake(&user.address, stake_amount)
            .await
            .map_err(|e| ApiError::BlockchainError(e))?;
    }

    // Initialize AI model validation
    let model = ai_service
        .initialize_model(&req.model_data)
        .await
        .map_err(|e| ApiError::AIServiceError(e))?;

    // Create validation request
    let request = ValidationRequest {
        id: Uuid::new_v4(),
        model_id: model.id,
        submitter: user.address,
        validation_level: req.validation_level.clone(),
        validator_preferences: req.validator_preferences.clone(),
        stake_amount: req.stake_amount,
        status: ValidationStatus::Pending,
        created_at: Utc::now(),
    };

    // Submit for validation
    let response = validation_service
        .submit_request(request)
        .await
        .map_err(|e| ApiError::ValidationError(e.to_string()))?;

    Ok(HttpResponse::Accepted().json(ValidationResponse {
        request_id: response.id,
        status: response.status,
        estimated_completion_time: response.estimated_completion_time,
    }))
}

async fn list_validations(
    user: AuthenticatedUser,
    validation_service: web::Data<Arc<ValidationService>>,
    query: web::Query<ListValidationsQuery>,
) -> ApiResult<HttpResponse> {
    let validations = validation_service
        .list_user_validations(&user.address, query.into_inner())
        .await
        .map_err(|e| ApiError::ValidationError(e.to_string()))?;

    Ok(HttpResponse::Ok().json(validations))
}

async fn get_validation(
    request_id: web::Path<Uuid>,
    user: AuthenticatedUser,
    validation_service: web::Data<Arc<ValidationService>>,
) -> ApiResult<HttpResponse> {
    let validation = validation_service
        .get_validation(*request_id, &user.address)
        .await
        .map_err(|e| ApiError::ValidationError(e.to_string()))?;

    Ok(HttpResponse::Ok().json(validation))
}

async fn get_validation_status(
    request_id: web::Path<Uuid>,
    user: AuthenticatedUser,
    validation_service: web::Data<Arc<ValidationService>>,
) -> ApiResult<HttpResponse> {
    let status = validation_service
        .get_validation_status(*request_id, &user.address)
        .await
        .map_err(|e| ApiError::ValidationError(e.to_string()))?;

    Ok(HttpResponse::Ok().json(status))
}

async fn get_validation_result(
    request_id: web::Path<Uuid>,
    user: AuthenticatedUser,
    validation_service: web::Data<Arc<ValidationService>>,
) -> ApiResult<HttpResponse> {
    let result = validation_service
        .get_validation_result(*request_id, &user.address)
        .await
        .map_err(|e| ApiError::ValidationError(e.to_string()))?;

    match result {
        Some(result) => Ok(HttpResponse::Ok().json(result)),
        None => Err(ApiError::NotFound("Validation result not found".into())),
    }
}

async fn get_validation_metrics(
    request_id: web::Path<Uuid>,
    user: AuthenticatedUser,
    validation_service: web::Data<Arc<ValidationService>>,
) -> ApiResult<HttpResponse> {
    let metrics = validation_service
        .get_validation_metrics(*request_id, &user.address)
        .await
        .map_err(|e| ApiError::ValidationError(e.to_string()))?;

    Ok(HttpResponse::Ok().json(metrics))
}

async fn cancel_validation(
    request_id: web::Path<Uuid>,
    user: AuthenticatedUser,
    validation_service: web::Data<Arc<ValidationService>>,
    blockchain_service: web::Data<Arc<BlockchainService>>,
) -> ApiResult<HttpResponse> {
    // Cancel validation process
    validation_service
        .cancel_validation(*request_id, &user.address)
        .await
        .map_err(|e| ApiError::ValidationError(e.to_string()))?;

    // Return staked tokens if any
    blockchain_service
        .return_stake(*request_id, &user.address)
        .await
        .map_err(|e| ApiError::BlockchainError(e))?;

    Ok(HttpResponse::Ok().json(json!({
        "message": "Validation cancelled successfully"
    })))
}

async fn submit_dispute(
    request_id: web::Path<Uuid>,
    dispute_data: web::Json<DisputeRequest>,
    user: AuthenticatedUser,
    validation_service: web::Data<Arc<ValidationService>>,
    blockchain_service: web::Data<Arc<BlockchainService>>,
) -> ApiResult<HttpResponse> {
    // Validate dispute request
    dispute_data.validate().map_err(|e| ApiError::ValidationError(e))?;

    // Lock dispute stake
    blockchain_service
        .lock_dispute_stake(&user.address, dispute_data.stake_amount)
        .await
        .map_err(|e| ApiError::BlockchainError(e))?;

    // Submit dispute
    let dispute = validation_service
        .submit_dispute(*request_id, &user.address, dispute_data.into_inner())
        .await
        .map_err(|e| ApiError::ValidationError(e.to_string()))?;

    Ok(HttpResponse::Accepted().json(dispute))
}

#[derive(Debug, Deserialize)]
pub struct ListValidationsQuery {
    pub status: Option<ValidationStatus>,
    pub from_date: Option<DateTime<Utc>>,
    pub to_date: Option<DateTime<Utc>>,
    pub page: Option<u32>,
    pub per_page: Option<u32>,
}

#[derive(Debug, Deserialize, Validate)]
pub struct DisputeRequest {
    #[validate(length(min = 1, max = 1024))]
    pub reason: String,
    pub evidence: Option<serde_json::Value>,
    #[validate(range(min = 1))]
    pub stake_amount: u64,
}