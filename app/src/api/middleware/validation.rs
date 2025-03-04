use actix_web::{
    dev::{forward_ready, Service, ServiceRequest, ServiceResponse, Transform},
    error::Error,
    http::StatusCode,
    web::Json,
    HttpMessage, HttpResponse,
};
use futures::future::{ready, Ready};
use serde::Deserialize;
use std::{
    future::Future,
    pin::Pin,
    rc::Rc,
    task::{Context, Poll},
};

use crate::utils::error::{ApiError, ApiResult};

// Maximum size limits for various request components
const MAX_JSON_PAYLOAD: usize = 10 * 1024 * 1024; // 10MB
const MAX_MODEL_SIZE: usize = 100 * 1024 * 1024; // 100MB
const MAX_METADATA_SIZE: usize = 1024 * 1024; // 1MB

#[derive(Debug, Deserialize)]
pub struct ValidationConfig {
    pub max_payload_size: Option<usize>,
    pub allowed_model_types: Option<Vec<String>>,
    pub required_fields: Option<Vec<String>>,
}

pub struct RequestValidation {
    config: ValidationConfig,
}

impl RequestValidation {
    pub fn new(config: ValidationConfig) -> Self {
        Self { config }
    }

    pub fn default() -> Self {
        Self {
            config: ValidationConfig {
                max_payload_size: Some(MAX_JSON_PAYLOAD),
                allowed_model_types: None,
                required_fields: None,
            },
        }
    }
}

impl<S, B> Transform<S, ServiceRequest> for RequestValidation
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error> + 'static,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type InitError = ();
    type Transform = RequestValidationMiddleware<S>;
    type Future = Ready<Result<Self::Transform, Self::InitError>>;

    fn new_transform(&self, service: S) -> Self::Future {
        ready(Ok(RequestValidationMiddleware {
            service: Rc::new(service),
            config: self.config.clone(),
        }))
    }
}

pub struct RequestValidationMiddleware<S> {
    service: Rc<S>,
    config: ValidationConfig,
}

impl<S, B> Service<ServiceRequest> for RequestValidationMiddleware<S>
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error> + 'static,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>>>>;

    forward_ready!(service);

    fn call(&self, mut req: ServiceRequest) -> Self::Future {
        let svc = self.service.clone();
        let config = self.config.clone();

        Box::pin(async move {
            // Validate content type
            if let Some(content_type) = req.headers().get("content-type") {
                let content_type = content_type.to_str().map_err(|_| {
                    ApiError::ValidationError("Invalid content-type header".into())
                })?;

                if !content_type.starts_with("application/json") {
                    return Err(ApiError::ValidationError(
                        "Content-type must be application/json".into(),
                    )
                    .into());
                }
            }

            // Validate payload size
            if let Some(max_size) = config.max_payload_size {
                if let Some(content_length) = req.headers().get("content-length") {
                    let size = content_length
                        .to_str()
                        .map_err(|_| {
                            ApiError::ValidationError("Invalid content-length header".into())
                        })?
                        .parse::<usize>()
                        .map_err(|_| {
                            ApiError::ValidationError("Invalid content-length value".into())
                        })?;

                    if size > max_size {
                        return Err(ApiError::ValidationError("Payload too large".into()).into());
                    }
                }
            }

            // Validate request body if it's a model submission
            if req.path().contains("/models") && req.method() == "POST" {
                validate_model_submission(&mut req, &config).await?;
            }

            // Validate request body if it's a validation result
            if req.path().contains("/validations") && req.method() == "POST" {
                validate_validation_result(&mut req).await?;
            }

            let res = svc.call(req).await?;
            Ok(res)
        })
    }
}

async fn validate_model_submission(
    req: &mut ServiceRequest,
    config: &ValidationConfig,
) -> ApiResult<()> {
    #[derive(Debug, Deserialize)]
    struct ModelSubmission {
        model_type: String,
        metadata: String,
        #[serde(default)]
        parameters: Option<serde_json::Value>,
    }

    let body = req.extract::<Json<ModelSubmission>>().await.map_err(|_| {
        ApiError::ValidationError("Invalid model submission format".into())
    })?;

    // Validate model type
    if let Some(ref allowed_types) = config.allowed_model_types {
        if !allowed_types.contains(&body.model_type) {
            return Err(ApiError::ValidationError(format!(
                "Unsupported model type: {}",
                body.model_type
            )));
        }
    }

    // Validate metadata size
    if body.metadata.len() > MAX_METADATA_SIZE {
        return Err(ApiError::ValidationError("Metadata too large".into()));
    }

    // Validate required fields
    if let Some(ref required) = config.required_fields {
        let metadata: serde_json::Value = serde_json::from_str(&body.metadata).map_err(|_| {
            ApiError::ValidationError("Invalid metadata format".into())
        })?;

        for field in required {
            if !metadata.get(field).is_some() {
                return Err(ApiError::ValidationError(format!(
                    "Missing required field: {}",
                    field
                )));
            }
        }
    }

    Ok(())
}

async fn validate_validation_result(req: &mut ServiceRequest) -> ApiResult<()> {
    #[derive(Debug, Deserialize)]
    struct ValidationResult {
        metrics: ValidationMetrics,
        signature: String,
    }

    #[derive(Debug, Deserialize)]
    struct ValidationMetrics {
        performance: PerformanceMetrics,
        safety: SafetyMetrics,
        resources: ResourceMetrics,
    }

    #[derive(Debug, Deserialize)]
    struct PerformanceMetrics {
        latency: f64,
        throughput: f64,
        error_rate: f64,
    }

    #[derive(Debug, Deserialize)]
    struct SafetyMetrics {
        safety_score: u8,
        concerns: Vec<String>,
    }

    #[derive(Debug, Deserialize)]
    struct ResourceMetrics {
        memory_usage: u64,
        cpu_utilization: f64,
    }

    let body = req.extract::<Json<ValidationResult>>().await.map_err(|_| {
        ApiError::ValidationError("Invalid validation result format".into())
    })?;

    // Validate metrics ranges
    if body.metrics.performance.error_rate < 0.0 || body.metrics.performance.error_rate > 1.0 {
        return Err(ApiError::ValidationError(
            "Error rate must be between 0 and 1".into(),
        ));
    }

    if body.metrics.safety.safety_score > 100 {
        return Err(ApiError::ValidationError(
            "Safety score must be between 0 and 100".into(),
        ));
    }

    if body.metrics.resources.cpu_utilization > 100.0 {
        return Err(ApiError::ValidationError(
            "CPU utilization must be between 0 and 100".into(),
        ));
    }

    // Validate signature format
    if !body.signature.starts_with("0x") || body.signature.len() != 132 {
        return Err(ApiError::ValidationError("Invalid signature format".into()));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::test::{self, TestRequest};
    use actix_web::{web, App, HttpResponse};

    async fn test_handler() -> HttpResponse {
        HttpResponse::Ok().finish()
    }

    #[actix_web::test]
    async fn test_content_type_validation() {
        let app = test::init_service(
            App::new()
                .wrap(RequestValidation::default())
                .route("/test", web::post().to(test_handler)),
        )
        .await;

        let req = TestRequest::post()
            .uri("/test")
            .header("content-type", "text/plain")
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[actix_web::test]
    async fn test_payload_size_validation() {
        let app = test::init_service(
            App::new()
                .wrap(RequestValidation::new(ValidationConfig {
                    max_payload_size: Some(100),
                    allowed_model_types: None,
                    required_fields: None,
                }))
                .route("/test", web::post().to(test_handler)),
        )
        .await;

        let req = TestRequest::post()
            .uri("/test")
            .header("content-type", "application/json")
            .header("content-length", "1000")
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[actix_web::test]
    async fn test_model_type_validation() {
        let app = test::init_service(
            App::new()
                .wrap(RequestValidation::new(ValidationConfig {
                    max_payload_size: None,
                    allowed_model_types: Some(vec!["llm".to_string()]),
                    required_fields: None,
                }))
                .route("/models", web::post().to(test_handler)),
        )
        .await;

        let payload = r#"{
            "model_type": "unsupported",
            "metadata": "{}",
            "parameters": null
        }"#;

        let req = TestRequest::post()
            .uri("/models")
            .header("content-type", "application/json")
            .set_payload(payload)
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }
}
