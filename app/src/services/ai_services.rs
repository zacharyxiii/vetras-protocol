use std::sync::Arc;
use tokio::sync::RwLock;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use async_trait::async_trait;

use crate::models::validation::{
    ModelMetrics, ValidationRequest, ValidationResult,
    ModelType, SafetyMetrics, PerformanceMetrics
};
use crate::utils::error::ApiError;

const MAX_PARALLEL_EVALUATIONS: usize = 4;
const EVALUATION_TIMEOUT: u64 = 300; // 5 minutes
const API_RETRY_ATTEMPTS: u32 = 3;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIModelConfig {
    pub model_type: ModelType,
    pub api_endpoint: String,
    pub api_key: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub timeout_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationMetrics {
    pub latency_ms: u64,
    pub tokens_per_second: f64,
    pub memory_usage_mb: u64,
    pub error_rate: f64,
    pub validation_score: f64,
}

pub struct AIService {
    http_client: Client,
    config: Arc<RwLock<AIModelConfig>>,
    metrics_cache: Arc<RwLock<lru::LruCache<String, EvaluationMetrics>>>,
}

#[async_trait]
pub trait ModelEvaluator: Send + Sync {
    async fn evaluate_model(&self, request: &ValidationRequest) -> Result<ValidationResult, ApiError>;
    async fn get_model_metrics(&self, model_id: &str) -> Result<ModelMetrics, ApiError>;
    async fn validate_safety(&self, model_output: &str) -> Result<SafetyMetrics, ApiError>;
}

impl AIService {
    pub fn new(config: AIModelConfig) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_seconds))
            .pool_max_idle_per_host(10)
            .build()
            .expect("Failed to create HTTP client");

        Self {
            http_client: client,
            config: Arc::new(RwLock::new(config)),
            metrics_cache: Arc::new(RwLock::new(lru::LruCache::new(100))),
        }
    }

    pub async fn update_config(&self, new_config: AIModelConfig) {
        let mut config = self.config.write().await;
        *config = new_config;
    }

    async fn validate_model_compatibility(&self, model_type: &ModelType) -> Result<(), ApiError> {
        let config = self.config.read().await;
        match (&config.model_type, model_type) {
            (ModelType::LLM { .. }, ModelType::LLM { .. }) => Ok(()),
            (ModelType::ComputerVision { .. }, ModelType::ComputerVision { .. }) => Ok(()),
            _ => Err(ApiError::InvalidModelType(format!(
                "Model type {:?} is not compatible with evaluator type {:?}",
                model_type, config.model_type
            ))),
        }
    }

    async fn execute_with_retry<F, T, E>(&self, operation: F) -> Result<T, ApiError>
    where
        F: Fn() -> futures::future::BoxFuture<'_, Result<T, E>>,
        E: std::error::Error + Send + Sync + 'static,
    {
        let mut attempts = 0;
        let mut last_error = None;

        while attempts < API_RETRY_ATTEMPTS {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    attempts += 1;
                    last_error = Some(e);
                    if attempts < API_RETRY_ATTEMPTS {
                        tokio::time::sleep(std::time::Duration::from_secs(2u64.pow(attempts))).await;
                    }
                }
            }
        }

        Err(ApiError::ExternalServiceError(format!(
            "Operation failed after {} attempts: {:?}",
            API_RETRY_ATTEMPTS,
            last_error.unwrap()
        )))
    }

    async fn evaluate_performance(&self, model_id: &str) -> Result<PerformanceMetrics, ApiError> {
        let start_time = std::time::Instant::now();
        
        // Run standard performance test suite
        let metrics = self.run_performance_tests(model_id).await?;
        
        // Calculate performance metrics
        let performance = PerformanceMetrics {
            latency: metrics.latency_ms as u32,
            throughput: (metrics.tokens_per_second * 1000.0) as u32,
            error_rate: metrics.error_rate as f32,
            custom: std::collections::HashMap::new(),
        };

        // Cache the results
        let mut cache = self.metrics_cache.write().await;
        cache.put(model_id.to_string(), metrics);

        Ok(performance)
    }

    async fn run_performance_tests(&self, model_id: &str) -> Result<EvaluationMetrics, ApiError> {
        let config = self.config.read().await;
        
        // Prepare test cases
        let test_cases = self.generate_test_cases(&config).await?;
        
        // Execute tests in parallel with rate limiting
        let mut results = Vec::new();
        let semaphore = tokio::sync::Semaphore::new(MAX_PARALLEL_EVALUATIONS);
        
        for test_case in test_cases {
            let permit = semaphore.acquire().await.map_err(|e| {
                ApiError::InternalError(format!("Failed to acquire semaphore: {}", e))
            })?;
            
            let client = self.http_client.clone();
            let config = config.clone();
            
            let handle = tokio::spawn(async move {
                let _permit = permit;
                Self::execute_single_test(&client, &config, &test_case).await
            });
            
            results.push(handle);
        }

        // Collect and aggregate results
        let mut total_latency = 0u64;
        let mut total_tokens = 0f64;
        let mut total_memory = 0u64;
        let mut error_count = 0u64;
        let mut total_tests = 0u64;

        for handle in results {
            match handle.await {
                Ok(Ok(result)) => {
                    total_latency += result.latency_ms;
                    total_tokens += result.tokens_per_second;
                    total_memory = total_memory.max(result.memory_usage_mb);
                    total_tests += 1;
                }
                Ok(Err(_)) => error_count += 1,
                Err(e) => return Err(ApiError::InternalError(format!(
                    "Test execution failed: {}", e
                ))),
            }
        }

        if total_tests == 0 {
            return Err(ApiError::ValidationFailed(
                "All performance tests failed".to_string()
            ));
        }

        Ok(EvaluationMetrics {
            latency_ms: total_latency / total_tests,
            tokens_per_second: total_tokens / total_tests as f64,
            memory_usage_mb: total_memory,
            error_rate: error_count as f64 / (total_tests + error_count) as f64,
            validation_score: self.calculate_validation_score(
                total_latency / total_tests,
                total_tokens / total_tests as f64,
                error_count as f64 / (total_tests + error_count) as f64,
            ),
        })
    }

    fn calculate_validation_score(
        &self,
        latency: u64,
        tokens_per_second: f64,
        error_rate: f64,
    ) -> f64 {
        // Weight factors for different metrics
        const LATENCY_WEIGHT: f64 = 0.3;
        const THROUGHPUT_WEIGHT: f64 = 0.4;
        const ERROR_WEIGHT: f64 = 0.3;

        // Normalize metrics to 0-1 range
        let latency_score = 1.0 - (latency as f64 / 1000.0).min(1.0);
        let throughput_score = (tokens_per_second / 100.0).min(1.0);
        let error_score = 1.0 - error_rate;

        // Calculate weighted score
        (latency_score * LATENCY_WEIGHT +
         throughput_score * THROUGHPUT_WEIGHT +
         error_score * ERROR_WEIGHT) * 100.0
    }

    async fn generate_test_cases(&self, config: &AIModelConfig) -> Result<Vec<String>, ApiError> {
        // Generate appropriate test cases based on model type
        let test_cases = match config.model_type {
            ModelType::LLM { .. } => {
                vec![
                    "Analyze the sentiment of this text",
                    "Summarize the following paragraph",
                    "Generate a response to this query",
                    "Translate this text to French",
                ]
            }
            ModelType::ComputerVision { .. } => {
                vec![
                    "Detect objects in this image",
                    "Classify the scene type",
                    "Identify the dominant colors",
                    "Detect faces and emotions",
                ]
            }
            _ => return Err(ApiError::InvalidModelType(
                "Unsupported model type for test generation".to_string()
            )),
        };

        Ok(test_cases.into_iter().map(String::from).collect())
    }

    async fn execute_single_test(
        client: &Client,
        config: &AIModelConfig,
        test_case: &str,
    ) -> Result<EvaluationMetrics, ApiError> {
        let start_time = std::time::Instant::now();

        let response = client
            .post(&config.api_endpoint)
            .header("Authorization", format!("Bearer {}", config.api_key))
            .json(&serde_json::json!({
                "prompt": test_case,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
            }))
            .send()
            .await
            .map_err(|e| ApiError::ExternalServiceError(e.to_string()))?;

        let response_data = response
            .json::<serde_json::Value>()
            .await
            .map_err(|e| ApiError::ExternalServiceError(e.to_string()))?;

        let latency = start_time.elapsed().as_millis() as u64;
        let tokens = response_data
            .get("usage")
            .and_then(|u| u.get("total_tokens"))
            .and_then(|t| t.as_u64())
            .unwrap_or(0);

        Ok(EvaluationMetrics {
            latency_ms: latency,
            tokens_per_second: tokens as f64 / (latency as f64 / 1000.0),
            memory_usage_mb: response_data
                .get("usage")
                .and_then(|u| u.get("memory_used"))
                .and_then(|m| m.as_u64())
                .unwrap_or(0),
            error_rate: 0.0,
            validation_score: 0.0,
        })
    }
}

#[async_trait]
impl ModelEvaluator for AIService {
    async fn evaluate_model(&self, request: &ValidationRequest) -> Result<ValidationResult, ApiError> {
        // Validate model compatibility
        self.validate_model_compatibility(&request.model_type).await?;

        // Evaluate performance
        let performance = self.evaluate_performance(&request.model_id).await?;

        // Generate validation result
        Ok(ValidationResult {
            model_id: request.model_id.clone(),
            timestamp: chrono::Utc::now(),
            performance,
            safety: SafetyMetrics {
                safety_score: 95, // This would be calculated based on safety checks
                concerns: vec![],
                recommendations: vec![],
            },
            validation_score: performance.latency as f64,
            status: "completed".to_string(),
        })
    }

    async fn get_model_metrics(&self, model_id: &str) -> Result<ModelMetrics, ApiError> {
        let cache = self.metrics_cache.read().await;
        if let Some(metrics) = cache.get(model_id) {
            return Ok(ModelMetrics {
                latency: metrics.latency_ms,
                throughput: metrics.tokens_per_second as u64,
                error_rate: metrics.error_rate,
                validation_score: metrics.validation_score,
            });
        }

        Err(ApiError::NotFound(format!(
            "No metrics found for model {}", model_id
        )))
    }

    async fn validate_safety(&self, model_output: &str) -> Result<SafetyMetrics, ApiError> {
        // Implement safety validation logic
        let safety_score = self.analyze_safety_concerns(model_output).await?;
        
        Ok(SafetyMetrics {
            safety_score,
            concerns: vec![],
            recommendations: vec![],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;
    use mockall::predicate::*;
    use mockall::mock;

    mock! {
        HttpClient {
            fn post(&self, url: &str) -> RequestBuilder;
            fn send(&self) -> Result<Response, reqwest::Error>;
        }
    }

    #[test]
    async fn test_service_initialization() {
        let config = AIModelConfig {
            model_type: ModelType::LLM {
                architecture: "transformer".to_string(),
                parameter_count: 1_000_000,
                context_window: 2048,
            },
            api_endpoint: "https://api.example.com/v1".to_string(),
            api_key: "test_key".to_string(),
            max_tokens: 1000,
            temperature: 0.7,
            timeout_seconds: 30,
        };

        let service = AIService::new(config.clone());

        // Test compatibility with same model type
        let result = service
            .validate_model_compatibility(&ModelType::LLM {
                architecture: "transformer".to_string(),
                parameter_count: 2_000_000,
                context_window: 4096,
            })
            .await;
        assert!(result.is_ok());

        // Test incompatibility with different model type
        let result = service
            .validate_model_compatibility(&ModelType::ComputerVision {
                architecture: "resnet".to_string(),
                input_resolution: (224, 224),
                model_family: "classification".to_string(),
            })
            .await;
        assert!(result.is_err());
    }

    #[test]
    async fn test_performance_evaluation() {
        let config = AIModelConfig {
            model_type: ModelType::LLM {
                architecture: "transformer".to_string(),
                parameter_count: 1_000_000,
                context_window: 2048,
            },
            api_endpoint: "https://api.example.com/v1".to_string(),
            api_key: "test_key".to_string(),
            max_tokens: 1000,
            temperature: 0.7,
            timeout_seconds: 30,
        };

        let service = AIService::new(config);
        let metrics = service.evaluate_performance("test-model-1").await;
        
        match metrics {
            Ok(perf) => {
                assert!(perf.latency <= 5000); // Expect latency under 5s
                assert!(perf.throughput > 0);
                assert!(perf.error_rate >= 0.0 && perf.error_rate <= 1.0);
            }
            Err(e) => panic!("Performance evaluation failed: {:?}", e),
        }
    }

    #[test]
    async fn test_metrics_caching() {
        let config = AIModelConfig {
            model_type: ModelType::LLM {
                architecture: "transformer".to_string(),
                parameter_count: 1_000_000,
                context_window: 2048,
            },
            api_endpoint: "https://api.example.com/v1".to_string(),
            api_key: "test_key".to_string(),
            max_tokens: 1000,
            temperature: 0.7,
            timeout_seconds: 30,
        };

        let service = AIService::new(config);
        let model_id = "test-model-2";

        // Initially, metrics should not be in cache
        let result = service.get_model_metrics(model_id).await;
        assert!(result.is_err());

        // Perform evaluation to populate cache
        let _ = service.evaluate_performance(model_id).await;

        // Now metrics should be in cache
        let result = service.get_model_metrics(model_id).await;
        assert!(result.is_ok());
    }

    #[test]
    async fn test_validation_score_calculation() {
        let config = AIModelConfig {
            model_type: ModelType::LLM {
                architecture: "transformer".to_string(),
                parameter_count: 1_000_000,
                context_window: 2048,
            },
            api_endpoint: "https://api.example.com/v1".to_string(),
            api_key: "test_key".to_string(),
            max_tokens: 1000,
            temperature: 0.7,
            timeout_seconds: 30,
        };

        let service = AIService::new(config);

        // Test ideal case
        let score = service.calculate_validation_score(100, 95.0, 0.01);
        assert!(score > 90.0); // Should be high score for good performance

        // Test worst case
        let score = service.calculate_validation_score(2000, 10.0, 0.5);
        assert!(score < 50.0); // Should be low score for poor performance

        // Test middle case
        let score = service.calculate_validation_score(500, 50.0, 0.1);
        assert!(score > 50.0 && score < 90.0); // Should be moderate score
    }

    #[test]
    async fn test_retry_mechanism() {
        let config = AIModelConfig {
            model_type: ModelType::LLM {
                architecture: "transformer".to_string(),
                parameter_count: 1_000_000,
                context_window: 2048,
            },
            api_endpoint: "https://api.example.com/v1".to_string(),
            api_key: "test_key".to_string(),
            max_tokens: 1000,
            temperature: 0.7,
            timeout_seconds: 30,
        };

        let service = AIService::new(config);

        // Test operation that succeeds after retries
        let operation = || Box::pin(async {
            static mut ATTEMPTS: u32 = 0;
            unsafe {
                ATTEMPTS += 1;
                if ATTEMPTS < 3 {
                    Err("Temporary failure".into())
                } else {
                    Ok(42)
                }
            }
        });

        let result = service.execute_with_retry(operation).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);

        // Test operation that always fails
        let operation = || Box::pin(async {
            Err("Permanent failure".into())
        });

        let result = service.execute_with_retry(operation).await;
        assert!(result.is_err());
    }

    #[test]
    async fn test_safety_validation() {
        let config = AIModelConfig {
            model_type: ModelType::LLM {
                architecture: "transformer".to_string(),
                parameter_count: 1_000_000,
                context_window: 2048,
            },
            api_endpoint: "https://api.example.com/v1".to_string(),
            api_key: "test_key".to_string(),
            max_tokens: 1000,
            temperature: 0.7,
            timeout_seconds: 30,
        };

        let service = AIService::new(config);

        // Test safe content
        let result = service.validate_safety("This is safe content").await;
        assert!(result.is_ok());
        assert!(result.unwrap().safety_score > 90);

        // Test potentially unsafe content
        let result = service
            .validate_safety("This contains potentially harmful content [...]")
            .await;
        assert!(result.is_ok());
        assert!(result.unwrap().safety_score < 90);
    }

    #[test]
    async fn test_parallel_evaluation() {
        let config = AIModelConfig {
            model_type: ModelType::LLM {
                architecture: "transformer".to_string(),
                parameter_count: 1_000_000,
                context_window: 2048,
            },
            api_endpoint: "https://api.example.com/v1".to_string(),
            api_key: "test_key".to_string(),
            max_tokens: 1000,
            temperature: 0.7,
            timeout_seconds: 30,
        };

        let service = AIService::new(config);

        // Create multiple test cases
        let test_cases = vec!["test1", "test2", "test3", "test4", "test5"];

        // Execute tests in parallel
        let mut handles = Vec::new();
        for test_case in test_cases {
            let service_clone = service.clone();
            let handle = tokio::spawn(async move {
                service_clone.evaluate_performance(test_case).await
            });
            handles.push(handle);
        }

        // Verify all tests complete
        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_ok());
        }
    }
}LM {
                architecture: "transformer".to_string(),
                parameter_count: 1_000_000,
                context_window: 2048,
            },
            api_endpoint: "https://api.example.com/v1".to_string(),
            api_key: "test_key".to_string(),
            max_tokens: 1000,
            temperature: 0.7,
            timeout_seconds: 30,
        };

        let service = AIService::new(config.clone());
        assert!(service.metrics_cache.read().await.cap() == 100);
    }

    #[test]
    async fn test_model_compatibility_validation() {
        let config = AIModelConfig {
            model_type: ModelType::LLM {
                architecture: "transformer".to_string(),
                parameter_count: 1_000_000,
                context_window: 2048,
            },
            api_endpoint: "https://api.example.com/v1".to_string(),
            api_key: "test_key".to_string(),
            max_tokens: 1000,
            temperature: 0.7,
            timeout_seconds: 30,
        };