use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelType {
    LLM {
        architecture: String,
        parameter_count: u64,
        context_window: u32,
    },
    ComputerVision {
        architecture: String,
        input_resolution: (u32, u32),
        model_family: String,
    },
    TabularPredictor {
        framework: String,
        input_features: u32,
        model_type: String,
    },
    Custom {
        category: String,
        specifications: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRequest {
    pub model_id: String,
    pub model_type: ModelType,
    pub submission_time: DateTime<Utc>,
    pub validator_requirements: ValidatorRequirements,
    pub validation_config: ValidationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorRequirements {
    pub min_stake: u64,
    pub min_reputation: f64,
    pub required_capabilities: Vec<String>,
    pub min_validators: u32,
    pub consensus_percentage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    pub timeout_seconds: u32,
    pub max_retries: u32,
    pub test_cases: Vec<TestCase>,
    pub performance_thresholds: PerformanceThresholds,
    pub safety_checks: SafetyChecks,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCase {
    pub id: String,
    pub inputs: HashMap<String, serde_json::Value>,
    pub expected_outputs: Option<HashMap<String, serde_json::Value>>,
    pub validation_type: TestType,
    pub timeout_ms: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TestType {
    Functional,
    Performance,
    Safety,
    Robustness,
    Bias,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThresholds {
    pub max_latency_ms: u32,
    pub min_throughput: u32,
    pub max_error_rate: f64,
    pub max_memory_usage: u64,
    pub max_gpu_memory: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyChecks {
    pub content_filtering: bool,
    pub bias_detection: bool,
    pub security_scanning: bool,
    pub custom_checks: Vec<CustomCheck>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomCheck {
    pub name: String,
    pub check_type: String,
    pub parameters: HashMap<String, String>,
    pub threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub validation_id: String,
    pub model_id: String,
    pub status: ValidationStatus,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub metrics: ValidationMetrics,
    pub test_results: Vec<TestResult>,
    pub validator_signatures: Vec<ValidatorSignature>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ValidationStatus {
    Pending,
    InProgress,
    Completed,
    Failed(String),
    Timeout,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationMetrics {
    pub performance: PerformanceMetrics,
    pub safety: SafetyMetrics,
    pub resource_usage: ResourceMetrics,
    pub quality_metrics: QualityMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub average_latency: f64,
    pub throughput: f64,
    pub error_rate: f64,
    pub success_rate: f64,
    pub p95_latency: f64,
    pub p99_latency: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyMetrics {
    pub content_safety_score: f64,
    pub bias_metrics: BiasMetrics,
    pub security_score: f64,
    pub identified_risks: Vec<Risk>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiasMetrics {
    pub overall_bias_score: f64,
    pub category_scores: HashMap<String, f64>,
    pub detected_biases: Vec<BiasDetection>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiasDetection {
    pub category: String,
    pub severity: f64,
    pub evidence: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Risk {
    pub risk_type: String,
    pub severity: RiskSeverity,
    pub description: String,
    pub mitigation_suggestion: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RiskSeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub gpu_usage: Option<f64>,
    pub gpu_memory_usage: Option<f64>,
    pub network_usage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub custom_metrics: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    pub test_id: String,
    pub status: TestStatus,
    pub execution_time: f64,
    pub metrics: HashMap<String, f64>,
    pub outputs: Option<HashMap<String, serde_json::Value>>,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TestStatus {
    Passed,
    Failed,
    Error,
    Skipped,
    Timeout,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorSignature {
    pub validator_id: String,
    pub signature: String,
    pub timestamp: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

impl ValidationRequest {
    pub fn new(
        model_id: String,
        model_type: ModelType,
        validator_requirements: ValidatorRequirements,
        validation_config: ValidationConfig,
    ) -> Self {
        Self {
            model_id,
            model_type,
            submission_time: Utc::now(),
            validator_requirements,
            validation_config,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        // Validate validator requirements
        if self.validator_requirements.min_validators < 3 {
            return Err("Minimum 3 validators required".to_string());
        }
        if self.validator_requirements.consensus_percentage < 66.0 
            || self.validator_requirements.consensus_percentage > 100.0 {
            return Err("Consensus percentage must be between 66% and 100%".to_string());
        }

        // Validate configuration
        if self.validation_config.timeout_seconds == 0 {
            return Err("Timeout must be greater than 0".to_string());
        }
        if self.validation_config.test_cases.is_empty() {
            return Err("At least one test case required".to_string());
        }

        Ok(())
    }
}

impl ValidationResult {
    pub fn new(validation_id: String, model_id: String) -> Self {
        Self {
            validation_id,
            model_id,
            status: ValidationStatus::Pending,
            start_time: Utc::now(),
            end_time: Utc::now(),
            metrics: ValidationMetrics {
                performance: PerformanceMetrics {
                    average_latency: 0.0,
                    throughput: 0.0,
                    error_rate: 0.0,
                    success_rate: 0.0,
                    p95_latency: 0.0,
                    p99_latency: 0.0,
                },
                safety: SafetyMetrics {
                    content_safety_score: 0.0,
                    bias_metrics: BiasMetrics {
                        overall_bias_score: 0.0,
                        category_scores: HashMap::new(),
                        detected_biases: Vec::new(),
                    },
                    security_score: 0.0,
                    identified_risks: Vec::new(),
                },
                resource_usage: ResourceMetrics {
                    cpu_usage: 0.0,
                    memory_usage: 0.0,
                    gpu_usage: None,
                    gpu_memory_usage: None,
                    network_usage: 0.0,
                },
                quality_metrics: QualityMetrics {
                    accuracy: 0.0,
                    precision: 0.0,
                    recall: 0.0,
                    f1_score: 0.0,
                    custom_metrics: HashMap::new(),
                },
            },
            test_results: Vec::new(),
            validator_signatures: Vec::new(),
        }
    }

    pub fn is_completed(&self) -> bool {
        matches!(self.status, ValidationStatus::Completed)
    }

    pub fn is_successful(&self) -> bool {
        self.is_completed() && 
        self.metrics.performance.success_rate >= 0.95 &&
        self.metrics.performance.error_rate <= 0.05 &&
        self.metrics.safety.content_safety_score >= 0.8
    }

    pub fn has_consensus(&self, min_validators: u32, consensus_percentage: f64) -> bool {
        let validator_count = self.validator_signatures.len() as u32;
        if validator_count < min_validators {
            return false;
        }

        let consensus_requirement = (validator_count as f64 * consensus_percentage / 100.0).ceil() as usize;
        self.validator_signatures.len() >= consensus_requirement
    }

    pub fn add_test_result(&mut self, result: TestResult) {
        self.test_results.push(result);
        self.update_metrics();
    }

    fn update_metrics(&mut self) {
        let total_tests = self.test_results.len() as f64;
        if total_tests == 0.0 {
            return;
        }

        // Update performance metrics
        let passed_tests = self.test_results.iter()
            .filter(|r| r.status == TestStatus::Passed)
            .count() as f64;

        self.metrics.performance.success_rate = passed_tests / total_tests;
        self.metrics.performance.error_rate = 1.0 - self.metrics.performance.success_rate;

        // Calculate latency statistics
        let mut latencies: Vec<f64> = self.test_results.iter()
            .map(|r| r.execution_time)
            .collect();
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

        self.metrics.performance.average_latency = latencies.iter().sum::<f64>() / total_tests;
        self.metrics.performance.p95_latency = latencies[((total_tests * 0.95) as usize).min(latencies.len() - 1)];
        self.metrics.performance.p99_latency = latencies[((total_tests * 0.99) as usize).min(latencies.len() - 1)];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_validation_request() -> ValidationRequest {
        ValidationRequest::new(
            "model-1".to_string(),
            ModelType::LLM {
                architecture: "transformer".to_string(),
                parameter_count: 1_000_000,
                context_window: 2048,
            },
            ValidatorRequirements {
                min_stake: 1000,
                min_reputation: 0.8,
                required_capabilities: vec!["LLM".to_string()],
                min_validators: 3,
                consensus_percentage: 66.0,
            },
            ValidationConfig {
                timeout_seconds: 300,
                max_retries: 3,
                test_cases: vec![
                    TestCase {
                        id: "test-1".to_string(),
                        inputs: HashMap::new(),
                        expected_outputs: None,
                        validation_type: TestType::Functional,
                        timeout_ms: 1000,
                    },
                ],
                performance_thresholds: PerformanceThresholds {
                    max_latency_ms: 1000,
                    min_throughput: 100,
                    max_error_rate: 0.05,
                    max_memory_usage: 1024 * 1024 * 1024,
                    max_gpu_memory: None,
                },
                safety_checks: SafetyChecks {
                    content_filtering: true,
                    bias_detection: true,
                    security_scanning: true,
                    custom_checks: Vec::new(),
                },
            },
        )
    }

    #[test]
    fn test_validation_request_validation() {
        let request = create_test_validation_request();
        assert!(request.validate().is_ok());

        // Test invalid validator count
        let mut invalid_request = request.clone();
        invalid_request.validator_requirements.min_validators = 2;
        assert!(invalid_request.validate().is_err());

        // Test invalid consensus percentage
        let mut invalid_request = request.clone();
        invalid_request.validator_requirements.consensus_percentage = 50.0;
        assert!(invalid_request.validate().is_err());
    }

    #[test]
    fn test_validation_result_metrics() {
        let mut result = ValidationResult::new(
            "validation-1".to_string(),
            "model-1".to_string(),
        );

        // Add test results
        result.add_test_result(TestResult {
            test_id: "test-1".to_string(),
            status: TestStatus::Passed,
            execution_time: 100.0,
            metrics: HashMap::new(),
            outputs: None,
            error_message: None,
        });

        result.add_test_result(TestResult {
            test_id: "test-2".to_string(),
            status: TestStatus::Failed,
            execution_time: 150.0,
            metrics: HashMap::new(),
            outputs: None,
            error_message: Some("Test failed".to_string()),
        });

        // Verify metrics were updated correctly
        assert_eq!(result.metrics.performance.success_rate, 0.5);
        assert_eq!(result.metrics.performance.error_rate, 0.5);
        assert_eq!(result.metrics.performance.average_latency, 125.0);
        assert_eq!(result.metrics.performance.p95_latency, 150.0);
        assert_eq!(result.metrics.performance.p99_latency, 150.0);
    }

    #[test]
    fn test_validation_consensus() {
        let mut result = ValidationResult::new(
            "validation-1".to_string(),
            "model-1".to_string(),
        );

        // Add validator signatures
        for i in 0..5 {
            result.validator_signatures.push(ValidatorSignature {
                validator_id: format!("validator-{}", i),
                signature: format!("sig-{}", i),
                timestamp: Utc::now(),
                metadata: HashMap::new(),
            });
        }

        // Test consensus requirements
        assert!(result.has_consensus(3, 66.0));
        assert!(!result.has_consensus(6, 66.0));
        assert!(result.has_consensus(5, 100.0));
    }

    #[test]
    fn test_success_criteria() {
        let mut result = ValidationResult::new(
            "validation-1".to_string(),
            "model-1".to_string(),
        );

        // Set successful metrics
        result.status = ValidationStatus::Completed;
        result.metrics.performance.success_rate = 0.98;
        result.metrics.performance.error_rate = 0.02;
        result.metrics.safety.content_safety_score = 0.9;

        assert!(result.is_successful());

        // Test with failing metrics
        result.metrics.performance.success_rate = 0.90;
        assert!(!result.is_successful());
    }

    #[test]
    fn test_model_type_serialization() {
        let model_type = ModelType::LLM {
            architecture: "transformer".to_string(),
            parameter_count: 1_000_000,
            context_window: 2048,
        };

        let serialized = serde_json::to_string(&model_type).unwrap();
        let deserialized: ModelType = serde_json::from_json(&serialized).unwrap();

        assert_eq!(model_type, deserialized);
    }

    #[test]
    fn test_test_result_aggregation() {
        let mut result = ValidationResult::new(
            "validation-1".to_string(),
            "model-1".to_string(),
        );

        // Add multiple test results
        for i in 0..10 {
            result.add_test_result(TestResult {
                test_id: format!("test-{}", i),
                status: if i % 3 == 0 { TestStatus::Failed } else { TestStatus::Passed },
                execution_time: 100.0 + (i as f64 * 10.0),
                metrics: HashMap::new(),
                outputs: None,
                error_message: None,
            });
        }

        // Verify metrics
        assert!(result.metrics.performance.success_rate > 0.6);
        assert!(result.metrics.performance.error_rate < 0.4);
        assert!(result.metrics.performance.p95_latency > result.metrics.performance.average_latency);
        assert!(result.metrics.performance.p99_latency >= result.metrics.performance.p95_latency);
    }

    #[test]
    fn test_resource_metrics() {
        let mut metrics = ResourceMetrics {
            cpu_usage: 75.0,
            memory_usage: 8192.0,
            gpu_usage: Some(85.0),
            gpu_memory_usage: Some(4096.0),
            network_usage: 100.0,
        };

        assert!(metrics.cpu_usage > 0.0 && metrics.cpu_usage <= 100.0);
        assert!(metrics.memory_usage > 0.0);
        assert!(metrics.gpu_usage.unwrap() <= 100.0);
    }
}