use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;
use anyhow::{Result, Context, anyhow};
use std::sync::Arc;
use metrics::{counter, gauge, histogram};
use reqwest::{Client, header};
use std::time::{Duration, Instant};
use tokio::time::sleep;
use tracing::{info, warn, error};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LLMConfig {
    pub model: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub stop_sequences: Vec<String>,
    pub api_base: String,
    pub timeout: Duration,
}

impl Default for LLMConfig {
    fn default() -> Self {
        Self {
            model: "gpt-4".to_string(),
            max_tokens: 1000,
            temperature: 0.7,
            top_p: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            stop_sequences: vec![],
            api_base: "https://api.openai.com/v1".to_string(),
            timeout: Duration::from_secs(30),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ValidationPrompt {
    pub model_architecture: String,
    pub parameters: usize,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub training_data_description: String,
    pub intended_use: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LLMValidationResult {
    pub score: f64,
    pub confidence: f64,
    pub reasoning: String,
    pub suggestions: Vec<String>,
    pub potential_issues: Vec<String>,
    pub model_id: String,
    pub timestamp: i64,
}

pub struct LLMValidator {
    config: LLMConfig,
    client: Client,
    rate_limiter: Arc<Semaphore>,
    last_request: std::sync::Mutex<Instant>,
    metrics_prefix: String,
}

impl LLMValidator {
    pub fn new(config: LLMConfig) -> Result<Self> {
        let mut headers = header::HeaderMap::new();
        headers.insert(
            "Content-Type",
            header::HeaderValue::from_static("application/json"),
        );

        let client = Client::builder()
            .timeout(config.timeout)
            .default_headers(headers)
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self {
            config,
            client,
            rate_limiter: Arc::new(Semaphore::new(50)), // Max 50 concurrent requests
            last_request: std::sync::Mutex::new(Instant::now()),
            metrics_prefix: "vetras.llm".to_string(),
        })
    }

    pub async fn validate_model(&self, prompt: ValidationPrompt) -> Result<LLMValidationResult> {
        let _permit = self.rate_limiter.acquire().await?;
        self.enforce_rate_limit().await?;

        let start_time = Instant::now();
        counter!(&format!("{}.request.count", self.metrics_prefix), 1);

        let completion = self.generate_completion(&self.create_prompt(&prompt)).await?;
        let validation_result = self.parse_completion(&completion)?;

        histogram!(
            &format!("{}.request.duration", self.metrics_prefix),
            start_time.elapsed().as_secs_f64(),
        );

        Ok(validation_result)
    }

    async fn enforce_rate_limit(&self) -> Result<()> {
        const MIN_DELAY: Duration = Duration::from_millis(100);
        
        let mut last_request = self.last_request.lock().unwrap();
        let now = Instant::now();
        let elapsed = now.duration_since(*last_request);
        
        if elapsed < MIN_DELAY {
            sleep(MIN_DELAY - elapsed).await;
        }
        
        *last_request = Instant::now();
        Ok(())
    }

    fn create_prompt(&self, prompt: &ValidationPrompt) -> String {
        format!(
            r#"Analyze the following AI model for validation:

Architecture: {}
Parameters: {}
Input Shape: {:?}
Output Shape: {:?}
Training Data: {}
Intended Use: {}

Provide a detailed analysis covering:
1. Model architecture appropriateness
2. Potential issues or concerns
3. Suggestions for improvement
4. Confidence score (0-1)
5. Overall validation score (0-1)

Format the response as JSON with the following structure:
{{
    "score": float,
    "confidence": float,
    "reasoning": string,
    "suggestions": [string],
    "potential_issues": [string]
}}
"#,
            prompt.model_architecture,
            prompt.parameters,
            prompt.input_shape,
            prompt.output_shape,
            prompt.training_data_description,
            prompt.intended_use
        )
    }

    async fn generate_completion(&self, prompt: &str) -> Result<String> {
        #[derive(Serialize)]
        struct Request {
            model: String,
            prompt: String,
            max_tokens: usize,
            temperature: f32,
            top_p: f32,
            frequency_penalty: f32,
            presence_penalty: f32,
            stop: Vec<String>,
        }

        #[derive(Deserialize)]
        struct Response {
            choices: Vec<Choice>,
        }

        #[derive(Deserialize)]
        struct Choice {
            text: String,
        }

        let request = Request {
            model: self.config.model.clone(),
            prompt: prompt.to_string(),
            max_tokens: self.config.max_tokens,
            temperature: self.config.temperature,
            top_p: self.config.top_p,
            frequency_penalty: self.config.frequency_penalty,
            presence_penalty: self.config.presence_penalty,
            stop: self.config.stop_sequences.clone(),
        };

        let response: Response = self.client
            .post(&format!("{}/completions", self.config.api_base))
            .json(&request)
            .send()
            .await
            .context("Failed to send request to LLM API")?
            .json()
            .await
            .context("Failed to parse LLM API response")?;

        response.choices
            .first()
            .map(|c| c.text.clone())
            .ok_or_else(|| anyhow!("No completion returned from LLM API"))
    }

    fn parse_completion(&self, completion: &str) -> Result<LLMValidationResult> {
        #[derive(Deserialize)]
        struct CompletionResponse {
            score: f64,
            confidence: f64,
            reasoning: String,
            suggestions: Vec<String>,
            potential_issues: Vec<String>,
        }

        let response: CompletionResponse = serde_json::from_str(completion)
            .context("Failed to parse LLM completion as JSON")?;

        Ok(LLMValidationResult {
            score: response.score,
            confidence: response.confidence,
            reasoning: response.reasoning,
            suggestions: response.suggestions,
            potential_issues: response.potential_issues,
            model_id: self.config.model.clone(),
            timestamp: chrono::Utc::now().timestamp(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::timeout;
    use wiremock::{MockServer, Mock, ResponseTemplate};
    use wiremock::matchers::{method, path};

    #[tokio::test]
    async fn test_llm_validation() {
        let mock_server = MockServer::start().await;
        
        let mut config = LLMConfig::default();
        config.api_base = mock_server.uri();
        
        Mock::given(method("POST"))
            .and(path("/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{
                    "text": r#"{
                        "score": 0.85,
                        "confidence": 0.92,
                        "reasoning": "The model architecture is appropriate for the intended use case",
                        "suggestions": ["Consider increasing model capacity"],
                        "potential_issues": ["Potential overfitting risk"]
                    }"#
                }]
            })))
            .mount(&mock_server)
            .await;
            
        let validator = LLMValidator::new(config).unwrap();
        
        let prompt = ValidationPrompt {
            model_architecture: "Transformer".to_string(),
            parameters: 1_000_000,
            input_shape: vec![1, 512],
            output_shape: vec![1, 1000],
            training_data_description: "ImageNet dataset".to_string(),
            intended_use: "Image classification".to_string(),
        };
        
        let result = timeout(
            Duration::from_secs(5),
            validator.validate_model(prompt)
        ).await.unwrap().unwrap();
        
        assert!(result.score > 0.0);
        assert!(result.confidence > 0.0);
        assert!(!result.reasoning.is_empty());
        assert!(!result.suggestions.is_empty());
        assert!(!result.potential_issues.is_empty());
    }

    #[tokio::test]
    async fn test_rate_limiting() {
        let config = LLMConfig::default();
        let validator = LLMValidator::new(config).unwrap();
        
        let start = Instant::now();
        for _ in 0..5 {
            validator.enforce_rate_limit().await.unwrap();
        }
        
        assert!(start.elapsed() >= Duration::from_millis(400));
    }
}