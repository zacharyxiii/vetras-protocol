use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum TimeFrame {
    Hour,
    Day,
    Week,
    Month,
    Year,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStats {
    pub timestamp: DateTime<Utc>,
    pub validation_stats: ValidationStats,
    pub economic_stats: EconomicStats,
    pub node_stats: NodeStats,
    pub model_stats: ModelStats,
    pub consensus_stats: ConsensusStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationStats {
    pub total_validations: u64,
    pub successful_validations: u64,
    pub failed_validations: u64,
    pub success_rate: f64,
    pub average_validation_time: f64,
    pub validation_distribution: HashMap<String, u64>,
    pub error_distribution: HashMap<String, u64>,
    pub validation_trend: Vec<TimeSeriesPoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EconomicStats {
    pub total_stake: u64,
    pub active_stake: u64,
    pub total_rewards_distributed: u64,
    pub average_reward_per_validation: f64,
    pub stake_distribution: HashMap<String, u64>,
    pub reward_history: Vec<TimeSeriesPoint>,
    pub token_metrics: TokenMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeStats {
    pub total_nodes: u64,
    pub active_nodes: u64,
    pub node_types: HashMap<String, u64>,
    pub geographical_distribution: HashMap<String, u64>,
    pub stake_distribution_score: f64,
    pub network_health: NetworkHealth,
    pub performance_metrics: NetworkPerformance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStats {
    pub total_models: u64,
    pub active_models: u64,
    pub model_types: HashMap<String, u64>,
    pub average_model_size: u64,
    pub model_performance: ModelPerformance,
    pub model_distribution: HashMap<String, ModelTypeStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusStats {
    pub average_consensus_time: f64,
    pub consensus_success_rate: f64,
    pub validator_participation: f64,
    pub consensus_rounds: Vec<ConsensusRound>,
    pub validator_stats: HashMap<String, ValidatorStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesPoint {
    pub timestamp: DateTime<Utc>,
    pub value: f64,
    pub metadata: Option<HashMap<String, String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenMetrics {
    pub total_supply: u64,
    pub circulating_supply: u64,
    pub staked_amount: u64,
    pub staking_apy: f64,
    pub price_data: Option<PriceData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceData {
    pub current_price: f64,
    pub price_change_24h: f64,
    pub volume_24h: f64,
    pub market_cap: f64,
    pub price_history: Vec<TimeSeriesPoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkHealth {
    pub decentralization_score: f64,
    pub reliability_score: f64,
    pub security_score: f64,
    pub health_score: f64,
    pub issues: Vec<NetworkIssue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPerformance {
    pub average_response_time: f64,
    pub requests_per_second: f64,
    pub error_rate: f64,
    pub resource_utilization: ResourceUtilization,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub network_bandwidth: f64,
    pub storage_usage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelTypeStats {
    pub count: u64,
    pub success_rate: f64,
    pub average_validation_time: f64,
    pub resource_usage: ResourceUtilization,
    pub performance_trend: Vec<TimeSeriesPoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformance {
    pub average_accuracy: f64,
    pub average_latency: f64,
    pub success_rate: f64,
    pub performance_by_type: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusRound {
    pub round_id: String,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub validator_count: u32,
    pub consensus_reached: bool,
    pub consensus_time: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorStats {
    pub participation_rate: f64,
    pub success_rate: f64,
    pub average_response_time: f64,
    pub stake_amount: u64,
    pub rewards_earned: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkIssue {
    pub issue_type: String,
    pub severity: IssueSeverity,
    pub description: String,
    pub detected_at: DateTime<Utc>,
    pub resolved_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum IssueSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl NetworkStats {
    pub fn new(
        validation_stats: ValidationStats,
        economic_stats: EconomicStats,
        node_stats: NodeStats,
        model_stats: ModelStats,
        consensus_stats: ConsensusStats,
    ) -> Self {
        Self {
            timestamp: Utc::now(),
            validation_stats,
            economic_stats,
            node_stats,
            model_stats,
            consensus_stats,
        }
    }

    pub fn calculate_network_health(&self) -> f64 {
        let validation_health = self.validation_stats.success_rate;
        let economic_health = self.economic_stats.active_stake as f64 / self.economic_stats.total_stake as f64;
        let node_health = self.node_stats.active_nodes as f64 / self.node_stats.total_nodes as f64;
        let consensus_health = self.consensus_stats.consensus_success_rate;

        // Weighted average of health metrics
        (validation_health * 0.3 +
         economic_health * 0.2 +
         node_health * 0.2 +
         consensus_health * 0.3) * 100.0
    }
}

impl ValidationStats {
    pub fn new() -> Self {
        Self {
            total_validations: 0,
            successful_validations: 0,
            failed_validations: 0,
            success_rate: 0.0,
            average_validation_time: 0.0,
            validation_distribution: HashMap::new(),
            error_distribution: HashMap::new(),
            validation_trend: Vec::new(),
        }
    }

    pub fn add_validation_result(&mut self, success: bool, validation_time: f64, error_type: Option<String>) {
        self.total_validations += 1;
        if success {
            self.successful_validations += 1;
        } else {
            self.failed_validations += 1;
            if let Some(error) = error_type {
                *self.error_distribution.entry(error).or_insert(0) += 1;
            }
        }

        self.success_rate = self.successful_validations as f64 / self.total_validations as f64;
        
        // Update average validation time using running average
        self.average_validation_time = (self.average_validation_time * (self.total_validations - 1) as f64
            + validation_time) / self.total_validations as f64;

        // Add trend point
        self.validation_trend.push(TimeSeriesPoint {
            timestamp: Utc::now(),
            value: self.success_rate,
            metadata: None,
        });
    }
}

impl NetworkHealth {
    pub fn calculate_health_score(&mut self) {
        self.health_score = (
            self.decentralization_score * 0.3 +
            self.reliability_score * 0.3 +
            self.security_score * 0.4
        ) * 100.0;
    }

    pub fn has_critical_issues(&self) -> bool {
        self.issues.iter().any(|issue| issue.severity == IssueSeverity::Critical)
    }
}

impl ConsensusStats {
    pub fn add_consensus_round(&mut self, round: ConsensusRound) {
        // Update average consensus time
        let new_count = self.consensus_rounds.len() as f64 + 1.0;
        self.average_consensus_time = (self.average_consensus_time * (new_count - 1.0) 
            + round.consensus_time) / new_count;

        // Update success rate
        let successful_rounds = self.consensus_rounds.iter()
            .filter(|r| r.consensus_reached)
            .count() as f64 + if round.consensus_reached { 1.0 } else { 0.0 };
        self.consensus_success_rate = successful_rounds / new_count;

        // Add round to history
        self.consensus_rounds.push(round);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_stats() -> NetworkStats {
        NetworkStats::new(
            ValidationStats::new(),
            EconomicStats {
                total_stake: 1_000_000,
                active_stake: 800_000,
                total_rewards_distributed: 50_000,
                average_reward_per_validation: 10.0,
                stake_distribution: HashMap::new(),
                reward_history: Vec::new(),
                token_metrics: TokenMetrics {
                    total_supply: 1_000_000,
                    circulating_supply: 750_000,
                    staked_amount: 500_000,
                    staking_apy: 10.0,
                    price_data: None,
                },
            },
            NodeStats {
                total_nodes: 100,
                active_nodes: 80,
                node_types: HashMap::new(),
                geographical_distribution: HashMap::new(),
                stake_distribution_score: 0.8,
                network_health: NetworkHealth {
                    decentralization_score: 0.8,
                    reliability_score: 0.9,
                    security_score: 0.85,
                    health_score: 0.0,
                    issues: Vec::new(),
                },
                performance_metrics: NetworkPerformance {
                    average_response_time: 100.0,
                    requests_per_second: 1000.0,
                    error_rate: 0.01,
                    resource_utilization: ResourceUtilization {
                        cpu_usage: 60.0,
                        memory_usage: 70.0,
                        network_bandwidth: 500.0,
                        storage_usage: 40.0,
                    },
                },
            },
            ModelStats {
                total_models: 50,
                active_models: 30,
                model_types: HashMap::new(),
                average_model_size: 1_000_000,
                model_performance: ModelPerformance {
                    average_accuracy: 0.95,
                    average_latency: 100.0,
                    success_rate: 0.98,
                    performance_by_type: HashMap::new(),
                },
                model_distribution: HashMap::new(),
            },
            ConsensusStats {
                average_consensus_time: 2.0,
                consensus_success_rate: 0.95,
                validator_participation: 0.9,
                consensus_rounds: Vec::new(),
                validator_stats: HashMap::new(),
            },
        )
    }

    #[test]
    fn test_network_health_calculation() {
        let stats = create_test_stats();
        let health_score = stats.calculate_network_health();
        assert!(health_score > 80.0); // High health score for good metrics
    }

    #[test]
    fn test_validation_stats_updates() {
        let mut stats = ValidationStats::new();
        
        // Add successful validations
        for _ in 0..8 {
            stats.add_validation_result(true, 100.0, None);
        }
        
        // Add failed validations
        for _ in 0..2 {
            stats.add_validation_result(false, 150.0, Some("timeout".to_string()));
        }

        assert_eq!(stats.total_validations, 10);
        assert_eq!(stats.successful_validations, 8);
        assert_eq!(stats.failed_validations, 2);
        assert_eq!(stats.success_rate, 0.8);
        assert!(stats.average_validation_time > 100.0);
        assert_eq!(stats.error_distribution.get("timeout"), Some(&2));
    }

    #[test]
    fn test_consensus_stats_updates() {
        let mut stats = ConsensusStats {
            average_consensus_time: 0.0,
            consensus_success_rate: 0.0,
            validator_participation: 0.9,
            consensus_rounds: Vec::new(),
            validator_stats: HashMap::new(),
        };

        // Add successful round
        stats.add_consensus_round(ConsensusRound {
            round_id: "round-1".to_string(),
            start_time: Utc::now(),
            end_time: Utc::now(),
            validator_count: 10,
            consensus_reached: true,
            consensus_time: 2.0,
        });

        assert_eq!(stats.consensus_success_rate, 1.0);
        assert_eq!(stats.average_consensus_time, 2.0);

        // Add failed round
        stats.add_consensus_round(ConsensusRound {
            round_id: "round-2".to_string(),
            start_time: Utc::now(),
            end_time: Utc::now(),
            validator_count: 10,
            consensus_reached: false,
            consensus_time: 4.0,
        });

        assert_eq!(stats.consensus_success_rate, 0.5);
        assert_eq!(stats.average_consensus_time, 3.0);
    }

    #[test]
    fn test_network_health_critical_issues() {
        let mut health = NetworkHealth {
            decentralization_score: 0.8,
            reliability_score: 0.9,
            security_score: 0.85,
            health_score: 0.0,
            issues: vec![NetworkIssue {
                issue_type: "security".to_string(),
                severity: IssueSeverity::Critical,
                description: "Critical security vulnerability".to_string(),
                detected_at: Utc::now(),
                resolved_at: None,
            }],
        };

        assert!(health.has_critical_issues());
        health.calculate_health_score();
        assert!(health.health_score > 80.0);

        // Clear critical issues
        health.issues.clear();
        assert!(!health.has_critical_issues());
    }

    #[test]
    fn test_model_performance_metrics() {
        let stats = ModelStats {
            total_models: 50,
            active_models: 30,
            model_types: {
                let mut types = HashMap::new();
                types.insert("LLM".to_string(), 20);
                types.insert("CV".to_string(), 30);
                types
            },
            average_model_size: 1_000_000,
            model_performance: ModelPerformance {
                average_accuracy: 0.95,
                average_latency: 100.0,
                success_rate: 0.98,
                performance_by_type: {
                    let mut perf = HashMap::new();
                    perf.insert("LLM".to_string(), 0.96);
                    perf.insert("CV".to_string(), 0.94);
                    perf
                },
            },
            model_distribution: HashMap::new(),
        };

        assert_eq!(stats.total_models, 50);
        assert_eq!(stats.active_models, 30);
        assert_eq!(stats.model_types.get("LLM"), Some(&20));
        assert!(stats.model_performance.performance_by_type.get("LLM").unwrap() > &0.95);
    }

    #[test]
    fn test_resource_utilization() {
        let utilization = ResourceUtilization {
            cpu_usage: 60.0,
            memory_usage: 70.0,
            network_bandwidth: 500.0,
            storage_usage: 40.0,
        };

        assert!(utilization.cpu_usage > 0.0 && utilization.cpu_usage <= 100.0);
        assert!(utilization.memory_usage > 0.0 && utilization.memory_usage <= 100.0);
        assert!(utilization.storage_usage > 0.0 && utilization.storage_usage <= 100.0);
    }

    #[test]
    fn test_economic_stats() {
        let stats = EconomicStats {
            total_stake: 1_000_000,
            active_stake: 800_000,
            total_rewards_distributed: 50_000,
            average_reward_per_validation: 10.0,
            stake_distribution: {
                let mut dist = HashMap::new();
                dist.insert("validator-1".to_string(), 300_000);
                dist.insert("validator-2".to_string(), 500_000);
                dist
            },
            reward_history: vec![
                TimeSeriesPoint {
                    timestamp: Utc::now(),
                    value: 100.0,
                    metadata: None,
                }
            ],
            token_metrics: TokenMetrics {
                total_supply: 1_000_000,
                circulating_supply: 750_000,
                staked_amount: 500_000,
                staking_apy: 10.0,
                price_data: None,
            },
        };

        assert!(stats.active_stake <= stats.total_stake);
        assert!(stats.token_metrics.staked_amount <= stats.token_metrics.total_supply);
        assert!(stats.token_metrics.circulating_supply <= stats.token_metrics.total_supply);
    }

    #[test]
    fn test_time_series_data() {
        let mut trend = Vec::new();
        let now = Utc::now();

        // Add time series points
        for i in 0..10 {
            trend.push(TimeSeriesPoint {
                timestamp: now + chrono::Duration::hours(i),
                value: i as f64 * 1.5,
                metadata: Some({
                    let mut meta = HashMap::new();
                    meta.insert("type".to_string(), "test".to_string());
                    meta
                }),
            });
        }

        assert_eq!(trend.len(), 10);
        assert!(trend.last().unwrap().value > trend.first().unwrap().value);
        assert!(trend.last().unwrap().timestamp > trend.first().unwrap().timestamp);
    }
}