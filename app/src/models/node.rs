use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NodeType {
    Validator,
    Observer,
    Storage,
    Compute,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NodeStatus {
    Active,
    Inactive,
    Penalized,
    Maintenance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub node_id: String,
    pub node_type: NodeType,
    pub endpoint: String,
    pub capacity: u32,
    pub status: NodeStatus,
    pub last_seen: DateTime<Utc>,
    pub capabilities: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeStats {
    pub total_validations: u64,
    pub successful_validations: u64,
    pub failed_validations: u64,
    pub uptime_percentage: f64,
    pub average_response_time: f64,
    pub total_stake: u64,
    pub stake_distribution_score: f64,
    pub hardware_metrics: HardwareMetrics,
    pub validation_history: ValidationHistory,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareMetrics {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub disk_usage: f64,
    pub network_bandwidth: f64,
    pub gpu_metrics: Option<GpuMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMetrics {
    pub gpu_usage: f64,
    pub gpu_memory_usage: f64,
    pub temperature: f64,
    pub power_usage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationHistory {
    pub daily_stats: HashMap<String, DailyStats>,
    pub performance_trend: Vec<PerformancePoint>,
    pub recent_validations: Vec<ValidationEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyStats {
    pub date: DateTime<Utc>,
    pub total_validations: u32,
    pub success_rate: f64,
    pub average_response_time: f64,
    pub total_rewards: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformancePoint {
    pub timestamp: DateTime<Utc>,
    pub success_rate: f64,
    pub response_time: f64,
    pub stake_amount: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationEntry {
    pub validation_id: String,
    pub model_id: String,
    pub timestamp: DateTime<Utc>,
    pub status: ValidationStatus,
    pub processing_time: f64,
    pub reward_amount: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ValidationStatus {
    Success,
    Failure,
    Timeout,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeUpdateRequest {
    pub endpoint: Option<String>,
    pub capacity: Option<u32>,
    pub status: Option<NodeStatus>,
    pub capabilities: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeRegistrationInfo {
    pub node_type: NodeType,
    pub endpoint: String,
    pub capacity: u32,
    pub public_key: String,
    pub capabilities: Vec<String>,
    pub hardware_specs: HardwareSpecs,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareSpecs {
    pub cpu_cores: u32,
    pub memory_gb: u32,
    pub disk_gb: u32,
    pub gpu_info: Option<GpuInfo>,
    pub network_bandwidth_mbps: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub model: String,
    pub memory_gb: u32,
    pub cuda_cores: u32,
    pub cuda_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeAggregateStats {
    pub total_nodes: u64,
    pub active_nodes: u64,
    pub total_stake: u64,
    pub average_uptime: f64,
    pub average_response_time: f64,
    pub total_validations_24h: u64,
    pub success_rate_24h: f64,
    pub node_distribution: NodeDistribution,
    pub network_health: NetworkHealth,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeDistribution {
    pub by_type: HashMap<NodeType, u64>,
    pub by_status: HashMap<NodeStatus, u64>,
    pub by_capability: HashMap<String, u64>,
    pub geographical: HashMap<String, u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkHealth {
    pub decentralization_score: f64,
    pub stake_distribution_score: f64,
    pub network_reliability: f64,
    pub validation_efficiency: f64,
}

impl NodeInfo {
    pub fn new(
        node_id: String,
        node_type: NodeType,
        endpoint: String,
        capacity: u32,
        capabilities: Vec<String>,
    ) -> Self {
        Self {
            node_id,
            node_type,
            endpoint,
            capacity,
            status: NodeStatus::Active,
            last_seen: Utc::now(),
            capabilities,
        }
    }

    pub fn update_status(&mut self, status: NodeStatus) {
        self.status = status;
        self.last_seen = Utc::now();
    }

    pub fn is_active(&self) -> bool {
        matches!(self.status, NodeStatus::Active)
    }

    pub fn has_capability(&self, capability: &str) -> bool {
        self.capabilities.contains(&capability.to_string())
    }
}

impl NodeStats {
    pub fn success_rate(&self) -> f64 {
        if self.total_validations == 0 {
            0.0
        } else {
            self.successful_validations as f64 / self.total_validations as f64
        }
    }

    pub fn is_performing_well(&self) -> bool {
        self.success_rate() >= 0.95 && self.uptime_percentage >= 99.0
    }

    pub fn calculate_reliability_score(&self) -> f64 {
        const UPTIME_WEIGHT: f64 = 0.4;
        const SUCCESS_RATE_WEIGHT: f64 = 0.4;
        const RESPONSE_TIME_WEIGHT: f64 = 0.2;

        // Normalize response time (assuming 1000ms is the target)
        let response_time_score = (1000.0 / self.average_response_time).min(1.0);

        // Calculate weighted score
        (self.uptime_percentage / 100.0) * UPTIME_WEIGHT +
        self.success_rate() * SUCCESS_RATE_WEIGHT +
        response_time_score * RESPONSE_TIME_WEIGHT
    }
}

impl ValidationHistory {
    pub fn add_validation(&mut self, entry: ValidationEntry) {
        // Update daily stats
        let date_key = entry.timestamp.date().to_string();
        let daily_stats = self.daily_stats.entry(date_key)
            .or_insert_with(|| DailyStats {
                date: entry.timestamp,
                total_validations: 0,
                success_rate: 0.0,
                average_response_time: 0.0,
                total_rewards: 0,
            });

        daily_stats.total_validations += 1;
        daily_stats.total_rewards += entry.reward_amount;

        // Update performance trend
        self.performance_trend.push(PerformancePoint {
            timestamp: entry.timestamp,
            success_rate: if entry.status == ValidationStatus::Success { 1.0 } else { 0.0 },
            response_time: entry.processing_time,
            stake_amount: 0, // This would be set based on actual stake info
        });

        // Keep only recent validations
        self.recent_validations.push(entry);
        if self.recent_validations.len() > 100 {
            self.recent_validations.remove(0);
        }
    }

    pub fn calculate_success_rate(&self, window_hours: u32) -> f64 {
        let cutoff = Utc::now() - chrono::Duration::hours(window_hours as i64);
        let recent = self.recent_validations.iter()
            .filter(|entry| entry.timestamp >= cutoff)
            .collect::<Vec<_>>();

        if recent.is_empty() {
            return 0.0;
        }

        let successful = recent.iter()
            .filter(|entry| entry.status == ValidationStatus::Success)
            .count();

        successful as f64 / recent.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_info_creation() {
        let node = NodeInfo::new(
            "node-1".to_string(),
            NodeType::Validator,
            "http://localhost:8000".to_string(),
            100,
            vec!["LLM".to_string(), "CV".to_string()],
        );

        assert_eq!(node.node_id, "node-1");
        assert_eq!(node.status, NodeStatus::Active);
        assert!(node.is_active());
        assert!(node.has_capability("LLM"));
        assert!(!node.has_capability("NLP"));
    }

    #[test]
    fn test_node_stats_calculations() {
        let stats = NodeStats {
            total_validations: 1000,
            successful_validations: 950,
            failed_validations: 50,
            uptime_percentage: 99.5,
            average_response_time: 500.0,
            total_stake: 1000000,
            stake_distribution_score: 0.8,
            hardware_metrics: HardwareMetrics {
                cpu_usage: 60.0,
                memory_usage: 70.0,
                disk_usage: 50.0,
                network_bandwidth: 100.0,
                gpu_metrics: None,
            },
            validation_history: ValidationHistory {
                daily_stats: HashMap::new(),
                performance_trend: Vec::new(),
                recent_validations: Vec::new(),
            },
        };

        assert_eq!(stats.success_rate(), 0.95);
        assert!(stats.is_performing_well());
        assert!(stats.calculate_reliability_score() > 0.9);
    }

    #[test]
    fn test_validation_history() {
        let mut history = ValidationHistory {
            daily_stats: HashMap::new(),
            performance_trend: Vec::new(),
            recent_validations: Vec::new(),
        };

        // Add some validation entries
        for i in 0..5 {
            history.add_validation(ValidationEntry {
                validation_id: format!("val-{}", i),
                model_id: "model-1".to_string(),
                timestamp: Utc::now(),
                status: if i % 2 == 0 { ValidationStatus::Success } else { ValidationStatus::Failure },
                processing_time: 100.0,
                reward_amount: 100,
            });
        }

        assert_eq!(history.recent_validations.len(), 5);
        assert!(history.calculate_success_rate(24) == 0.6);
    }

    #[test]
    fn test_node_status_transitions() {
        let mut node = NodeInfo::new(
            "node-1".to_string(),
            NodeType::Validator,
            "http://localhost:8000".to_string(),
            100,
            vec!["LLM".to_string()],
        );

        assert_eq!(node.status, NodeStatus::Active);
        
        node.update_status(NodeStatus::Maintenance);
        assert_eq!(node.status, NodeStatus::Maintenance);
        assert!(!node.is_active());

        node.update_status(NodeStatus::Active);
        assert!(node.is_active());
    }
}
