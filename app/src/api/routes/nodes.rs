use actix_web::{web, HttpResponse, Scope};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::services::blockchain_service::BlockchainService;
use crate::utils::error::{ApiError, ApiResult};
use crate::utils::cache::{Cache, CacheKey};
use crate::models::node::{NodeInfo, NodeStatus, NodeStats, NodeType};

#[derive(Debug, Deserialize)]
pub struct NodeQuery {
    pub status: Option<NodeStatus>,
    pub node_type: Option<NodeType>,
    pub from_date: Option<DateTime<Utc>>,
    pub to_date: Option<DateTime<Utc>>,
}

#[derive(Debug, Deserialize)]
pub struct NodeRegistration {
    pub node_type: NodeType,
    pub endpoint: String,
    pub capacity: u32,
    pub public_key: String,
    pub capabilities: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct NodeList {
    pub nodes: Vec<NodeInfo>,
    pub total: u64,
    pub active: u64,
}

#[derive(Debug, Serialize)]
pub struct NodePerformance {
    pub uptime_percentage: f64,
    pub average_response_time: f64,
    pub successful_validations: u64,
    pub failed_validations: u64,
    pub total_stake: u64,
    pub reputation_score: f64,
}

pub fn node_routes(
    blockchain_service: Arc<BlockchainService>,
    cache: Arc<Cache>,
) -> Scope {
    web::scope("/nodes")
        .service(
            web::resource("")
                .route(web::get().to(list_nodes))
                .route(web::post().to(register_node))
        )
        .service(
            web::resource("/{node_id}")
                .route(web::get().to(get_node_info))
                .route(web::put().to(update_node))
                .route(web::delete().to(deregister_node))
        )
        .service(
            web::resource("/{node_id}/performance")
                .route(web::get().to(get_node_performance))
        )
        .service(
            web::resource("/{node_id}/stats")
                .route(web::get().to(get_node_stats))
        )
        .service(
            web::resource("/stats/aggregate")
                .route(web::get().to(get_aggregate_stats))
        )
}

async fn list_nodes(
    query: web::Query<NodeQuery>,
    blockchain_service: web::Data<Arc<BlockchainService>>,
    cache: web::Data<Arc<Cache>>,
) -> ApiResult<HttpResponse> {
    let cache_key = CacheKey::new("node_list", &query);
    if let Some(cached) = cache.get(&cache_key).await {
        return Ok(HttpResponse::Ok().json(cached));
    }

    let nodes = blockchain_service
        .get_nodes(
            query.status.as_ref(),
            query.node_type.as_ref(),
            query.from_date,
            query.to_date,
        )
        .await
        .map_err(ApiError::BlockchainError)?;

    let active_count = nodes.iter().filter(|n| n.status == NodeStatus::Active).count() as u64;

    let response = NodeList {
        nodes,
        total: nodes.len() as u64,
        active: active_count,
    };

    cache.set(&cache_key, &response, chrono::Duration::minutes(1)).await;
    Ok(HttpResponse::Ok().json(response))
}

async fn register_node(
    registration: web::Json<NodeRegistration>,
    blockchain_service: web::Data<Arc<BlockchainService>>,
) -> ApiResult<HttpResponse> {
    // Validate node endpoint is accessible
    if !validate_node_endpoint(&registration.endpoint).await {
        return Err(ApiError::ValidationError("Node endpoint is not accessible".into()));
    }

    // Verify public key format and signature
    if !verify_node_key(&registration.public_key) {
        return Err(ApiError::ValidationError("Invalid public key format".into()));
    }

    let node_info = blockchain_service
        .register_node(
            registration.node_type.clone(),
            &registration.endpoint,
            registration.capacity,
            &registration.public_key,
            registration.capabilities.clone(),
        )
        .await
        .map_err(ApiError::BlockchainError)?;

    Ok(HttpResponse::Created().json(node_info))
}

async fn get_node_info(
    node_id: web::Path<String>,
    blockchain_service: web::Data<Arc<BlockchainService>>,
    cache: web::Data<Arc<Cache>>,
) -> ApiResult<HttpResponse> {
    let cache_key = CacheKey::new("node_info", node_id.as_str());
    if let Some(cached) = cache.get(&cache_key).await {
        return Ok(HttpResponse::Ok().json(cached));
    }

    let node_info = blockchain_service
        .get_node_info(&node_id)
        .await
        .map_err(ApiError::BlockchainError)?;

    cache.set(&cache_key, &node_info, chrono::Duration::minutes(1)).await;
    Ok(HttpResponse::Ok().json(node_info))
}

async fn update_node(
    node_id: web::Path<String>,
    update: web::Json<NodeInfo>,
    blockchain_service: web::Data<Arc<BlockchainService>>,
) -> ApiResult<HttpResponse> {
    let updated_info = blockchain_service
        .update_node(&node_id, update.into_inner())
        .await
        .map_err(ApiError::BlockchainError)?;

    Ok(HttpResponse::Ok().json(updated_info))
}

async fn deregister_node(
    node_id: web::Path<String>,
    blockchain_service: web::Data<Arc<BlockchainService>>,
) -> ApiResult<HttpResponse> {
    blockchain_service
        .deregister_node(&node_id)
        .await
        .map_err(ApiError::BlockchainError)?;

    Ok(HttpResponse::NoContent().finish())
}

async fn get_node_performance(
    node_id: web::Path<String>,
    blockchain_service: web::Data<Arc<BlockchainService>>,
    cache: web::Data<Arc<Cache>>,
) -> ApiResult<HttpResponse> {
    let cache_key = CacheKey::new("node_performance", node_id.as_str());
    if let Some(cached) = cache.get(&cache_key).await {
        return Ok(HttpResponse::Ok().json(cached));
    }

    let stats = blockchain_service
        .get_node_stats(&node_id)
        .await
        .map_err(ApiError::BlockchainError)?;

    let performance = calculate_node_performance(&stats);
    
    cache.set(&cache_key, &performance, chrono::Duration::minutes(5)).await;
    Ok(HttpResponse::Ok().json(performance))
}

async fn get_node_stats(
    node_id: web::Path<String>,
    blockchain_service: web::Data<Arc<BlockchainService>>,
    cache: web::Data<Arc<Cache>>,
) -> ApiResult<HttpResponse> {
    let cache_key = CacheKey::new("node_stats", node_id.as_str());
    if let Some(cached) = cache.get(&cache_key).await {
        return Ok(HttpResponse::Ok().json(cached));
    }

    let stats = blockchain_service
        .get_node_stats(&node_id)
        .await
        .map_err(ApiError::BlockchainError)?;

    cache.set(&cache_key, &stats, chrono::Duration::minutes(5)).await;
    Ok(HttpResponse::Ok().json(stats))
}

async fn get_aggregate_stats(
    blockchain_service: web::Data<Arc<BlockchainService>>,
    cache: web::Data<Arc<Cache>>,
) -> ApiResult<HttpResponse> {
    let cache_key = CacheKey::new("aggregate_stats", &());
    if let Some(cached) = cache.get(&cache_key).await {
        return Ok(HttpResponse::Ok().json(cached));
    }

    let stats = blockchain_service
        .get_aggregate_node_stats()
        .await
        .map_err(ApiError::BlockchainError)?;

    cache.set(&cache_key, &stats, chrono::Duration::minutes(5)).await;
    Ok(HttpResponse::Ok().json(stats))
}

fn calculate_node_performance(stats: &NodeStats) -> NodePerformance {
    let total_validations = stats.successful_validations + stats.failed_validations;
    let success_rate = if total_validations > 0 {
        stats.successful_validations as f64 / total_validations as f64
    } else {
        0.0
    };

    NodePerformance {
        uptime_percentage: stats.uptime_percentage,
        average_response_time: stats.average_response_time,
        successful_validations: stats.successful_validations,
        failed_validations: stats.failed_validations,
        total_stake: stats.total_stake,
        reputation_score: calculate_reputation_score(success_rate, stats.uptime_percentage),
    }
}

fn calculate_reputation_score(success_rate: f64, uptime_percentage: f64) -> f64 {
    // Weight factors for different metrics
    const SUCCESS_WEIGHT: f64 = 0.6;
    const UPTIME_WEIGHT: f64 = 0.4;

    // Calculate weighted score
    (success_rate * SUCCESS_WEIGHT + uptime_percentage * UPTIME_WEIGHT) * 100.0
}

async fn validate_node_endpoint(endpoint: &str) -> bool {
    // Attempt to connect and verify node is responsive
    match reqwest::Client::new()
        .get(endpoint)
        .timeout(std::time::Duration::from_secs(5))
        .send()
        .await
    {
        Ok(response) => response.status().is_success(),
        Err(_) => false,
    }
}

fn verify_node_key(public_key: &str) -> bool {
    // Verify the public key format and signature
    // This is a simplified check - in production, you'd want more robust verification
    public_key.starts_with("0x") && public_key.len() == 66
}

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::{test, App};
    use crate::test_utils::mock_blockchain_service::MockBlockchainService;
    use crate::test_utils::mock_cache::MockCache;

    #[actix_web::test]
    async fn test_list_nodes() {
        // Create test app
        let app = test::init_service(
            App::new()
                .app_data(web::Data::new(Arc::new(MockBlockchainService::new())))
                .app_data(web::Data::new(Arc::new(MockCache::new())))
                .service(node_routes(
                    Arc::new(MockBlockchainService::new()),
                    Arc::new(MockCache::new()),
                ))
        ).await;

        // Create test request
        let req = test::TestRequest::get()
            .uri("/nodes?status=active")
            .to_request();

        // Send request and verify response
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());

        let nodes: NodeList = test::read_body_json(resp).await;
        assert!(nodes.total >= nodes.active);
    }

    #[actix_web::test]
    async fn test_register_node() {
        let app = test::init_service(
            App::new()
                .app_data(web::Data::new(Arc::new(MockBlockchainService::new())))
                .service(node_routes(
                    Arc::new(MockBlockchainService::new()),
                    Arc::new(MockCache::new()),
                ))
        ).await;

        let registration = NodeRegistration {
            node_type: NodeType::Validator,
            endpoint: "https://validator1.example.com".to_string(),
            capacity: 100,
            public_key: "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef".to_string(),
            capabilities: vec!["LLM".to_string(), "CV".to_string()],
        };

        let req = test::TestRequest::post()
            .uri("/nodes")
            .set_json(&registration)
            .to_request();

        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), 201);
    }

    #[actix_web::test]
    async fn test_get_node_info() {
        let app = test::init_service(
            App::new()
                .app_data(web::Data::new(Arc::new(MockBlockchainService::new())))
                .app_data(web::Data::new(Arc::new(MockCache::new())))
                .service(node_routes(
                    Arc::new(MockBlockchainService::new()),
                    Arc::new(MockCache::new()),
                ))
        ).await;

        let req = test::TestRequest::get()
            .uri("/nodes/test-node-1")
            .to_request();

        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
    }

    #[actix_web::test]
    async fn test_node_performance() {
        let app = test::init_service(
            App::new()
                .app_data(web::Data::new(Arc::new(MockBlockchainService::new())))
                .app_data(web::Data::new(Arc::new(MockCache::new())))
                .service(node_routes(
                    Arc::new(MockBlockchainService::new()),
                    Arc::new(MockCache::new()),
                ))
        ).await;

        let req = test::TestRequest::get()
            .uri("/nodes/test-node-1/performance")
            .to_request();

        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());

        let performance: NodePerformance = test::read_body_json(resp).await;
        assert!(performance.reputation_score >= 0.0 && performance.reputation_score <= 100.0);
    }

    #[test]
    fn test_calculate_reputation_score() {
        let score = calculate_reputation_score(0.95, 0.98);
        assert!(score > 90.0); // High score for good performance

        let score = calculate_reputation_score(0.5, 0.5);
        assert!(score == 50.0); // Middle score for average performance

        let score = calculate_reputation_score(0.2, 0.3);
        assert!(score < 30.0); // Low score for poor performance
    }

    #[actix_web::test]
    async fn test_deregister_node() {
        let app = test::init_service(
            App::new()
                .app_data(web::Data::new(Arc::new(MockBlockchainService::new())))
                .service(node_routes(
                    Arc::new(MockBlockchainService::new()),
                    Arc::new(MockCache::new()),
                ))
        ).await;

        let req = test::TestRequest::delete()
            .uri("/nodes/test-node-1")
            .to_request();

        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), 204);
    }

    #[actix_web::test]
    async fn test_update_node() {
        let app = test::init_service(
            App::new()
                .app_data(web::Data::new(Arc::new(MockBlockchainService::new())))
                .service(node_routes(
                    Arc::new(MockBlockchainService::new()),
                    Arc::new(MockCache::new()),
                ))
        ).await;

        let update = NodeInfo {
            node_id: "test-node-1".to_string(),
            node_type: NodeType::Validator,
            endpoint: "https://new-endpoint.example.com".to_string(),
            capacity: 200,
            status: NodeStatus::Active,
            last_seen: Utc::now(),
            capabilities: vec!["LLM".to_string(), "CV".to_string()],
        };

        let req = test::TestRequest::put()
            .uri("/nodes/test-node-1")
            .set_json(&update)
            .to_request();

        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
    }

    #[actix_web::test]
    async fn test_aggregate_stats() {
        let app = test::init_service(
            App::new()
                .app_data(web::Data::new(Arc::new(MockBlockchainService::new())))
                .app_data(web::Data::new(Arc::new(MockCache::new())))
                .service(node_routes(
                    Arc::new(MockBlockchainService::new()),
                    Arc::new(MockCache::new()),
                ))
        ).await;

        let req = test::TestRequest::get()
            .uri("/nodes/stats/aggregate")
            .to_request();

        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
    }

    #[test]
    fn test_verify_node_key() {
        let valid_key = "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef";
        assert!(verify_node_key(valid_key));

        let invalid_key = "invalid-key";
        assert!(!verify_node_key(invalid_key));

        let wrong_prefix = "1x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef";
        assert!(!verify_node_key(wrong_prefix));
    }
}