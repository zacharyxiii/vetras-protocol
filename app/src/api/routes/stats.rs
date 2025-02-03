use actix_web::{web, HttpResponse, Scope};
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::services::blockchain_service::BlockchainService;
use crate::services::validation_service::ValidationService;
use crate::utils::error::{ApiError, ApiResult};
use crate::utils::cache::{Cache, CacheKey};
use crate::models::stats::{
    NetworkStats, ValidationStats, EconomicStats,
    NodeStats, TimeFrame, ModelStats, ConsensusStats
};

#[derive(Debug, Deserialize)]
pub struct StatsQuery {
    pub timeframe: Option<TimeFrame>,
    pub from_date: Option<DateTime<Utc>>,
    pub to_date: Option<DateTime<Utc>>,
}

#[derive(Debug, Serialize)]
pub struct NetworkOverview {
    pub total_validators: u64,
    pub active_validators: u64,
    pub total_stake: u64,
    pub total_validations: u64,
    pub success_rate: f64,
    pub average_consensus_time: f64,
    pub network_health: f64,
}

pub fn stats_routes(
    validation_service: Arc<ValidationService>,
    blockchain_service: Arc<BlockchainService>,
    cache: Arc<Cache>,
) -> Scope {
    web::scope("/stats")
        .service(
            web::resource("")
                .route(web::get().to(get_network_stats))
        )
        .service(
            web::resource("/validations")
                .route(web::get().to(get_validation_stats))
        )
        .service(
            web::resource("/economic")
                .route(web::get().to(get_economic_stats))
        )
        .service(
            web::resource("/nodes")
                .route(web::get().to(get_node_stats))
        )
        .service(
            web::resource("/models")
                .route(web::get().to(get_model_stats))
        )
        .service(
            web::resource("/consensus")
                .route(web::get().to(get_consensus_stats))
        )
        .service(
            web::resource("/overview")
                .route(web::get().to(get_network_overview))
        )
}

async fn get_network_stats(
    query: web::Query<StatsQuery>,
    validation_service: web::Data<Arc<ValidationService>>,
    blockchain_service: web::Data<Arc<BlockchainService>>,
    cache: web::Data<Arc<Cache>>,
) -> ApiResult<HttpResponse> {
    let cache_key = CacheKey::new("network_stats", &query);
    if let Some(cached) = cache.get(&cache_key).await {
        return Ok(HttpResponse::Ok().json(cached));
    }

    let timeframe = query.timeframe.unwrap_or(TimeFrame::Day);
    let to_date = query.to_date.unwrap_or_else(Utc::now);
    let from_date = query.from_date.unwrap_or_else(|| {
        match timeframe {
            TimeFrame::Hour => to_date - Duration::hours(1),
            TimeFrame::Day => to_date - Duration::days(1),
            TimeFrame::Week => to_date - Duration::weeks(1),
            TimeFrame::Month => to_date - Duration::days(30),
            TimeFrame::Year => to_date - Duration::days(365),
        }
    });

    let stats = NetworkStats {
        validation_stats: validation_service
            .get_validation_stats(from_date, to_date)
            .await
            .map_err(|e| ApiError::InternalError(e.to_string()))?,
        economic_stats: blockchain_service
            .get_economic_stats(from_date, to_date)
            .await
            .map_err(ApiError::BlockchainError)?,
        node_stats: blockchain_service
            .get_node_stats(from_date, to_date)
            .await
            .map_err(ApiError::BlockchainError)?,
        timestamp: Utc::now(),
    };

    cache.set(&cache_key, &stats, Duration::minutes(5)).await;
    Ok(HttpResponse::Ok().json(stats))
}

async fn get_validation_stats(
    query: web::Query<StatsQuery>,
    validation_service: web::Data<Arc<ValidationService>>,
    cache: web::Data<Arc<Cache>>,
) -> ApiResult<HttpResponse> {
    let cache_key = CacheKey::new("validation_stats", &query);
    if let Some(cached) = cache.get(&cache_key).await {
        return Ok(HttpResponse::Ok().json(cached));
    }

    let to_date = query.to_date.unwrap_or_else(Utc::now);
    let from_date = query.from_date.unwrap_or_else(|| to_date - Duration::days(1));

    let stats = validation_service
        .get_validation_stats(from_date, to_date)
        .await
        .map_err(|e| ApiError::InternalError(e.to_string()))?;

    cache.set(&cache_key, &stats, Duration::minutes(5)).await;
    Ok(HttpResponse::Ok().json(stats))
}

async fn get_economic_stats(
    query: web::Query<StatsQuery>,
    blockchain_service: web::Data<Arc<BlockchainService>>,
    cache: web::Data<Arc<Cache>>,
) -> ApiResult<HttpResponse> {
    let cache_key = CacheKey::new("economic_stats", &query);
    if let Some(cached) = cache.get(&cache_key).await {
        return Ok(HttpResponse::Ok().json(cached));
    }

    let to_date = query.to_date.unwrap_or_else(Utc::now);
    let from_date = query.from_date.unwrap_or_else(|| to_date - Duration::days(1));

    let stats = blockchain_service
        .get_economic_stats(from_date, to_date)
        .await
        .map_err(ApiError::BlockchainError)?;

    cache.set(&cache_key, &stats, Duration::minutes(5)).await;
    Ok(HttpResponse::Ok().json(stats))
}

async fn get_node_stats(
    query: web::Query<StatsQuery>,
    blockchain_service: web::Data<Arc<BlockchainService>>,
    cache: web::Data<Arc<Cache>>,
) -> ApiResult<HttpResponse> {
    let cache_key = CacheKey::new("node_stats", &query);
    if let Some(cached) = cache.get(&cache_key).await {
        return Ok(HttpResponse::Ok().json(cached));
    }

    let to_date = query.to_date.unwrap_or_else(Utc::now);
    let from_date = query.from_date.unwrap_or_else(|| to_date - Duration::days(1));

    let stats = blockchain_service
        .get_node_stats(from_date, to_date)
        .await
        .map_err(ApiError::BlockchainError)?;

    cache.set(&cache_key, &stats, Duration::minutes(5)).await;
    Ok(HttpResponse::Ok().json(stats))
}

async fn get_model_stats(
    query: web::Query<StatsQuery>,
    validation_service: web::Data<Arc<ValidationService>>,
    cache: web::Data<Arc<Cache>>,
) -> ApiResult<HttpResponse> {
    let cache_key = CacheKey::new("model_stats", &query);
    if let Some(cached) = cache.get(&cache_key).await {
        return Ok(HttpResponse::Ok().json(cached));
    }

    let to_date = query.to_date.unwrap_or_else(Utc::now);
    let from_date = query.from_date.unwrap_or_else(|| to_date - Duration::days(1));

    let stats = validation_service
        .get_model_stats(from_date, to_date)
        .await
        .map_err(|e| ApiError::InternalError(e.to_string()))?;

    cache.set(&cache_key, &stats, Duration::minutes(5)).await;
    Ok(HttpResponse::Ok().json(stats))
}

async fn get_consensus_stats(
    query: web::Query<StatsQuery>,
    validation_service: web::Data<Arc<ValidationService>>,
    cache: web::Data<Arc<Cache>>,
) -> ApiResult<HttpResponse> {
    let cache_key = CacheKey::new("consensus_stats", &query);
    if let Some(cached) = cache.get(&cache_key).await {
        return Ok(HttpResponse::Ok().json(cached));
    }

    let to_date = query.to_date.unwrap_or_else(Utc::now);
    let from_date = query.from_date.unwrap_or_else(|| to_date - Duration::days(1));

    let stats = validation_service
        .get_consensus_stats(from_date, to_date)
        .await
        .map_err(|e| ApiError::InternalError(e.to_string()))?;

    cache.set(&cache_key, &stats, Duration::minutes(5)).await;
    Ok(HttpResponse::Ok().json(stats))
}

async fn get_network_overview(
    validation_service: web::Data<Arc<ValidationService>>,
    blockchain_service: web::Data<Arc<BlockchainService>>,
    cache: web::Data<Arc<Cache>>,
) -> ApiResult<HttpResponse> {
    let cache_key = CacheKey::new("network_overview", &());
    if let Some(cached) = cache.get(&cache_key).await {
        return Ok(HttpResponse::Ok().json(cached));
    }

    // Get current network state
    let node_stats = blockchain_service
        .get_current_node_stats()
        .await
        .map_err(ApiError::BlockchainError)?;

    let validation_stats = validation_service
        .get_current_validation_stats()
        .await
        .map_err(|e| ApiError::InternalError(e.to_string()))?;

    let overview = NetworkOverview {
        total_validators: node_stats.total_validators,
        active_validators: node_stats.active_validators,
        total_stake: node_stats.total_stake,
        total_validations: validation_stats.total_validations,
        success_rate: validation_stats.success_rate,
        average_consensus_time: validation_stats.average_consensus_time,
        network_health: calculate_network_health(&node_stats, &validation_stats),
    };

    cache.set(&cache_key, &overview, Duration::minutes(1)).await;
    Ok(HttpResponse::Ok().json(overview))
}

fn calculate_network_health(node_stats: &NodeStats, validation_stats: &ValidationStats) -> f64 {
    let validator_health = (node_stats.active_validators as f64) / (node_stats.total_validators as f64);
    let validation_health = validation_stats.success_rate;
    let stake_distribution = node_stats.stake_distribution_score;
    
    // Weight the factors
    0.4 * validator_health + 0.4 * validation_health + 0.2 * stake_distribution
}

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::test;
    use crate::test_utils::{
        mock_validation_service::MockValidationService,
        mock_blockchain_service::MockBlockchainService,
        mock_cache::MockCache,
    };

    #[actix_web::test]
    async fn test_get_network_stats() {
        // Create mocks
        let mut validation_service = MockValidationService::new();
        validation_service
            .expect_get_validation_stats()
            .return_once(|_, _| {
                Ok(ValidationStats {
                    total_validations: 1000,
                    successful_validations: 950,
                    failed_validations: 50,
                    success_rate: 0.95,
                    average_validation_time: 120.5,
                    // ... other fields
                })
            });

        // Create test app
        let app = test::init_service(
            App::new()
                .app_data(web::Data::new(Arc::new(validation_service)))
                .app_data(web::Data::new(Arc::new(MockBlockchainService::new())))
                .app_data(web::Data::new(Arc::new(MockCache::new())))
                .service(stats_routes(
                    Arc::new(MockValidationService::new()),
                    Arc::new(MockBlockchainService::new()),
                    Arc::new(MockCache::new()),
                ))
        ).await;

        // Create test request
        let req = test::TestRequest::get()
            .uri("/stats?timeframe=day")
            .to_request();

        // Send request and verify response
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        let stats: NetworkStats = test::read_body_json(resp).await;
        assert!(stats.validation_stats.success_rate > 0.94);
    }
}