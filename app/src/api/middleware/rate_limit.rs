use actix_web::{
    dev::{forward_ready, Service, ServiceRequest, ServiceResponse, Transform},
    Error, HttpResponse,
};
use futures::future::{ready, Ready};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::sync::RwLock;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::rc::Rc;

#[derive(Clone)]
pub struct RateLimit {
    requests_per_second: u32,
    burst_size: u32,
}

#[derive(Debug)]
struct RateLimitState {
    last_checked: Instant,
    tokens: f64,
}

impl RateLimit {
    pub fn new(requests_per_second: u32, burst_size: u32) -> Self {
        Self {
            requests_per_second,
            burst_size,
        }
    }
}

#[derive(Clone)]
struct RateLimitStore {
    limits: Arc<RwLock<HashMap<String, RateLimitState>>>,
    cleanup_interval: Duration,
    last_cleanup: Arc<RwLock<Instant>>,
}

impl RateLimitStore {
    fn new(cleanup_interval: Duration) -> Self {
        Self {
            limits: Arc::new(RwLock::new(HashMap::new())),
            cleanup_interval,
            last_cleanup: Arc::new(RwLock::new(Instant::now())),
        }
    }

    async fn cleanup(&self) {
        let mut last_cleanup = self.last_cleanup.write().await;
        if last_cleanup.elapsed() >= self.cleanup_interval {
            let mut limits = self.limits.write().await;
            limits.retain(|_, state| state.last_checked.elapsed() < self.cleanup_interval);
            *last_cleanup = Instant::now();
        }
    }
}

impl<S, B> Transform<S, ServiceRequest> for RateLimit
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error> + 'static,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type InitError = ();
    type Transform = RateLimitMiddleware<S>;
    type Future = Ready<Result<Self::Transform, Self::InitError>>;

    fn new_transform(&self, service: S) -> Self::Future {
        ready(Ok(RateLimitMiddleware {
            service: Rc::new(service),
            store: RateLimitStore::new(Duration::from_secs(3600)),
            requests_per_second: self.requests_per_second,
            burst_size: self.burst_size,
        }))
    }
}

pub struct RateLimitMiddleware<S> {
    service: Rc<S>,
    store: RateLimitStore,
    requests_per_second: u32,
    burst_size: u32,
}

impl<S, B> Service<ServiceRequest> for RateLimitMiddleware<S>
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error> + 'static,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>>>>;

    forward_ready!(service);

    fn call(&self, req: ServiceRequest) -> Self::Future {
        let svc = self.service.clone();
        let store = self.store.clone();
        let rps = self.requests_per_second as f64;
        let burst = self.burst_size as f64;

        Box::pin(async move {
            // Clean up old entries
            store.cleanup().await;

            // Get client identifier (IP address or API key)
            let client_id = match req.headers().get("X-API-Key") {
                Some(api_key) => api_key.to_str().unwrap_or("default").to_string(),
                None => req.peer_addr()
                    .map(|addr| addr.ip().to_string())
                    .unwrap_or_else(|| "unknown".to_string()),
            };

            let now = Instant::now();
            let mut limits = store.limits.write().await;

            let state = limits.entry(client_id).or_insert_with(|| RateLimitState {
                last_checked: now,
                tokens: burst,
            });

            // Calculate token replenishment
            let elapsed = now.duration_since(state.last_checked).as_secs_f64();
            state.tokens = (state.tokens + elapsed * rps).min(burst);
            state.last_checked = now;

            // Check if we have enough tokens
            if state.tokens < 1.0 {
                return Ok(ServiceResponse::new(
                    req.into_parts().0,
                    HttpResponse::TooManyRequests()
                        .insert_header(("Retry-After", "1"))
                        .finish(),
                ));
            }

            // Consume a token
            state.tokens -= 1.0;
            drop(limits);

            // Process the request
            let res = svc.call(req).await?;
            Ok(res)
        })
    }
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
    async fn test_rate_limit_basic() {
        let app = test::init_service(
            App::new()
                .wrap(RateLimit::new(2, 2)) // 2 requests per second, burst of 2
                .route("/test", web::get().to(test_handler)),
        )
        .await;

        // First request should succeed
        let req = TestRequest::get().uri("/test").to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status().as_u16(), 200);

        // Second request should succeed
        let req = TestRequest::get().uri("/test").to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status().as_u16(), 200);

        // Third request should fail
        let req = TestRequest::get().uri("/test").to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status().as_u16(), 429);
    }

    #[actix_web::test]
    async fn test_rate_limit_with_api_key() {
        let app = test::init_service(
            App::new()
                .wrap(RateLimit::new(1, 1))
                .route("/test", web::get().to(test_handler)),
        )
        .await;

        // Request with API key should succeed
        let req = TestRequest::get()
            .uri("/test")
            .header("X-API-Key", "test-key")
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status().as_u16(), 200);

        // Second request with same API key should fail
        let req = TestRequest::get()
            .uri("/test")
            .header("X-API-Key", "test-key")
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status().as_u16(), 429);
    }
}