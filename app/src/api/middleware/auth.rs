use actix_web::{
    dev::{forward_ready, Service, ServiceRequest, ServiceResponse, Transform},
    Error, HttpMessage,
};
use futures::future::{ready, LocalBoxFuture, Ready};
use jsonwebtoken::{decode, DecodingKey, Validation, Algorithm};
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll};

use crate::utils::error::{ApiError, ApiResult};

const VETRAS_JWT_SECRET: &str = "VETRAS_JWT_SECRET";

#[derive(Debug, Serialize, Deserialize)]
struct Claims {
    sub: String,
    exp: usize,
    role: String,
    validator_pubkey: Option<String>,
}

pub struct Auth;

impl Auth {
    pub fn validator_only() -> AuthMiddleware {
        AuthMiddleware {
            required_role: Some("validator".to_string()),
        }
    }

    pub fn admin_only() -> AuthMiddleware {
        AuthMiddleware {
            required_role: Some("admin".to_string()),
        }
    }
}

pub struct AuthMiddleware {
    required_role: Option<String>,
}

impl<S, B> Transform<S, ServiceRequest> for AuthMiddleware
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error> + 'static,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type InitError = ();
    type Transform = AuthMiddlewareService<S>;
    type Future = Ready<Result<Self::Transform, Self::InitError>>;

    fn new_transform(&self, service: S) -> Self::Future {
        ready(Ok(AuthMiddlewareService {
            service: Rc::new(service),
            required_role: self.required_role.clone(),
        }))
    }
}

pub struct AuthMiddlewareService<S> {
    service: Rc<S>,
    required_role: Option<String>,
}

impl<S, B> Service<ServiceRequest> for AuthMiddlewareService<S>
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error> + 'static,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type Future = LocalBoxFuture<'static, Result<Self::Response, Self::Error>>;

    forward_ready!(service);

    fn call(&self, req: ServiceRequest) -> Self::Future {
        let svc = self.service.clone();
        let required_role = self.required_role.clone();

        Box::pin(async move {
            // Extract token from Authorization header
            let token = match req.headers().get("Authorization") {
                Some(auth_header) => {
                    let auth_str = auth_header.to_str().map_err(|_| {
                        ApiError::AuthenticationError("Invalid authorization header".into())
                    })?;
                    if !auth_str.starts_with("Bearer ") {
                        return Err(ApiError::AuthenticationError("Invalid token format".into()).into());
                    }
                    auth_str[7..].to_string()
                }
                None => {
                    return Err(ApiError::AuthenticationError("Missing authorization header".into()).into())
                }
            };

            // Verify and decode token
            let secret = std::env::var(VETRAS_JWT_SECRET).expect("JWT secret must be set");
            let key = DecodingKey::from_secret(secret.as_bytes());
            let mut validation = Validation::new(Algorithm::HS256);
            validation.validate_exp = true;

            let token_data = decode::<Claims>(&token, &key, &validation).map_err(|e| {
                ApiError::AuthenticationError(format!("Invalid token: {}", e))
            })?;

            let claims = token_data.claims;

            // Check role if required
            if let Some(required_role) = required_role {
                if claims.role != required_role {
                    return Err(ApiError::AuthorizationError("Insufficient permissions".into()).into());
                }
            }

            // Add claims to request extensions
            req.extensions_mut().insert(claims);

            // Continue with the request
            let res = svc.call(req).await?;
            Ok(res)
        })
    }
}

// Helper to extract validated user claims
pub fn get_claims(req: &ServiceRequest) -> ApiResult<Claims> {
    req.extensions()
        .get::<Claims>()
        .cloned()
        .ok_or_else(|| ApiError::AuthenticationError("No authentication claims found".into()))
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
    async fn test_auth_middleware_missing_token() {
        let app = test::init_service(
            App::new()
                .wrap(Auth::validator_only())
                .route("/test", web::get().to(test_handler)),
        )
        .await;

        let req = TestRequest::get().uri("/test").to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status().as_u16(), 401);
    }

    #[actix_web::test]
    async fn test_auth_middleware_invalid_token() {
        let app = test::init_service(
            App::new()
                .wrap(Auth::validator_only())
                .route("/test", web::get().to(test_handler)),
        )
        .await;

        let req = TestRequest::get()
            .uri("/test")
            .header("Authorization", "Bearer invalid_token")
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status().as_u16(), 401);
    }

    // Additional tests would require mock JWT creation
}