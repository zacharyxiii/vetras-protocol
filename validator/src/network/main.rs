use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, error, Level};
use tracing_subscriber::FmtSubscriber;
use clap::{App, Arg};
use anyhow::Result;

use crate::{
    ai::{
        evaluator::ModelEvaluator,
        llm::LLMEngine,
        models::ModelRegistry,
    },
    consensus::{
        engine::ConsensusEngine,
        types::ValidatorState,
    },
    network::{
        peer::PeerManager,
        protocol::NetworkProtocol,
    },
    storage::{
        database::Database,
        ipfs::IPFSClient,
    },
};

// Configuration constants
const DEFAULT_CONFIG_PATH: &str = "config/default.toml";
const DEFAULT_RPC_PORT: u16 = 8899;
const DEFAULT_P2P_PORT: u16 = 8900;

pub struct ValidatorNode {
    state: Arc<RwLock<ValidatorState>>,
    model_evaluator: Arc<ModelEvaluator>,
    consensus_engine: Arc<ConsensusEngine>,
    peer_manager: Arc<PeerManager>,
    database: Arc<Database>,
    ipfs: Arc<IPFSClient>,
    llm_engine: Arc<LLMEngine>,
    model_registry: Arc<ModelRegistry>,
}

impl ValidatorNode {
    pub async fn new(config_path: &str) -> Result<Self> {
        info!("Initializing VETRAS validator node...");
        
        // Load configuration
        let config = Self::load_config(config_path)?;
        
        // Initialize components
        let database = Arc::new(Database::connect(&config.database_url).await?);
        let ipfs = Arc::new(IPFSClient::new(&config.ipfs_endpoint)?);
        
        // Initialize AI components
        let llm_engine = Arc::new(LLMEngine::new(&config.llm_config).await?);
        let model_registry = Arc::new(ModelRegistry::new(database.clone()));
        let model_evaluator = Arc::new(ModelEvaluator::new(
            llm_engine.clone(),
            model_registry.clone(),
        ));

        // Initialize network components
        let state = Arc::new(RwLock::new(ValidatorState::new(config.validator_pubkey)));
        let peer_manager = Arc::new(PeerManager::new(
            config.p2p_port,
            config.bootstrap_peers,
        ).await?);
        
        // Initialize consensus engine
        let consensus_engine = Arc::new(ConsensusEngine::new(
            state.clone(),
            peer_manager.clone(),
            config.consensus_config,
        ));

        Ok(Self {
            state,
            model_evaluator,
            consensus_engine,
            peer_manager,
            database,
            ipfs,
            llm_engine,
            model_registry,
        })
    }

    pub async fn start(&self) -> Result<()> {
        info!("Starting VETRAS validator node...");

        // Start network services
        self.peer_manager.start().await?;
        
        // Start consensus engine
        self.consensus_engine.start().await?;
        
        // Start model evaluation service
        self.start_model_evaluation().await?;
        
        // Start metrics collection
        self.start_metrics_collection().await?;
        
        info!("VETRAS validator node is running");
        Ok(())
    }

    async fn start_model_evaluation(&self) -> Result<()> {
        let evaluator = self.model_evaluator.clone();
        let state = self.state.clone();
        let db = self.database.clone();
        
        tokio::spawn(async move {
            loop {
                if let Err(e) = evaluator.process_pending_evaluations(&state, &db).await {
                    error!("Error processing evaluations: {}", e);
                }
                tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
            }
        });
        
        Ok(())
    }

    async fn start_metrics_collection(&self) -> Result<()> {
        let state = self.state.clone();
        let db = self.database.clone();
        
        tokio::spawn(async move {
            loop {
                if let Err(e) = Self::collect_and_store_metrics(&state, &db).await {
                    error!("Error collecting metrics: {}", e);
                }
                tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;
            }
        });
        
        Ok(())
    }

    async fn collect_and_store_metrics(
        state: &RwLock<ValidatorState>,
        db: &Database,
    ) -> Result<()> {
        let metrics = {
            let state = state.read().await;
            state.collect_metrics()
        };
        
        db.store_metrics(metrics).await?;
        Ok(())
    }

    fn load_config(config_path: &str) -> Result<Config> {
        let config_str = std::fs::read_to_string(config_path)?;
        let config: Config = toml::from_str(&config_str)?;
        Ok(config)
    }

    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down VETRAS validator node...");
        
        // Stop consensus engine
        self.consensus_engine.stop().await?;
        
        // Stop peer manager
        self.peer_manager.stop().await?;
        
        // Close database connection
        self.database.close().await?;
        
        info!("Shutdown complete");
        Ok(())
    }
}

#[derive(Debug, serde::Deserialize)]
pub struct Config {
    validator_pubkey: String,
    database_url: String,
    ipfs_endpoint: String,
    llm_config: LLMConfig,
    consensus_config: ConsensusConfig,
    p2p_port: u16,
    bootstrap_peers: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Parse command line arguments
    let matches = App::new("VETRAS Validator Node")
        .version(env!("CARGO_PKG_VERSION"))
        .author("VETRAS Team")
        .about("AI Model Validation Node")
        .arg(Arg::with_name("config")
            .short('c')
            .long("config")
            .value_name("FILE")
            .help("Sets the configuration file path")
            .takes_value(true))
        .get_matches();

    // Setup logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_file(true)
        .with_line_number(true)
        .with_thread_ids(true)
        .with_target(false)
        .with_thread_names(true)
        .with_ansi(true)
        .pretty()
        .build();
    tracing::subscriber::set_global_default(subscriber)?;

    // Initialize and start validator node
    let config_path = matches.value_of("config").unwrap_or(DEFAULT_CONFIG_PATH);
    let node = ValidatorNode::new(config_path).await?;
    
    // Handle shutdown signals
    let node_clone = node.clone();
    ctrlc::set_handler(move || {
        info!("Received shutdown signal");
        tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(node_clone.shutdown())
            .unwrap();
        std::process::exit(0);
    })?;

    // Start the node
    node.start().await?;

    // Keep the main thread alive
    tokio::signal::ctrl_c().await?;
    node.shutdown().await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_validator_node_initialization() {
        // Create temporary config file
        let config = r#"
            validator_pubkey = "test_pubkey"
            database_url = "postgres://localhost/test"
            ipfs_endpoint = "http://localhost:5001"
            p2p_port = 8900
            bootstrap_peers = []
            
            [llm_config]
            endpoint = "http://localhost:8080"
            model_name = "test-model"
            
            [consensus_config]
            threshold = 0.75
            round_timeout = 30
        "#;
        
        let mut temp_file = NamedTempFile::new().unwrap();
        std::io::Write::write_all(&mut temp_file, config.as_bytes()).unwrap();
        
        // Test node initialization
        let node = ValidatorNode::new(temp_file.path().to_str().unwrap()).await;
        assert!(node.is_ok());
    }

    #[tokio::test]
    async fn test_validator_node_shutdown() {
        // Create minimal test configuration
        let config = r#"
            validator_pubkey = "test_pubkey"
            database_url = "postgres://localhost/test"
            ipfs_endpoint = "http://localhost:5001"
            p2p_port = 8901
            bootstrap_peers = []
            
            [llm_config]
            endpoint = "http://localhost:8080"
            model_name = "test-model"
            
            [consensus_config]
            threshold = 0.75
            round_timeout = 30
        "#;
        
        let mut temp_file = NamedTempFile::new().unwrap();
        std::io::Write::write_all(&mut temp_file, config.as_bytes()).unwrap();
        
        // Initialize and test shutdown
        let node = ValidatorNode::new(temp_file.path().to_str().unwrap()).await.unwrap();
        let shutdown_result = node.shutdown().await;
        assert!(shutdown_result.is_ok());
    }
}
