use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, mpsc, broadcast};
use tokio::net::{TcpListener, TcpStream};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use anyhow::{Result, Context, anyhow};
use serde::{Serialize, Deserialize};
use ed25519_dalek::{PublicKey, Keypair, Signature};
use metrics::{counter, gauge, histogram};
use tracing::{info, warn, error, debug};

const MAX_MESSAGE_SIZE: usize = 10 * 1024 * 1024; // 10MB
const PEER_TIMEOUT: Duration = Duration::from_secs(30);
const DISCOVERY_INTERVAL: Duration = Duration::from_secs(60);
const MAX_PEERS: usize = 50;
const PING_INTERVAL: Duration = Duration::from_secs(15);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerInfo {
    pub address: SocketAddr,
    pub public_key: PublicKey,
    pub version: String,
    pub features: Vec<String>,
    pub node_type: NodeType,
    pub reputation: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NodeType {
    Validator,
    Observer,
    Bootstrap,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMessage {
    pub message_type: MessageType,
    pub sender: PublicKey,
    pub signature: Signature,
    pub timestamp: i64,
    pub payload: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageType {
    Discovery,
    Ping,
    Pong,
    PeerList,
    ValidationMessage,
    Disconnect,
}

pub struct PeerManager {
    keypair: Keypair,
    listen_addr: SocketAddr,
    peers: Arc<RwLock<HashMap<PublicKey, ConnectedPeer>>>,
    known_addresses: Arc<RwLock<HashSet<SocketAddr>>>,
    bootstrap_nodes: Vec<SocketAddr>,
    message_tx: broadcast::Sender<NetworkMessage>,
    message_rx: broadcast::Receiver<NetworkMessage>,
    validation_tx: mpsc::Sender<Vec<u8>>,
    node_info: PeerInfo,
}

struct ConnectedPeer {
    info: PeerInfo,
    stream: TcpStream,
    last_seen: Instant,
    messages_received: u64,
    bytes_transferred: u64,
}

impl PeerManager {
    pub async fn new(
        keypair: Keypair,
        listen_addr: SocketAddr,
        bootstrap_nodes: Vec<SocketAddr>,
        validation_tx: mpsc::Sender<Vec<u8>>,
    ) -> Result<Self> {
        let (message_tx, message_rx) = broadcast::channel(1000);
        
        let node_info = PeerInfo {
            address: listen_addr,
            public_key: keypair.public,
            version: env!("CARGO_PKG_VERSION").to_string(),
            features: vec!["validation".to_string()],
            node_type: NodeType::Validator,
            reputation: 0,
        };

        Ok(Self {
            keypair,
            listen_addr,
            peers: Arc::new(RwLock::new(HashMap::new())),
            known_addresses: Arc::new(RwLock::new(HashSet::new())),
            bootstrap_nodes,
            message_tx,
            message_rx,
            validation_tx,
            node_info,
        })
    }

    pub async fn start(&self) -> Result<()> {
        let listener = TcpListener::bind(self.listen_addr).await
            .context("Failed to bind to address")?;
            
        info!("P2P node listening on {}", self.listen_addr);

        // Start peer discovery
        self.start_discovery().await?;
        
        // Start peer maintenance
        self.start_maintenance().await?;
        
        // Handle incoming connections
        loop {
            match listener.accept().await {
                Ok((stream, addr)) => {
                    debug!("New incoming connection from {}", addr);
                    self.handle_incoming_connection(stream).await?;
                }
                Err(e) => {
                    error!("Error accepting connection: {}", e);
                }
            }
        }
    }

    async fn start_discovery(&self) -> Result<()> {
        let peers = self.peers.clone();
        let known_addresses = self.known_addresses.clone();
        let bootstrap_nodes = self.bootstrap_nodes.clone();
        let node_info = self.node_info.clone();
        
        tokio::spawn(async move {
            loop {
                // Connect to bootstrap nodes if we have few peers
                if peers.read().await.len() < MAX_PEERS / 2 {
                    for &addr in &bootstrap_nodes {
                        if !known_addresses.read().await.contains(&addr) {
                            if let Err(e) = Self::connect_to_peer(addr, &node_info).await {
                                warn!("Failed to connect to bootstrap node {}: {}", addr, e);
                            }
                        }
                    }
                }
                
                tokio::time::sleep(DISCOVERY_INTERVAL).await;
            }
        });
        
        Ok(())
    }

    async fn start_maintenance(&self) -> Result<()> {
        let peers = self.peers.clone();
        let message_tx = self.message_tx.clone();
        
        tokio::spawn(async move {
            loop {
                let mut to_remove = Vec::new();
                
                // Check peer health and send pings
                {
                    let mut peers_lock = peers.write().await;
                    for (pubkey, peer) in peers_lock.iter_mut() {
                        if peer.last_seen.elapsed() > PEER_TIMEOUT {
                            to_remove.push(*pubkey);
                        } else if peer.last_seen.elapsed() > PING_INTERVAL {
                            let ping = NetworkMessage {
                                message_type: MessageType::Ping,
                                sender: pubkey.clone(),
                                signature: Signature::new([0u8; 64]), // Placeholder
                                timestamp: chrono::Utc::now().timestamp(),
                                payload: vec![],
                            };
                            
                            if let Err(e) = message_tx.send(ping) {
                                error!("Failed to send ping: {}", e);
                            }
                        }
                    }
                    
                    // Remove disconnected peers
                    for pubkey in to_remove {
                        peers_lock.remove(&pubkey);
                        counter!("vetras.network.peer_disconnected", 1);
                    }
                }
                
                gauge!("vetras.network.active_peers", peers.read().await.len() as f64);
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        });
        
        Ok(())
    }

    async fn handle_incoming_connection(&self, mut stream: TcpStream) -> Result<()> {
        let peer_addr = stream.peer_addr()?;
        
        // Read handshake
        let mut size_buf = [0u8; 4];
        stream.read_exact(&mut size_buf).await?;
        let size = u32::from_be_bytes(size_buf) as usize;
        
        if size > MAX_MESSAGE_SIZE {
            return Err(anyhow!("Message too large"));
        }
        
        let mut data = vec![0u8; size];
        stream.read_exact(&mut data).await?;
        
        let peer_info: PeerInfo = bincode::deserialize(&data)?;
        
        // Verify peer hasn't exceeded max connections
        if self.peers.read().await.len() >= MAX_PEERS {
            return Err(anyhow!("Max peers reached"));
        }
        
        // Add to connected peers
        let mut peers = self.peers.write().await;
        peers.insert(peer_info.public_key, ConnectedPeer {
            info: peer_info.clone(),
            stream,
            last_seen: Instant::now(),
            messages_received: 0,
            bytes_transferred: 0,
        });
        
        counter!("vetras.network.peer_connected", 1);
        info!("New peer connected: {}", peer_addr);
        
        Ok(())
    }

    async fn connect_to_peer(addr: SocketAddr, node_info: &PeerInfo) -> Result<()> {
        let mut stream = TcpStream::connect(addr).await
            .context("Failed to connect to peer")?;
            
        // Send handshake
        let data = bincode::serialize(node_info)?;
        let size = data.len() as u32;
        
        stream.write_all(&size.to_be_bytes()).await?;
        stream.write_all(&data).await?;
        
        Ok(())
    }

    pub async fn broadcast(&self, message: NetworkMessage) -> Result<()> {
        let peers = self.peers.read().await;
        let message_data = bincode::serialize(&message)?;
        
        for peer in peers.values() {
            if let Err(e) = self.send_to_peer(&peer.stream, &message_data).await {
                warn!("Failed to send message to peer {}: {}", peer.info.address, e);
            }
        }
        
        counter!("vetras.network.messages_broadcast", 1);
        Ok(())
    }

    async fn send_to_peer(stream: &TcpStream, data: &[u8]) -> Result<()> {
        let size = data.len() as u32;
        let mut stream = stream.try_clone().await?;
        
        stream.write_all(&size.to_be_bytes()).await?;
        stream.write_all(data).await?;
        
        Ok(())
    }

    pub fn get_peer_count(&self) -> usize {
        self.peers.try_read().map(|p| p.len()).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::net::TcpStream;

    async fn setup_test_peer() -> (PeerManager, SocketAddr) {
        let keypair = Keypair::generate(&mut rand::thread_rng());
        let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let (tx, _) = mpsc::channel(100);
        
        let peer = PeerManager::new(
            keypair,
            addr,
            vec![],
            tx,
        ).await.unwrap();
        
        (peer, addr)
    }

    #[tokio::test]
    async fn test_peer_connection() {
        let (peer1, addr1) = setup_test_peer().await;
        let (peer2, _) = setup_test_peer().await;
        
        // Connect peers
        let stream = TcpStream::connect(addr1).await.unwrap();
        peer1.handle_incoming_connection(stream).await.unwrap();
        
        assert_eq!(peer1.get_peer_count(), 1);
    }

    #[tokio::test]
    async fn test_message_broadcast() {
        let (peer, _) = setup_test_peer().await;
        
        let message = NetworkMessage {
            message_type: MessageType::Ping,
            sender: peer.keypair.public,
            signature: Signature::new([0u8; 64]),
            timestamp: chrono::Utc::now().timestamp(),
            payload: vec![1, 2, 3],
        };
        
        peer.broadcast(message).await.unwrap();
    }
}
