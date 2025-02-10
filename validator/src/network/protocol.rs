use std::convert::TryFrom;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use anyhow::{Result, Context, anyhow};
use bytes::{Bytes, BytesMut, Buf, BufMut};
use serde::{Serialize, Deserialize};
use ed25519_dalek::{PublicKey, Signature};
use sha2::{Sha256, Digest};
use tokio_util::codec::{Decoder, Encoder};
use metrics::{counter, histogram};

// Protocol constants
pub const PROTOCOL_VERSION: u16 = 1;
pub const MAX_MESSAGE_SIZE: usize = 10 * 1024 * 1024; // 10MB
pub const MIN_PROTOCOL_VERSION: u16 = 1;
pub const MAGIC_BYTES: [u8; 4] = [0x56, 0x45, 0x54, 0x52]; // "VETR"

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolMessage {
    pub header: MessageHeader,
    pub payload: MessagePayload,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageHeader {
    pub magic: [u8; 4],
    pub version: u16,
    pub message_type: MessageType,
    pub payload_length: u32,
    pub timestamp: u64,
    pub sender: PublicKey,
    pub signature: Signature,
    pub message_id: [u8; 32],
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MessageType {
    // Handshake messages
    Hello,
    Welcome,
    Goodbye,
    
    // Peer discovery
    GetPeers,
    Peers,
    
    // Network status
    Ping,
    Pong,
    
    // Validation protocol
    SubmitModel,
    ValidationProposal,
    ValidationVote,
    ValidationResult,
    
    // Consensus messages
    ConsensusProposal,
    ConsensusVote,
    ConsensusCommit,
    
    // Error messages
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessagePayload {
    // Handshake payloads
    Hello {
        version: u16,
        node_id: PublicKey,
        features: Vec<String>,
        user_agent: String,
        timestamp: u64,
    },
    
    Welcome {
        node_id: PublicKey,
        features: Vec<String>,
    },
    
    // Peer discovery payloads
    PeerList {
        peers: Vec<PeerInfo>,
    },
    
    // Validation payloads
    SubmitModel {
        model_id: String,
        model_hash: [u8; 32],
        model_size: u64,
        model_format: String,
    },
    
    ValidationProposal {
        model_id: String,
        results: ValidationResults,
        proof: Vec<u8>,
    },
    
    ValidationVote {
        proposal_id: [u8; 32],
        vote: bool,
        reason: Option<String>,
    },
    
    ValidationResult {
        model_id: String,
        status: ValidationStatus,
        metrics: ValidationMetrics,
        signatures: Vec<(PublicKey, Signature)>,
    },
    
    // Consensus payloads
    ConsensusProposal {
        round: u64,
        proposal: Vec<u8>,
        proposer: PublicKey,
    },
    
    ConsensusVote {
        round: u64,
        vote: bool,
        voter: PublicKey,
    },
    
    // Error payload
    Error {
        code: u16,
        message: String,
    },
    
    // Empty payloads
    Ping(u64),
    Pong(u64),
    Goodbye,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerInfo {
    pub node_id: PublicKey,
    pub addresses: Vec<String>,
    pub features: Vec<String>,
    pub user_agent: String,
    pub last_seen: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResults {
    pub accuracy: f64,
    pub latency_ms: f64,
    pub memory_usage_mb: f64,
    pub throughput: f64,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationMetrics {
    pub processing_time: Duration,
    pub memory_peak: u64,
    pub cpu_usage: f64,
    pub gpu_usage: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ValidationStatus {
    Pending,
    InProgress,
    Success,
    Failed,
    Error,
}

// Protocol codec for encoding/decoding messages
pub struct ProtocolCodec {
    max_message_size: usize,
}

impl ProtocolCodec {
    pub fn new(max_message_size: usize) -> Self {
        Self { max_message_size }
    }
    
    fn verify_message(&self, msg: &ProtocolMessage) -> Result<()> {
        // Verify magic bytes
        if msg.header.magic != MAGIC_BYTES {
            return Err(anyhow!("Invalid magic bytes"));
        }
        
        // Verify protocol version
        if msg.header.version < MIN_PROTOCOL_VERSION {
            return Err(anyhow!("Protocol version too old"));
        }
        
        // Verify timestamp
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
            
        if msg.header.timestamp > now + 300 || msg.header.timestamp < now - 300 {
            return Err(anyhow!("Message timestamp out of bounds"));
        }
        
        // Verify message signature
        let message_bytes = self.serialize_message_content(&msg)?;
        if !self.verify_signature(&msg.header.sender, &message_bytes, &msg.header.signature) {
            return Err(anyhow!("Invalid message signature"));
        }
        
        Ok(())
    }
    
    fn serialize_message_content(&self, msg: &ProtocolMessage) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        
        // Serialize header fields except signature
        bytes.extend_from_slice(&msg.header.magic);
        bytes.extend_from_slice(&msg.header.version.to_be_bytes());
        bytes.extend_from_slice(&(msg.header.message_type as u8).to_be_bytes());
        bytes.extend_from_slice(&msg.header.payload_length.to_be_bytes());
        bytes.extend_from_slice(&msg.header.timestamp.to_be_bytes());
        bytes.extend_from_slice(msg.header.sender.as_bytes());
        
        // Serialize payload
        let payload_bytes = bincode::serialize(&msg.payload)?;
        bytes.extend_from_slice(&payload_bytes);
        
        Ok(bytes)
    }
    
    fn verify_signature(&self, pubkey: &PublicKey, message: &[u8], signature: &Signature) -> bool {
        pubkey.verify(message, signature).is_ok()
    }
    
    fn compute_message_id(msg: &ProtocolMessage) -> [u8; 32] {
        let mut hasher = Sha256::new();
        
        // Hash header fields
        hasher.update(&msg.header.magic);
        hasher.update(&msg.header.version.to_be_bytes());
        hasher.update(&[msg.header.message_type as u8]);
        hasher.update(&msg.header.payload_length.to_be_bytes());
        hasher.update(&msg.header.timestamp.to_be_bytes());
        hasher.update(msg.header.sender.as_bytes());
        
        // Hash payload
        if let Ok(payload_bytes) = bincode::serialize(&msg.payload) {
            hasher.update(&payload_bytes);
        }
        
        hasher.finalize().into()
    }
}

impl Decoder for ProtocolCodec {
    type Item = ProtocolMessage;
    type Error = anyhow::Error;
    
    fn decode(&mut self, src: &mut BytesMut) -> Result<Option<Self::Item>, Self::Error> {
        // Need at least a header to proceed
        if src.len() < std::mem::size_of::<MessageHeader>() {
            return Ok(None);
        }
        
        // Read and verify header
        let mut header_bytes = src.split_to(std::mem::size_of::<MessageHeader>());
        let header: MessageHeader = bincode::deserialize(&header_bytes)?;
        
        // Verify payload length
        if header.payload_length as usize > self.max_message_size {
            return Err(anyhow!("Message too large"));
        }
        
        // Need complete payload to proceed
        if src.len() < header.payload_length as usize {
            return Ok(None);
        }
        
        // Read payload
        let payload_bytes = src.split_to(header.payload_length as usize);
        let payload: MessagePayload = bincode::deserialize(&payload_bytes)?;
        
        let message = ProtocolMessage { header, payload };
        
        // Verify message
        self.verify_message(&message)?;
        
        counter!("vetras.protocol.messages_decoded", 1);
        Ok(Some(message))
    }
}

impl Encoder<ProtocolMessage> for ProtocolCodec {
    type Error = anyhow::Error;
    
    fn encode(&mut self, msg: ProtocolMessage, dst: &mut BytesMut) -> Result<(), Self::Error> {
        // Verify message before encoding
        self.verify_message(&msg)?;
        
        // Serialize header
        let header_bytes = bincode::serialize(&msg.header)?;
        dst.extend_from_slice(&header_bytes);
        
        // Serialize payload
        let payload_bytes = bincode::serialize(&msg.payload)?;
        dst.extend_from_slice(&payload_bytes);
        
        counter!("vetras.protocol.messages_encoded", 1);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::Keypair;
    use rand::rngs::OsRng;
    
    fn create_test_message() -> ProtocolMessage {
        let keypair = Keypair::generate(&mut OsRng);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
            
        ProtocolMessage {
            header: MessageHeader {
                magic: MAGIC_BYTES,
                version: PROTOCOL_VERSION,
                message_type: MessageType::Ping,
                payload_length: 8,
                timestamp: now,
                sender: keypair.public,
                signature: Signature::new([0u8; 64]),
                message_id: [0u8; 32],
            },
            payload: MessagePayload::Ping(now),
        }
    }
    
    #[test]
    fn test_message_encoding() {
        let mut codec = ProtocolCodec::new(MAX_MESSAGE_SIZE);
        let msg = create_test_message();
        let mut bytes = BytesMut::new();
        
        codec.encode(msg.clone(), &mut bytes).unwrap();
        
        let decoded = codec.decode(&mut bytes).unwrap().unwrap();
        assert_eq!(decoded.header.message_type, msg.header.message_type);
    }
    
    #[test]
    fn test_message_verification() {
        let codec = ProtocolCodec::new(MAX_MESSAGE_SIZE);
        let msg = create_test_message();
        
        assert!(codec.verify_message(&msg).is_ok());
    }
    
    #[test]
    fn test_invalid_message() {
        let mut codec = ProtocolCodec::new(MAX_MESSAGE_SIZE);
        let mut msg = create_test_message();
        msg.header.magic = [0; 4];
        
        let mut bytes = BytesMut::new();
        assert!(codec.encode(msg, &mut bytes).is_err());
    }
}
