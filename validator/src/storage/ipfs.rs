use std::path::{Path, PathBuf};
use std::time::Duration;
use anyhow::{Result, Context, anyhow};
use ipfs_api_backend_hyper::{IpfsApi, IpfsClient, TryFromUri};
use tokio::fs::File;
use tokio::io::AsyncReadExt;
use futures::{StreamExt, TryStreamExt};
use serde::{Serialize, Deserialize};
use metrics::{counter, gauge, histogram};
use cid::Cid;
use blake3;
use tracing::{info, warn, error};

const IPFS_TIMEOUT: Duration = Duration::from_secs(60);
const MAX_MODEL_SIZE: u64 = 10 * 1024 * 1024 * 1024; // 10GB
const PIN_TIMEOUT: Duration = Duration::from_secs(300);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelLocation {
    pub cid: String,
    pub size: u64,
    pub multiaddrs: Vec<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub pinned: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageMetrics {
    pub total_stored: u64,
    pub total_retrieved: u64,
    pub pin_count: u64,
    pub average_retrieval_time: f64,
}

pub struct IpfsStorage {
    client: IpfsClient,
    local_cache: PathBuf,
    metrics: StorageMetrics,
}

impl IpfsStorage {
    pub async fn new<P: AsRef<Path>>(ipfs_api: &str, cache_dir: P) -> Result<Self> {
        let client = IpfsClient::from_str(ipfs_api)
            .context("Failed to create IPFS client")?;
            
        let cache_dir = cache_dir.as_ref().to_path_buf();
        tokio::fs::create_dir_all(&cache_dir).await
            .context("Failed to create cache directory")?;
            
        Ok(Self {
            client,
            local_cache: cache_dir,
            metrics: StorageMetrics {
                total_stored: 0,
                total_retrieved: 0,
                pin_count: 0,
                average_retrieval_time: 0.0,
            },
        })
    }

    pub async fn store_model<P: AsRef<Path>>(&mut self, path: P) -> Result<ModelLocation> {
        let path = path.as_ref();
        let file_size = tokio::fs::metadata(path).await?.len();
        
        if file_size > MAX_MODEL_SIZE {
            return Err(anyhow!("Model exceeds maximum size limit"));
        }
        
        // Read file and compute hash
        let mut file = File::open(path).await?;
        let mut data = Vec::new();
        file.read_to_end(&mut data).await?;
        
        let hash = self.compute_hash(&data);
        info!("Storing model with hash: {}", hex::encode(&hash));
        
        // Add to IPFS
        let response = tokio::time::timeout(
            IPFS_TIMEOUT,
            self.client.add(data)
        ).await??;
        
        // Pin the content
        self.pin_content(&response.hash).await?;
        
        // Get node addresses
        let addrs = self.get_multiaddrs(&response.hash).await?;
        
        let location = ModelLocation {
            cid: response.hash,
            size: file_size,
            multiaddrs: addrs,
            timestamp: chrono::Utc::now(),
            pinned: true,
        };
        
        // Update metrics
        self.metrics.total_stored += file_size;
        self.metrics.pin_count += 1;
        gauge!("vetras.ipfs.stored_bytes", self.metrics.total_stored as f64);
        counter!("vetras.ipfs.models_stored", 1);
        
        Ok(location)
    }

    pub async fn retrieve_model(&mut self, location: &ModelLocation) -> Result<PathBuf> {
        let start = std::time::Instant::now();
        let cache_path = self.local_cache.join(&location.cid);
        
        // Check cache first
        if cache_path.exists() {
            let metadata = tokio::fs::metadata(&cache_path).await?;
            if metadata.len() == location.size {
                counter!("vetras.ipfs.cache_hits", 1);
                return Ok(cache_path);
            }
        }
        
        // Retrieve from IPFS
        info!("Retrieving model: {}", location.cid);
        let mut stream = self.client.cat(&location.cid).await?;
        let mut file = File::create(&cache_path).await?;
        
        let mut downloaded_size = 0u64;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            tokio::io::AsyncWriteExt::write_all(&mut file, &chunk).await?;
            downloaded_size += chunk.len() as u64;
            
            if downloaded_size > location.size {
                return Err(anyhow!("Downloaded size exceeds expected size"));
            }
        }
        
        if downloaded_size != location.size {
            return Err(anyhow!("Incomplete download"));
        }
        
        // Update metrics
        self.metrics.total_retrieved += location.size;
        let retrieval_time = start.elapsed().as_secs_f64();
        self.update_average_retrieval_time(retrieval_time);
        
        gauge!("vetras.ipfs.retrieved_bytes", self.metrics.total_retrieved as f64);
        histogram!("vetras.ipfs.retrieval_time", retrieval_time);
        counter!("vetras.ipfs.models_retrieved", 1);
        
        Ok(cache_path)
    }

    pub async fn pin_content(&mut self, cid: &str) -> Result<()> {
        tokio::time::timeout(
            PIN_TIMEOUT,
            self.client.pin_add(cid, false)
        ).await??;
        
        self.metrics.pin_count += 1;
        counter!("vetras.ipfs.pins", 1);
        Ok(())
    }

    pub async fn unpin_content(&mut self, cid: &str) -> Result<()> {
        self.client.pin_rm(cid, false).await?;
        self.metrics.pin_count = self.metrics.pin_count.saturating_sub(1);
        counter!("vetras.ipfs.unpins", 1);
        Ok(())
    }

    pub async fn get_multiaddrs(&self, cid: &str) -> Result<Vec<String>> {
        let mut addrs = Vec::new();
        
        let providers = self.client.dht_findprovs(cid).await?;
        for provider in providers {
            for addr in provider.addrs {
                addrs.push(addr.to_string());
            }
        }
        
        Ok(addrs)
    }

    pub async fn clean_cache(&self) -> Result<()> {
        let mut entries = tokio::fs::read_dir(&self.local_cache).await?;
        let mut cleaned = 0u64;
        
        while let Some(entry) = entries.next_entry().await? {
            let metadata = entry.metadata().await?;
            if metadata.is_file() {
                tokio::fs::remove_file(entry.path()).await?;
                cleaned += metadata.len();
            }
        }
        
        counter!("vetras.ipfs.cache_cleaned_bytes", cleaned as f64);
        Ok(())
    }

    pub fn get_metrics(&self) -> &StorageMetrics {
        &self.metrics
    }

    fn compute_hash(&self, data: &[u8]) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(data);
        *hasher.finalize().as_bytes()
    }

    fn update_average_retrieval_time(&mut self, new_time: f64) {
        const ALPHA: f64 = 0.1; // Exponential moving average factor
        self.metrics.average_retrieval_time = 
            ALPHA * new_time + (1.0 - ALPHA) * self.metrics.average_retrieval_time;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use std::io::Write;

    async fn setup_test_storage() -> (IpfsStorage, TempDir) {
        let cache_dir = TempDir::new().unwrap();
        let storage = IpfsStorage::new(
            "http://localhost:5001",
            cache_dir.path()
        ).await.unwrap();
        
        (storage, cache_dir)
    }

    #[tokio::test]
    async fn test_store_and_retrieve() {
        let (mut storage, temp_dir) = setup_test_storage().await;
        
        // Create test file
        let test_file = temp_dir.path().join("test_model.bin");
        let test_data = vec![0u8; 1024];
        std::fs::File::create(&test_file)
            .unwrap()
            .write_all(&test_data)
            .unwrap();
            
        // Store model
        let location = storage.store_model(&test_file).await.unwrap();
        assert!(!location.cid.is_empty());
        assert_eq!(location.size, 1024);
        
        // Retrieve model
        let retrieved_path = storage.retrieve_model(&location).await.unwrap();
        assert!(retrieved_path.exists());
        
        // Verify content
        let retrieved_data = std::fs::read(retrieved_path).unwrap();
        assert_eq!(retrieved_data, test_data);
    }

    #[tokio::test]
    async fn test_cache_cleaning() {
        let (storage, _) = setup_test_storage().await;
        
        // Create some cache files
        for i in 0..5 {
            let path = storage.local_cache.join(format!("test_{}", i));
            std::fs::File::create(&path).unwrap().write_all(&[0u8; 100]).unwrap();
        }
        
        storage.clean_cache().await.unwrap();
        
        // Verify cache is empty
        let entries: Vec<_> = tokio::fs::read_dir(&storage.local_cache)
            .await
            .unwrap()
            .collect()
            .await;
            
        assert!(entries.is_empty());
    }

    #[tokio::test]
    #[should_panic]
    async fn test_size_limit() {
        let (mut storage, temp_dir) = setup_test_storage().await;
        
        // Create oversized file
        let large_file = temp_dir.path().join("large_model.bin");
        let large_data = vec![0u8; (MAX_MODEL_SIZE + 1) as usize];
        std::fs::File::create(&large_file)
            .unwrap()
            .write_all(&large_data)
            .unwrap();
            
        // This should fail
        storage.store_model(&large_file).await.unwrap();
    }
}
