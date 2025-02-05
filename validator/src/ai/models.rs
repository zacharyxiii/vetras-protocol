use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context, anyhow};
use sha2::{Sha256, Digest};
use chrono::{DateTime, Utc};
use async_trait::async_trait;
use metrics::{counter, gauge};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub id: String,
    pub name: String,
    pub version: String,
    pub framework: ModelFramework,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub hash: String,
    pub size_bytes: u64,
    pub parameters: u64,
    pub architecture: ModelArchitecture,
    pub input_specs: Vec<TensorSpec>,
    pub output_specs: Vec<TensorSpec>,
    pub compatibility: ModelCompatibility,
    pub validation_status: ValidationStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelFramework {
    ONNX,
    TensorFlow,
    PyTorch,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArchitecture {
    pub name: String,
    pub type_: ArchitectureType,
    pub layers: Vec<LayerInfo>,
    pub total_parameters: u64,
    pub trainable_parameters: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerInfo {
    pub name: String,
    pub type_: String,
    pub parameters: u64,
    pub input_shape: Vec<i64>,
    pub output_shape: Vec<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArchitectureType {
    CNN,
    RNN,
    Transformer,
    MLP,
    Hybrid,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSpec {
    pub name: String,
    pub shape: Vec<i64>,
    pub dtype: DataType,
    pub is_optional: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DataType {
    Float32,
    Float64,
    Int32,
    Int64,
    String,
    Bool,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCompatibility {
    pub minimum_runtime_version: String,
    pub supported_hardware: Vec<HardwareType>,
    pub supported_platforms: Vec<Platform>,
    pub memory_requirements: MemoryRequirements,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HardwareType {
    CPU,
    GPU,
    TPU,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Platform {
    Linux,
    Windows,
    MacOS,
    Android,
    IOS,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRequirements {
    pub minimum_ram: u64,
    pub recommended_ram: u64,
    pub gpu_memory: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ValidationStatus {
    Pending,
    InProgress,
    Valid,
    Invalid,
    Error,
}

pub struct ModelManager {
    models: Arc<RwLock<HashMap<String, ModelMetadata>>>,
    storage_path: PathBuf,
}

impl ModelManager {
    pub fn new<P: AsRef<Path>>(storage_path: P) -> Result<Self> {
        let storage_path = storage_path.as_ref().to_path_buf();
        tokio::fs::create_dir_all(&storage_path)?;
        
        Ok(Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            storage_path,
        })
    }

    pub async fn register_model<P: AsRef<Path>>(&self, path: P, name: String, version: String) -> Result<ModelMetadata> {
        let path = path.as_ref();
        let model_data = tokio::fs::read(path).await
            .context("Failed to read model file")?;
        
        let hash = self.compute_hash(&model_data);
        let size_bytes = model_data.len() as u64;
        
        let framework = self.detect_framework(&model_data)?;
        let architecture = self.analyze_architecture(&model_data, &framework)?;
        
        let model_path = self.storage_path.join(&hash);
        tokio::fs::copy(path, &model_path).await
            .context("Failed to copy model to storage")?;
            
        let metadata = ModelMetadata {
            id: hash.clone(),
            name,
            version,
            framework,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            hash,
            size_bytes,
            parameters: architecture.total_parameters,
            architecture,
            input_specs: self.analyze_input_specs(&model_data)?,
            output_specs: self.analyze_output_specs(&model_data)?,
            compatibility: self.analyze_compatibility(&model_data)?,
            validation_status: ValidationStatus::Pending,
        };
        
        self.models.write().await.insert(metadata.id.clone(), metadata.clone());
        counter!("s.models.registered", 1);
        
        Ok(metadata)
    }

    pub async fn get_model(&self, id: &str) -> Result<ModelMetadata> {
        self.models.read().await
            .get(id)
            .cloned()
            .ok_or_else(|| anyhow!("Model not found: {}", id))
    }

    pub async fn update_validation_status(&self, id: &str, status: ValidationStatus) -> Result<()> {
        let mut models = self.models.write().await;
        
        if let Some(model) = models.get_mut(id) {
            model.validation_status = status;
            model.updated_at = Utc::now();
            Ok(())
        } else {
            Err(anyhow!("Model not found: {}", id))
        }
    }

    pub async fn list_models(&self) -> Result<Vec<ModelMetadata>> {
        Ok(self.models.read().await.values().cloned().collect())
    }

    pub async fn delete_model(&self, id: &str) -> Result<()> {
        let mut models = self.models.write().await;
        
        if let Some(metadata) = models.remove(id) {
            let model_path = self.storage_path.join(&metadata.hash);
            tokio::fs::remove_file(model_path).await
                .context("Failed to delete model file")?;
            counter!("vetras.models.deleted", 1);
            Ok(())
        } else {
            Err(anyhow!("Model not found: {}", id))
        }
    }

    fn compute_hash(&self, data: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data);
        hex::encode(hasher.finalize())
    }

    fn detect_framework(&self, data: &[u8]) -> Result<ModelFramework> {
        // Check file signatures/magic numbers
        if data.len() < 8 {
            return Err(anyhow!("Model file too small"));
        }

        match &data[0..4] {
            b"ONNX" => Ok(ModelFramework::ONNX),
            b"PK\x03\x04" => Ok(ModelFramework::PyTorch), // ZIP signature for PyTorch
            _ if &data[0..2] == b"PK" => Ok(ModelFramework::TensorFlow), // Also ZIP for TF SavedModel
            _ => {
                // Additional framework detection logic
                if self.check_tensorflow_pb(data) {
                    Ok(ModelFramework::TensorFlow)
                } else {
                    Ok(ModelFramework::Custom("Unknown".to_string()))
                }
            }
        }
    }

    fn check_tensorflow_pb(&self, data: &[u8]) -> bool {
        // Check for TensorFlow protobuf format
        if data.len() < 12 {
            return false;
        }
        
        // Look for common TF graph signatures
        data.windows(10).any(|window| {
            window.starts_with(b"GraphDef") || 
            window.starts_with(b"SavedModel")
        })
    }

    fn analyze_architecture(&self, data: &[u8], framework: &ModelFramework) -> Result<ModelArchitecture> {
        match framework {
            ModelFramework::ONNX => self.analyze_onnx_architecture(data),
            ModelFramework::TensorFlow => self.analyze_tensorflow_architecture(data),
            ModelFramework::PyTorch => self.analyze_pytorch_architecture(data),
            ModelFramework::Custom(_) => self.analyze_custom_architecture(data),
        }
    }

    fn analyze_onnx_architecture(&self, data: &[u8]) -> Result<ModelArchitecture> {
        let model = tract_onnx::onnx()
            .model_for_read(&mut std::io::Cursor::new(data))
            .context("Failed to parse ONNX model")?;

        let mut layers = Vec::new();
        let mut total_parameters = 0;
        let mut trainable_parameters = 0;

        for node in model.graph.nodes.iter() {
            let mut layer_params = 0;
            for init in model.graph.initializers.iter() {
                if init.name == node.name {
                    layer_params = init.dims.iter().product::<i64>() as u64;
                    break;
                }
            }

            layers.push(LayerInfo {
                name: node.name.clone(),
                type_: node.op_type.clone(),
                parameters: layer_params,
                input_shape: node.input.iter()
                    .filter_map(|i| self.get_tensor_shape(&model.graph, i))
                    .next()
                    .unwrap_or_default(),
                output_shape: node.output.iter()
                    .filter_map(|o| self.get_tensor_shape(&model.graph, o))
                    .next()
                    .unwrap_or_default(),
            });

            total_parameters += layer_params;
            trainable_parameters += layer_params;
        }

        Ok(ModelArchitecture {
            name: "ONNX Model".to_string(),
            type_: self.detect_architecture_type(&layers),
            layers,
            total_parameters,
            trainable_parameters,
        })
    }

    fn get_tensor_shape(&self, graph: &tract_onnx::prelude::Graph, name: &str) -> Option<Vec<i64>> {
        graph.inputs.iter()
            .chain(graph.outputs.iter())
            .find(|t| t.name == name)
            .map(|t| t.shape.to_vec())
    }

    fn detect_architecture_type(&self, layers: &[LayerInfo]) -> ArchitectureType {
        let layer_types: Vec<_> = layers.iter()
            .map(|l| l.type_.to_lowercase())
            .collect();

        if layer_types.iter().any(|t| t.contains("conv")) {
            ArchitectureType::CNN
        } else if layer_types.iter().any(|t| t.contains("lstm") || t.contains("gru")) {
            ArchitectureType::RNN
        } else if layer_types.iter().any(|t| t.contains("attention") || t.contains("transformer")) {
            ArchitectureType::Transformer
        } else if layer_types.iter().all(|t| t.contains("dense") || t.contains("linear")) {
            ArchitectureType::MLP
        } else {
            ArchitectureType::Hybrid
        }
    }

    fn analyze_input_specs(&self, data: &[u8]) -> Result<Vec<TensorSpec>> {
        let model = tract_onnx::onnx()
            .model_for_read(&mut std::io::Cursor::new(data))
            .context("Failed to parse model for input specs")?;

        Ok(model.graph.inputs.iter()
            .map(|input| TensorSpec {
                name: input.name.clone(),
                shape: input.shape.to_vec(),
                dtype: self.convert_tract_type(&input.datum_type),
                is_optional: false,
            })
            .collect())
    }

    fn analyze_output_specs(&self, data: &[u8]) -> Result<Vec<TensorSpec>> {
        let model = tract_onnx::onnx()
            .model_for_read(&mut std::io::Cursor::new(data))
            .context("Failed to parse model for output specs")?;

        Ok(model.graph.outputs.iter()
            .map(|output| TensorSpec {
                name: output.name.clone(),
                shape: output.shape.to_vec(),
                dtype: self.convert_tract_type(&output.datum_type),
                is_optional: false,
            })
            .collect())
    }

    fn convert_tract_type(&self, dtype: &tract_onnx::prelude::DatumType) -> DataType {
        match dtype {
            tract_onnx::prelude::DatumType::F32 => DataType::Float32,
            tract_onnx::prelude::DatumType::F64 => DataType::Float64,
            tract_onnx::prelude::DatumType::I32 => DataType::Int32,
            tract_onnx::prelude::DatumType::I64 => DataType::Int64,
            tract_onnx::prelude::DatumType::String => DataType::String,
            tract_onnx::prelude::DatumType::Bool => DataType::Bool,
            _ => DataType::Custom(format!("{:?}", dtype)),
        }
    }

    fn analyze_compatibility(&self, data: &[u8]) -> Result<ModelCompatibility> {
        let size = data.len() as u64;
        
        Ok(ModelCompatibility {
            minimum_runtime_version: "1.0.0".to_string(),
            supported_hardware: vec![
                HardwareType::CPU,
                if size > 1_000_000_000 { // 1GB
                    HardwareType::GPU
                } else {
                    HardwareType::CPU
                }
            ],
            supported_platforms: vec![
                Platform::Linux,
                Platform::Windows,
                Platform::MacOS,
            ],
            memory_requirements: MemoryRequirements {
                minimum_ram: size * 2,
                recommended_ram: size * 4,
                gpu_memory: if size > 1_000_000_000 {
                    Some(size)
                } else {
                    None
                },
            },
        })
    }

    fn analyze_tensorflow_architecture(&self, _data: &[u8]) -> Result<ModelArchitecture> {
        // Similar structure to ONNX analysis but for TensorFlow models
        // Placeholder for TensorFlow implementation
        Ok(ModelArchitecture {
            name: "TensorFlow Model".to_string(),
            type_: ArchitectureType::Custom("TensorFlow".to_string()),
            layers: Vec::new(),
            total_parameters: 0,
            trainable_parameters: 0,
        })
    }

    fn analyze_pytorch_architecture(&self, _data: &[u8]) -> Result<ModelArchitecture> {
        // Similar structure to ONNX analysis but for PyTorch models
        // Placeholder for PyTorch implementation
        Ok(ModelArchitecture {
            name: "PyTorch Model".to_string(),
            type_: ArchitectureType::Custom("PyTorch".to_string()),
            layers: Vec::new(),
            total_parameters: 0,
            trainable_parameters: 0,
        })
    }

    fn analyze_custom_architecture(&self, _data: &[u8]) -> Result<ModelArchitecture> {
        Ok(ModelArchitecture {
            name: "Custom Model".to_string(),
            type_: ArchitectureType::Custom("Unknown".to_string()),
            layers: Vec::new(),
            total_parameters: 0,
            trainable_parameters: 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use std::io::Write;

    fn create_dummy_onnx_model() -> Vec<u8> {
        // Create a minimal ONNX model for testing
        let mut data = Vec::new();
        data.extend_from_slice(b"ONNX");
        data.extend_from_slice(&[0; 1000]); // Dummy content
        data
    }

    #[tokio::test]
    async fn test_model_registration() {
        let temp_dir = TempDir::new().unwrap();
        let manager = ModelManager::new(temp_dir.path()).unwrap();

        // Create a temporary model file
        let model_path = temp_dir.path().join("test_model.onnx");
        let model_data = create_dummy_onnx_model();
        std::fs::File::create(&model_path)
            .unwrap()
            .write_all(&model_data)
            .unwrap();

        let metadata = manager.register_model(
            model_path,
            "Test Model".to_string(),
            "1.0.0".to_string()
        ).await.unwrap();

        assert_eq!(metadata.name, "Test Model");
        assert_eq!(metadata.version, "1.0.0");
        assert_eq!(metadata.framework, ModelFramework::ONNX);
        assert_eq!(metadata.validation_status, ValidationStatus::Pending);
    }

    #[tokio::test]
    async fn test_model_lifecycle() {
        let temp_dir = TempDir::new().unwrap();
        let manager = ModelManager::new(temp_dir.path()).unwrap();

        // Register model
        let model_path = temp_dir.path().join("lifecycle_test.onnx");
        let model_data = create_dummy_onnx_model();
        std::fs::File::create(&model_path)
            .unwrap()
            .write_all(&model_data)
            .unwrap();

        let metadata = manager.register_model(
            model_path,
            "Lifecycle Test".to_string(),
            "1.0.0".to_string()
        ).await.unwrap();

        // Get model
        let retrieved = manager.get_model(&metadata.id).await.unwrap();
        assert_eq!(retrieved.id, metadata.id);

        // Update status
        manager.update_validation_status(&metadata.id, ValidationStatus::Valid).await.unwrap();
        let updated = manager.get_model(&metadata.id).await.unwrap();
        assert_eq!(updated.validation_status, ValidationStatus::Valid);

        // List models
        let models = manager.list_models().await.unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, metadata.id);

        // Delete model
        manager.delete_model(&metadata.id).await.unwrap();
        assert!(manager.get_model(&metadata.id).await.is_err());
    }

    #[tokio::test]
    async fn test_framework_detection() {
        let temp_dir = TempDir::new().unwrap();
        let manager = ModelManager::new(temp_dir.path()).unwrap();

        let onnx_data = create_dummy_onnx_model();
        assert_eq!(
            manager.detect_framework(&onnx_data).unwrap(),
            ModelFramework::ONNX
        );

        // Test invalid data
        let invalid_data = vec![0; 4];
        assert!(matches!(
            manager.detect_framework(&invalid_data).unwrap(),
            ModelFramework::Custom(_)
        ));
    }
}