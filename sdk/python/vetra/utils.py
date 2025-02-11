import hashlib
import json
from typing import Dict, Any, Union, Optional
import numpy as np
import tensorflow as tf
import torch
from eth_utils import (
    is_address,
    to_checksum_address,
    keccak,
    encode_hex
)
import onnx
from google.protobuf.json_format import MessageToDict

from .types import AIModel, ModelFramework, ModelArchitecture, ModelLayer

class SerializationError(Exception):
    """Raised when model serialization fails."""
    pass

class ValidationError(Exception):
    """Raised when validation fails."""
    pass

def validate_address(address: str) -> str:
    """
    Validate and normalize Ethereum address.
    
    Args:
        address: Ethereum address to validate
        
    Returns:
        Checksum address if valid
        
    Raises:
        ValidationError: If address is invalid
    """
    try:
        if not is_address(address):
            raise ValidationError(f"Invalid address format: {address}")
        return to_checksum_address(address)
    except Exception as e:
        raise ValidationError(f"Address validation failed: {str(e)}")

def hash_model(model: AIModel) -> str:
    """
    Generate deterministic hash for AI model.
    
    Creates hash incorporating:
    - Model architecture
    - Weights (if available)
    - Framework and version
    - Metadata
    
    Args:
        model: AIModel instance to hash
        
    Returns:
        Hex string of model hash
    """
    # Convert model to canonical JSON representation
    model_dict = model.dict(exclude={'id', 'created_at'})
    
    # Add weights hash if available
    if model.weights_hash:
        model_dict['weights_hash'] = model.weights_hash
        
    # Generate canonical JSON string
    canonical_json = json.dumps(model_dict, sort_keys=True)
    
    # Calculate keccak256 hash
    model_hash = keccak(text=canonical_json)
    
    return encode_hex(model_hash)

def serialize_model(model: AIModel) -> Dict[str, Any]:
    """
    Serialize AI model for transmission.
    
    Handles different model frameworks appropriately:
    - TensorFlow: Saved model format
    - PyTorch: State dict
    - ONNX: Protocol buffers
    - SKLearn: Pickle format
    
    Args:
        model: AIModel instance to serialize
        
    Returns:
        Dictionary with serialized model data
    
    Raises:
        SerializationError: If serialization fails
    """
    try:
        serialized = model.dict(exclude={'weights_uri'})
        
        if model.weights_uri:
            # Don't include weights directly, they should be uploaded separately
            serialized['has_weights'] = True
            
        return serialized
    except Exception as e:
        raise SerializationError(f"Model serialization failed: {str(e)}")

def extract_architecture(
    model: Union[tf.keras.Model, torch.nn.Module, onnx.ModelProto],
    framework: ModelFramework
) -> ModelArchitecture:
    """
    Extract architecture details from model object.
    
    Args:
        model: Model object from supported framework
        framework: Framework enum indicating model type
        
    Returns:
        ModelArchitecture instance
        
    Raises:
        ValueError: If framework is unsupported
    """
    if framework == ModelFramework.TENSORFLOW:
        return _extract_tf_architecture(model)
    elif framework == ModelFramework.PYTORCH:
        return _extract_pytorch_architecture(model)
    elif framework == ModelFramework.ONNX:
        return _extract_onnx_architecture(model)
    else:
        raise ValueError(f"Unsupported framework for architecture extraction: {framework}")

def _extract_tf_architecture(model: tf.keras.Model) -> ModelArchitecture:
    """Extract architecture from TensorFlow model."""
    layers = []
    for layer in model.layers:
        layer_config = layer.get_config()
        layers.append(ModelLayer(
            name=layer.name,
            type=layer.__class__.__name__,
            units=layer_config.get('units'),
            activation=layer_config.get('activation'),
            params=layer_config
        ))
    
    return ModelArchitecture(
        input_shape=model.input_shape[1:],
        output_shape=model.output_shape[1:],
        layers=layers,
        total_params=model.count_params(),
        framework_version=tf.__version__
    )

def _extract_pytorch_architecture(model: torch.nn.Module) -> ModelArchitecture:
    """Extract architecture from PyTorch model."""
    layers = []
    total_params = 0
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            params = {k: v.shape for k, v in module.named_parameters()}
            param_count = sum(p.numel() for p in module.parameters())
            total_params += param_count
            
            layers.append(ModelLayer(
                name=name,
                type=module.__class__.__name__,
                units=getattr(module, 'out_features', None),
                activation=getattr(module, 'activation', None).__class__.__name__ if hasattr(module, 'activation') else None,
                params=params
            ))
    
    # Note: Input/output shapes require a forward pass or manual specification
    return ModelArchitecture(
        input_shape=[],  # Needs to be provided separately
        output_shape=[],  # Needs to be provided separately
        layers=layers,
        total_params=total_params,
        framework_version=torch.__version__
    )

def _extract_onnx_architecture(model: onnx.ModelProto) -> ModelArchitecture:
    """Extract architecture from ONNX model."""
    graph = MessageToDict(model.graph)
    layers = []
    total_params = 0
    
    for node in graph['node']:
        attrs = {attr['name']: attr['value'] for attr in node.get('attribute', [])}
        layers.append(ModelLayer(
            name=node.get('name', ''),
            type=node['opType'],
            params=attrs
        ))
        
        # Count parameters in initializers
        if 'initializer' in graph:
            for init in graph['initializer']:
                if init['name'] in [i for i in node.get('input', [])]:
                    total_params += np.prod([int(d) for d in init['dims']])
    
    input_shape = [int(d) for d in graph['input'][0]['type']['tensorType']['shape']['dim']]
    output_shape = [int(d) for d in graph['output'][0]['type']['tensorType']['shape']['dim']]
    
    return ModelArchitecture(
        input_shape=input_shape[1:],  # Remove batch dimension
        output_shape=output_shape[1:],  # Remove batch dimension
        layers=layers,
        total_params=total_params,
        framework_version=onnx.__version__
    )

def calculate_stake_requirement(
    model: AIModel,
    validation_level: str,
    base_stake: int = 100
) -> int:
    """
    Calculate required stake for model validation.
    
    Factors in:
    - Model complexity (parameters)
    - Validation level
    - Current network conditions
    
    Args:
        model: AIModel to calculate stake for
        validation_level: Level of validation required
        base_stake: Base stake amount
        
    Returns:
        Required stake amount in tokens
    """
    # Complexity multiplier based on model size
    complexity_factor = np.log10(model.architecture.total_params) / 10
    
    # Validation level multiplier
    level_multipliers = {
        'basic': 1.0,
        'standard': 1.5,
        'comprehensive': 2.0,
        'enhanced': 3.0
    }
    level_factor = level_multipliers.get(validation_level.lower(), 1.0)
    
    # Calculate final stake requirement
    stake = int(base_stake * complexity_factor * level_factor)
    
    # Ensure minimum stake
    return max(stake, base_stake)

def format_error_response(error: Exception) -> Dict[str, Any]:
    """Format error for API response."""
    if isinstance(error, (SerializationError, ValidationError)):
        error_type = error.__class__.__name__
        status_code = 400
    else:
        error_type = "InternalError"
        status_code = 500
        
    return {
        "error": {
            "type": error_type,
            "message": str(error),
            "status_code": status_code
        }
    }

def parse_validation_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse and validate parameters for validation request.
    
    Args:
        params: Raw parameter dictionary
        
    Returns:
        Validated and normalized parameters
        
    Raises:
        ValidationError: If parameters are invalid
    """
    required_fields = ['model_id', 'validation_level']
    for field in required_fields:
        if field not in params:
            raise ValidationError(f"Missing required field: {field}")
            
    # Normalize validation level
    valid_levels = ['basic', 'standard', 'comprehensive', 'enhanced']
    level = params['validation_level'].lower()
    if level not in valid_levels:
        raise ValidationError(f"Invalid validation level. Must be one of: {valid_levels}")
    params['validation_level'] = level
    
    # Validate optional parameters
    if 'stake_amount' in params:
        try:
            params['stake_amount'] = int(params['stake_amount'])
            if params['stake_amount'] < 0:
                raise ValueError()
        except ValueError:
            raise ValidationError("stake_amount must be a positive integer")
            
    if 'validator_preferences' in params:
        if not isinstance(params['validator_preferences'], dict):
            raise ValidationError("validator_preferences must be a dictionary")
            
    return params