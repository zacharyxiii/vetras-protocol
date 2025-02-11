from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator
import numpy as np

class ModelFramework(str, Enum):
    """Supported AI model frameworks."""
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    ONNX = "onnx"
    SKLEARN = "sklearn"
    CUSTOM = "custom"

class ValidationState(str, Enum):
    """Possible states of validation process."""
    PENDING = "pending"
    PROCESSING = "processing"
    CONSENSUS_BUILDING = "consensus_building"
    COMPLETED = "completed"
    FAILED = "failed"
    DISPUTED = "disputed"

class ValidatorState(str, Enum):
    """Possible states of validator nodes."""
    ACTIVE = "active"
    OFFLINE = "offline"
    SLASHED = "slashed"
    PENDING = "pending"
    JAILED = "jailed"

class ValidationLevel(str, Enum):
    """Validation thoroughness levels."""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    ENHANCED = "enhanced"

@dataclass
class ModelLayer:
    """Representation of a neural network layer."""
    name: str
    type: str
    units: Optional[int] = None
    activation: Optional[str] = None
    params: Dict = field(default_factory=dict)

@dataclass
class ModelArchitecture:
    """Neural network architecture description."""
    input_shape: List[int]
    output_shape: List[int]
    layers: List[ModelLayer]
    total_params: int
    framework_version: str

class AIModel(BaseModel):
    """
    Complete AI model representation including metadata,
    architecture, and optional weights.
    """
    id: Optional[str] = Field(None)
    name: str = Field(..., min_length=1, max_length=256)
    version: str = Field(..., regex=r"^\d+\.\d+\.\d+$")
    framework: ModelFramework
    description: Optional[str] = Field(None, max_length=1024)
    architecture: ModelArchitecture
    weights_uri: Optional[str] = None
    weights_hash: Optional[str] = None
    metadata: Dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        arbitrary_types_allowed = True

    @validator('weights_uri')
    def validate_weights_uri(cls, v):
        """Validate weights URI format."""
        if v and not (v.startswith('ipfs://') or v.startswith('https://')):
            raise ValueError('weights_uri must start with ipfs:// or https://')
        return v

class ValidationMetrics(BaseModel):
    """Metrics computed during model validation."""
    accuracy: float = Field(..., ge=0, le=1)
    precision: float = Field(..., ge=0, le=1)
    recall: float = Field(..., ge=0, le=1)
    f1_score: float = Field(..., ge=0, le=1)
    loss: float = Field(..., ge=0)
    inference_time: float = Field(..., gt=0)
    memory_usage: float = Field(..., gt=0)
    adversarial_robustness: Optional[float] = Field(None, ge=0, le=1)
    bias_metrics: Optional[Dict[str, float]] = None

class SecurityAudit(BaseModel):
    """Security audit results for validated model."""
    passed_basic_checks: bool
    vulnerabilities: List[Dict[str, str]] = Field(default_factory=list)
    privacy_score: float = Field(..., ge=0, le=1)
    attack_surface_rating: str = Field(..., regex=r'^(Low|Medium|High)$')
    mitigation_suggestions: List[str] = Field(default_factory=list)

class ConsensusDetails(BaseModel):
    """Details about the validation consensus process."""
    total_validators: int = Field(..., gt=0)
    agreeing_validators: int = Field(..., ge=0)
    consensus_threshold: float = Field(..., ge=0.5, le=1.0)
    consensus_reached: bool
    voting_power_distribution: Dict[str, float]
    rounds_required: int = Field(..., ge=1)

class ValidationRequest(BaseModel):
    """
    Initial validation request with configuration
    and tracking details.
    """
    request_id: str
    model_id: str
    model_hash: str
    submitter_address: str
    validation_level: ValidationLevel
    stake_amount: Optional[int] = None
    validator_preferences: Optional[Dict] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: ValidationState = Field(default=ValidationState.PENDING)

class ValidationStatus(BaseModel):
    """Current status of validation request."""
    request_id: str
    status: ValidationState
    current_stage: str
    progress: float = Field(..., ge=0, le=1)
    validators_assigned: List[str]
    estimated_completion_time: Optional[datetime] = None
    last_updated: datetime
    error_message: Optional[str] = None

class ValidationResult(BaseModel):
    """
    Complete validation results including metrics,
    security audit, and consensus details.
    """
    request_id: str
    model_id: str
    status: ValidationState
    metrics: ValidationMetrics
    security_audit: SecurityAudit
    consensus_details: ConsensusDetails
    blockchain_tx: str
    ipfs_result_hash: str
    completed_at: datetime
    total_validation_time: float
    stake_returned: Optional[int] = None
    rewards_earned: Optional[int] = None

class ValidatorProfile(BaseModel):
    """Validator node profile and capabilities."""
    hardware_specs: Dict[str, str]
    supported_frameworks: List[ModelFramework]
    max_model_size: int
    specializations: List[str] = Field(default_factory=list)
    performance_score: float = Field(..., ge=0, le=1)
    reputation_score: float = Field(..., ge=0, le=1)

class ValidatorNode(BaseModel):
    """Complete validator node information."""
    address: str
    state: ValidatorState
    total_stake: int
    self_stake: int
    delegated_stake: int
    profile: ValidatorProfile
    total_validations: int
    successful_validations: int
    last_active: datetime
    slashing_history: List[Dict] = Field(default_factory=list)
    earnings_history: List[Dict] = Field(default_factory=list)

class ModelMetrics(BaseModel):
    """
    Comprehensive metrics for a validated model including
    historical data and performance trends.
    """
    model_id: str
    latest_validation: ValidationResult
    historical_validations: List[ValidationResult]
    performance_trend: Dict[str, List[float]]
    security_score_history: List[float]
    validation_frequency: float
    average_validation_time: float
    total_stake_history: List[Dict[str, Union[datetime, int]]]
    validator_distribution: Dict[str, int]

class StakingInfo(BaseModel):
    """
    Staking information for an address including
    positions, rewards, and history.
    """
    address: str
    total_stake: int
    self_stake: int
    delegated_stake: Dict[str, int]
    delegators: Dict[str, int]
    unclaimed_rewards: int
    reward_rate: float
    total_rewards_earned: int
    last_claim: Optional[datetime] = None
    staking_history: List[Dict] = Field(default_factory=list)
    delegation_history: List[Dict] = Field(default_factory=list)
    
    @validator('reward_rate')
    def validate_reward_rate(cls, v):
        """Validate reward rate is a reasonable percentage."""
        if not 0 <= v <= 100:
            raise ValueError('reward_rate must be between 0 and 100')
        return v

class DatasetMetadata(BaseModel):
    """Metadata for datasets used in validation."""
    name: str
    version: str
    size: int
    num_samples: int
    features: List[str]
    target: str
    split_ratios: Dict[str, float]
    hash: str
    uri: str = Field(..., regex=r'^(ipfs://|https://)')

class ValidationConfig(BaseModel):
    """Configuration for validation process."""
    validation_level: ValidationLevel
    metrics_to_compute: List[str]
    security_checks: List[str]
    timeout: int = Field(..., gt=0)
    min_validators: int = Field(..., gt=0)
    consensus_threshold: float = Field(..., ge=0.5, le=1.0)
    custom_parameters: Dict = Field(default_factory=dict)