import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from urllib.parse import urljoin
import aiohttp
import backoff
from eth_account import Account
from eth_account.messages import encode_defunct
from web3 import Web3

from .types import (
    AIModel,
    ValidationRequest,
    ValidationResult,
    ValidationStatus,
    ValidatorNode,
    ModelMetrics,
    StakingInfo,
)
from .utils import hash_model, serialize_model, validate_address

logger = logging.getLogger(__name__)

@dataclass
class ClientConfig:
    """Configuration for VETRAS client."""
    api_url: str
    web3_url: str
    contract_address: str
    private_key: Optional[str] = None
    request_timeout: int = 30
    max_retries: int = 3
    retry_delay: int = 1

class VetrasClient:
    """
    Client for interacting with the VETRAS AI model validation platform.
    
    Provides methods for:
    - Submitting AI models for validation
    - Tracking validation status
    - Managing validator stakes
    - Retrieving validation results and metrics
    """
    
    def __init__(self, config: ClientConfig):
        """Initialize VETRAS client with configuration."""
        self.config = config
        self.web3 = Web3(Web3.HTTPProvider(config.web3_url))
        
        if config.private_key:
            self.account = Account.from_key(config.private_key)
        else:
            self.account = None
            
        # Load ABI from package data
        with open("vetras/assets/contract_abi.json") as f:
            contract_abi = f.read()
        
        self.contract = self.web3.eth.contract(
            address=config.contract_address,
            abi=contract_abi
        )
        
        self._session = None
    
    async def __aenter__(self):
        """Set up async resources."""
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.request_timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up async resources."""
        if self._session:
            await self._session.close()
            
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3
    )
    async def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Dict:
        """Make HTTP request to VETRAS API with retries."""
        if not self._session:
            raise RuntimeError("Client must be used as async context manager")
            
        url = urljoin(self.config.api_url, endpoint)
        async with self._session.request(method, url, **kwargs) as response:
            response.raise_for_status()
            return await response.json()

    async def submit_model(
        self,
        model: AIModel,
        stake_amount: Optional[int] = None,
        validator_preferences: Optional[Dict] = None
    ) -> ValidationRequest:
        """
        Submit an AI model for validation.
        
        Args:
            model: AIModel object containing model data and metadata
            stake_amount: Optional amount to stake for priority validation
            validator_preferences: Optional preferences for validator selection
            
        Returns:
            ValidationRequest with request ID and initial status
        """
        # Calculate model hash
        model_hash = hash_model(model)
        
        # Prepare model data
        model_data = serialize_model(model)
        
        # Sign model hash if account available
        signature = None
        if self.account:
            message = encode_defunct(text=model_hash)
            signature = self.account.sign_message(message)
        
        # Prepare request payload
        payload = {
            "model_data": model_data,
            "model_hash": model_hash,
            "signature": signature.signature.hex() if signature else None,
            "stake_amount": stake_amount,
            "validator_preferences": validator_preferences
        }
        
        # Submit validation request
        result = await self._request(
            "POST",
            "/v1/validations",
            json=payload
        )
        
        return ValidationRequest(**result)
    
    async def get_validation_status(
        self,
        request_id: str
    ) -> ValidationStatus:
        """Get current status of a validation request."""
        result = await self._request(
            "GET",
            f"/v1/validations/{request_id}/status"
        )
        return ValidationStatus(**result)
    
    async def get_validation_result(
        self,
        request_id: str
    ) -> ValidationResult:
        """Get validation results once complete."""
        result = await self._request(
            "GET",
            f"/v1/validations/{request_id}/result"
        )
        return ValidationResult(**result)
        
    async def get_model_metrics(
        self,
        model_id: str
    ) -> ModelMetrics:
        """Get detailed metrics for a validated model."""
        result = await self._request(
            "GET", 
            f"/v1/models/{model_id}/metrics"
        )
        return ModelMetrics(**result)

    async def list_validators(
        self,
        active_only: bool = True
    ) -> List[ValidatorNode]:
        """Get list of validator nodes and their status."""
        result = await self._request(
            "GET",
            "/v1/validators",
            params={"active_only": active_only}
        )
        return [ValidatorNode(**v) for v in result]

    async def get_staking_info(
        self,
        address: Optional[str] = None
    ) -> StakingInfo:
        """
        Get staking information for an address.
        Uses authenticated user's address if none provided.
        """
        if address:
            validate_address(address)
        else:
            if not self.account:
                raise ValueError("No address provided and no account configured")
            address = self.account.address
            
        result = await self._request(
            "GET",
            f"/v1/staking/{address}"
        )
        return StakingInfo(**result)

    async def stake_tokens(
        self,
        amount: int,
        validator_address: Optional[str] = None
    ) -> Dict:
        """
        Stake tokens for validation priority or to become validator.
        
        Args:
            amount: Amount of tokens to stake
            validator_address: Optional validator address to delegate stake to
        """
        if not self.account:
            raise ValueError("Account required for staking")
            
        if validator_address:
            validate_address(validator_address)
            
        # Prepare transaction
        tx = self.contract.functions.stake(
            validator_address if validator_address else self.account.address,
            amount
        ).build_transaction({
            'from': self.account.address,
            'nonce': self.web3.eth.get_transaction_count(self.account.address),
        })
        
        # Sign transaction
        signed_tx = self.account.sign_transaction(tx)
        
        # Send transaction
        tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        # Wait for receipt
        receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
        
        return {
            "transaction_hash": receipt["transactionHash"].hex(),
            "block_number": receipt["blockNumber"],
            "status": receipt["status"]
        }

    async def unstake_tokens(
        self,
        amount: int
    ) -> Dict:
        """
        Unstake tokens from validation pool.
        
        Args:
            amount: Amount of tokens to unstake
        """
        if not self.account:
            raise ValueError("Account required for unstaking")
            
        # Prepare transaction  
        tx = self.contract.functions.unstake(amount).build_transaction({
            'from': self.account.address,
            'nonce': self.web3.eth.get_transaction_count(self.account.address),
        })
        
        # Sign transaction
        signed_tx = self.account.sign_transaction(tx)
        
        # Send transaction
        tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        # Wait for receipt
        receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
        
        return {
            "transaction_hash": receipt["transactionHash"].hex(),
            "block_number": receipt["blockNumber"],
            "status": receipt["status"]
        }

    async def claim_rewards(self) -> Dict:
        """Claim accumulated validation rewards."""
        if not self.account:
            raise ValueError("Account required for claiming rewards")
            
        # Prepare transaction
        tx = self.contract.functions.claimRewards().build_transaction({
            'from': self.account.address,
            'nonce': self.web3.eth.get_transaction_count(self.account.address),
        })
        
        # Sign transaction
        signed_tx = self.account.sign_transaction(tx)
        
        # Send transaction
        tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        # Wait for receipt
        receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
        
        return {
            "transaction_hash": receipt["transactionHash"].hex(),
            "block_number": receipt["blockNumber"],
            "status": receipt["status"],
            "rewards_amount": self.contract.functions.getClaimedAmount(
                receipt["transactionHash"].hex()
            ).call()
        }