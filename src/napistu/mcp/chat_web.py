"""
Chat utilities for web interface - rate limiting, cost tracking, and Claude client.

This module supports the REST API endpoints for the landing page chat interface.
Similar to client.py which provides MCP client utilities, this provides chat utilities.
"""

import asyncio
import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import anthropic
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from napistu.mcp.constants import (
    CHAT_DEFAULTS,
    CHAT_ENV_VARS,
    CHAT_SYSTEM_PROMPT,
    MCP_DEFAULTS,
    MCP_PRODUCTION_URL,
)

logger = logging.getLogger(__name__)


class ChatConfig(BaseModel):
    """
    Configuration for chat web interface with validation

    Public Methods
    --------------
    get_mcp_url()
        Get formatted MCP URL with /mcp/ path

    Private Methods
    --------------
    validate_api_key(v)
        Validate Anthropic API key is present and valid format
    validate_mcp_url(v)
        Validate MCP server URL format
    validate_rate_limits()
        Validate rate limits are sensible
    """

    model_config = {"frozen": True}  # Make immutable

    # Rate limits per IP
    rate_limit_per_hour: int = Field(
        default_factory=lambda: int(
            os.getenv(
                CHAT_ENV_VARS.RATE_LIMIT_PER_HOUR, CHAT_DEFAULTS.RATE_LIMIT_PER_HOUR
            )
        )
    )
    rate_limit_per_day: int = Field(
        default_factory=lambda: int(
            os.getenv(
                CHAT_ENV_VARS.RATE_LIMIT_PER_DAY, CHAT_DEFAULTS.RATE_LIMIT_PER_DAY
            )
        )
    )

    # Cost controls
    daily_budget: float = Field(
        default_factory=lambda: float(
            os.getenv(CHAT_ENV_VARS.DAILY_BUDGET, CHAT_DEFAULTS.DAILY_BUDGET)
        ),
        gt=0,
        description="Daily budget in USD, must be positive",
    )
    max_tokens: int = Field(
        default_factory=lambda: int(
            os.getenv(CHAT_ENV_VARS.MAX_TOKENS, CHAT_DEFAULTS.MAX_TOKENS)
        ),
        gt=0,
        le=200000,
    )
    max_message_length: int = Field(
        default_factory=lambda: int(
            os.getenv(
                CHAT_ENV_VARS.MAX_MESSAGE_LENGTH, CHAT_DEFAULTS.MAX_MESSAGE_LENGTH
            )
        ),
        gt=0,
    )

    # API configuration
    anthropic_api_key: str = Field(
        default_factory=lambda: os.getenv(CHAT_ENV_VARS.ANTHROPIC_API_KEY, ""),
        min_length=1,
        description="Anthropic API key (required)",
    )
    claude_model: str = Field(
        default_factory=lambda: os.getenv(
            CHAT_ENV_VARS.CLAUDE_MODEL, CHAT_DEFAULTS.CLAUDE_MODEL
        )
    )
    mcp_server_url: str = Field(
        default_factory=lambda: os.getenv(
            CHAT_ENV_VARS.MCP_SERVER_URL, MCP_PRODUCTION_URL
        )
    )

    def get_mcp_url(self) -> str:
        """Get formatted MCP URL with /mcp/ path"""
        base_url = self.mcp_server_url.rstrip("/")
        if not base_url.endswith(MCP_DEFAULTS.MCP_PATH):
            base_url = base_url + MCP_DEFAULTS.MCP_PATH
        return base_url + "/"

    @field_validator("anthropic_api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate Anthropic API key is present and valid format"""
        if not v:
            raise ValueError(
                f"{CHAT_ENV_VARS.ANTHROPIC_API_KEY} environment variable must be set"
            )
        if not v.startswith("sk-ant-"):
            raise ValueError(
                f"{CHAT_ENV_VARS.ANTHROPIC_API_KEY} must start with 'sk-ant-'"
            )
        if len(v) < 20:
            raise ValueError(
                f"{CHAT_ENV_VARS.ANTHROPIC_API_KEY} appears too short to be valid"
            )
        logger.info(f"✅ Anthropic API key validated (length: {len(v)})")
        return v

    @field_validator("mcp_server_url")
    @classmethod
    def validate_mcp_url(cls, v: str) -> str:
        """Validate MCP server URL format"""
        if not v.startswith(("http://", "https://")):
            raise ValueError("MCP_SERVER_URL must start with http:// or https://")
        logger.info(f"✅ MCP server URL: {v}")
        return v

    @model_validator(mode="after")
    def validate_rate_limits(self) -> "ChatConfig":
        """Validate rate limits are sensible"""
        if self.rate_limit_per_day < self.rate_limit_per_hour:
            raise ValueError("Daily rate limit must be >= hourly rate limit")
        return self


class RateLimiter:
    """
    In-memory rate limiter for IP-based throttling

    Public Methods
    --------------
    check_limit(ip)
        Check if IP has exceeded rate limits
    get_remaining_requests(ip)
        Get remaining requests for an IP address
    record_request(ip)
        Record a request for rate limiting

    Private Methods
    --------------
    _clean_old_timestamps(timestamps, cutoff)
        Remove timestamps older than cutoff
    _get_counts(ip)
        Get current request counts for an IP, cleaning old timestamps
    """

    def __init__(self):
        self.chat_config = get_chat_config()
        self.store: Dict[str, Dict[str, List[datetime]]] = defaultdict(
            lambda: {"hour": [], "day": []}
        )

    def check_limit(self, ip: str) -> Tuple[bool, str]:
        """Check if IP has exceeded rate limits"""
        hour_count, day_count = self._get_counts(ip)

        if hour_count >= self.chat_config.rate_limit_per_hour:
            return False, (
                f"Hourly limit exceeded ({self.chat_config.rate_limit_per_hour} "
                "messages/hour). Please try again later."
            )

        if day_count >= self.chat_config.rate_limit_per_day:
            return False, (
                f"Daily limit exceeded ({self.chat_config.rate_limit_per_day} "
                "messages/day). Please try again tomorrow."
            )

        return True, ""

    def get_remaining_requests(self, ip: str) -> Dict[str, int]:
        """
        Get remaining requests for an IP address.

        Cleans old timestamps and returns both hourly and daily remaining counts.

        Parameters
        ----------
        ip : str
            Client IP address

        Returns
        -------
        Dict[str, int]
            Dictionary with 'per_hour' and 'per_day' remaining counts
        """
        hour_count, day_count = self._get_counts(ip)

        return {
            "per_hour": self.chat_config.rate_limit_per_hour - hour_count,
            "per_day": self.chat_config.rate_limit_per_day - day_count,
        }

    def record_request(self, ip: str) -> None:
        """Record a request for rate limiting"""
        now = datetime.now()
        self.store[ip]["hour"].append(now)
        self.store[ip]["day"].append(now)

    def _clean_old_timestamps(
        self, timestamps: List[datetime], cutoff: datetime
    ) -> List[datetime]:
        """Remove timestamps older than cutoff"""
        return [ts for ts in timestamps if ts > cutoff]

    def _get_counts(self, ip: str) -> Tuple[int, int]:
        """
        Get current request counts for an IP, cleaning old timestamps.

        Parameters
        ----------
        ip : str
            Client IP address

        Returns
        -------
        Tuple[int, int]
            (hour_count, day_count) - number of requests in last hour and day
        """
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)

        # Clean old timestamps
        self.store[ip]["hour"] = self._clean_old_timestamps(
            self.store[ip]["hour"], hour_ago
        )
        self.store[ip]["day"] = self._clean_old_timestamps(
            self.store[ip]["day"], day_ago
        )

        # Count remaining
        return len(self.store[ip]["hour"]), len(self.store[ip]["day"])


class CostTracker:
    """
    Track daily API costs

    Public Methods
    --------------
    check_budget()
        Check if daily budget has been exceeded
    estimate_cost(usage)
        Estimate cost based on token usage
    get_stats()
        Get current cost stats
    record_cost(usage)
        Record estimated cost
    """

    # Claude Sonnet 4.5 pricing (as of Dec 2024)
    INPUT_COST_PER_MILLION = 3.0
    OUTPUT_COST_PER_MILLION = 15.0

    def __init__(self):
        self.chat_config = get_chat_config()
        self.date: Optional[str] = None
        self.cost: float = 0.0

    def check_budget(self) -> bool:
        """Check if daily budget has been exceeded"""
        today = datetime.now().date().isoformat()

        if self.date != today:
            self.date = today
            self.cost = 0.0

        return self.cost < self.chat_config.daily_budget

    def estimate_cost(self, usage: Dict[str, int]) -> float:
        """Estimate cost based on token usage"""
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        input_cost = (input_tokens / 1_000_000) * self.INPUT_COST_PER_MILLION
        output_cost = (output_tokens / 1_000_000) * self.OUTPUT_COST_PER_MILLION

        return input_cost + output_cost

    def get_stats(self) -> Dict[str, float]:
        """Get current cost stats"""
        today = datetime.now().date().isoformat()

        if self.date != today:
            cost_today = 0.0
        else:
            cost_today = self.cost

        return {
            "cost_today": round(cost_today, 2),
            "budget_remaining": round(self.chat_config.daily_budget - cost_today, 2),
        }

    def record_cost(self, usage: Dict[str, int]) -> None:
        """Record estimated cost"""
        cost = self.estimate_cost(usage)
        self.cost += cost
        logger.info(f"Request cost: ${cost:.4f}, total today: ${self.cost:.4f}")


class ClaudeClient:
    """
    Client for Claude API with MCP integration

    Attributes
    ----------
    chat_config : ChatConfig
        Chat configuration
    client : anthropic.Anthropic
        Anthropic client

    Public Methods
    --------------
    chat(user_message)
        Send a message to Claude with MCP tools
    """

    def __init__(self):
        self.chat_config = get_chat_config()
        self.client = anthropic.Anthropic(api_key=self.chat_config.anthropic_api_key)

    async def chat(self, user_message: str) -> Dict:
        """
        Send a message to Claude with MCP tools.

        Parameters
        ----------
        user_message : str
            User's question

        Returns
        -------
        Dict with 'response' (str) and 'usage' (dict)
        """
        logger.info(f"💬 Starting chat for message: {user_message[:50]}...")

        # Get MCP server URL from config
        mcp_url = self.chat_config.get_mcp_url()
        logger.info("📡 Calling Anthropic API ...")
        logger.critical(f"🔥 MCP URL → Anthropic: {repr(mcp_url)}")
        start_time = time.time()

        # Use asyncio.wait_for to enforce timeout
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.client.beta.messages.create,
                    model=self.chat_config.claude_model,
                    max_tokens=self.chat_config.max_tokens,
                    system=CHAT_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_message}],
                    mcp_servers=[
                        {"type": "url", "url": mcp_url, "name": "napistu-mcp"}
                    ],
                    extra_headers={"anthropic-beta": "mcp-client-2025-04-04"},
                ),
                timeout=60.0,  # 60 second timeout
            )
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.error(f"❌ Anthropic API call TIMED OUT after {elapsed:.2f}s")
            raise RuntimeError(f"Claude API timed out after {elapsed:.2f}s")

        elapsed = time.time() - start_time
        logger.info(f"✅ Anthropic API responded in {elapsed:.2f}s")

        # Extract response text
        response_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                response_text += block.text

        logger.info(f"📝 Response length: {len(response_text)} chars")

        # Track usage
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

        logger.info(
            f"📊 Token usage: {usage['input_tokens']} in / {usage['output_tokens']} out"
        )

        return {
            "response": response_text,
            "usage": usage,
        }


# Global instances (module-level singletons)


_chat_config: Optional[ChatConfig] = None
_claude_client: Optional[ClaudeClient] = None
_rate_limiter: Optional[RateLimiter] = None
_cost_tracker: Optional[CostTracker] = None


# Public gettr and convenience functions


def get_chat_config() -> ChatConfig:
    """Get the chat configuration singleton with detailed error logging"""
    global _chat_config
    if _chat_config is None:
        try:
            logger.info("Initializing ChatConfig...")
            _chat_config = ChatConfig()
            logger.info("✅ Chat configuration validated successfully")
            logger.info(f"   Model: {_chat_config.claude_model}")
            logger.info(f"   MCP URL: {_chat_config.get_mcp_url()}")
            logger.info(f"   Max tokens: {_chat_config.max_tokens}")
            logger.info(
                f"   Rate limits: {_chat_config.rate_limit_per_hour}/hr, {_chat_config.rate_limit_per_day}/day"
            )
        except ValidationError as e:
            logger.error("❌ ChatConfig validation failed with Pydantic errors:")
            for error in e.errors():
                field = error["loc"][0] if error["loc"] else "unknown"
                msg = error["msg"]
                logger.error(f"   Field '{field}': {msg}")
            raise
        except Exception as e:
            logger.error(
                f"❌ ChatConfig initialization failed: {type(e).__name__}: {e}"
            )
            import traceback

            logger.error(traceback.format_exc())
            raise
    return _chat_config


def get_claude_client() -> ClaudeClient:
    """Get or create Claude client singleton with error handling"""
    global _claude_client
    if _claude_client is None:
        try:
            logger.info("Initializing ClaudeClient...")
            _claude_client = ClaudeClient()
            logger.info("✅ Claude client initialized successfully")
        except Exception as e:
            logger.error(
                f"❌ ClaudeClient initialization failed: {type(e).__name__}: {e}"
            )
            import traceback

            logger.error(traceback.format_exc())
            raise
    return _claude_client


def get_rate_limiter() -> RateLimiter:
    """Get or create rate limiter singleton"""
    global _rate_limiter
    if _rate_limiter is None:
        try:
            logger.info("Initializing RateLimiter...")
            _rate_limiter = RateLimiter()
            logger.info("✅ Rate limiter initialized successfully")
        except Exception as e:
            logger.error(
                f"❌ RateLimiter initialization failed: {type(e).__name__}: {e}"
            )
            import traceback

            logger.error(traceback.format_exc())
            raise
    return _rate_limiter


def get_cost_tracker() -> CostTracker:
    """Get or create cost tracker singleton"""
    global _cost_tracker
    if _cost_tracker is None:
        try:
            logger.info("Initializing CostTracker...")
            _cost_tracker = CostTracker()
            logger.info("✅ Cost tracker initialized successfully")
        except Exception as e:
            logger.error(
                f"❌ CostTracker initialization failed: {type(e).__name__}: {e}"
            )
            import traceback

            logger.error(traceback.format_exc())
            raise
    return _cost_tracker
