"""
Chat utilities for web interface - rate limiting, cost tracking, and Claude client.

This module supports the REST API endpoints for the landing page chat interface.
Similar to client.py which provides MCP client utilities, this provides chat utilities.
"""

import logging
import os
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import anthropic
from pydantic import BaseModel, Field, field_validator, model_validator

from napistu.mcp.constants import (
    CHAT_DEFAULTS,
    CHAT_ENV_VARS,
    CHAT_SYSTEM_PROMPT,
    MCP_DEFAULTS,
    MCP_PRODUCTION_URL,
)

logger = logging.getLogger(__name__)


class ChatConfig(BaseModel):
    """Configuration for chat web interface with validation"""

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

    def get_mcp_url(self) -> str:
        """Get formatted MCP URL with /mcp/ path"""
        base_url = self.mcp_server_url.rstrip("/")
        if not base_url.endswith(MCP_DEFAULTS.MCP_PATH):
            base_url = base_url + MCP_DEFAULTS.MCP_PATH
        return base_url + "/"


class RateLimiter:
    """In-memory rate limiter for IP-based throttling"""

    def __init__(self):
        self.chat_config = get_chat_config()
        self.store: Dict[str, Dict[str, List[datetime]]] = defaultdict(
            lambda: {"hour": [], "day": []}
        )

    def _clean_old_timestamps(
        self, timestamps: List[datetime], cutoff: datetime
    ) -> List[datetime]:
        """Remove timestamps older than cutoff"""
        return [ts for ts in timestamps if ts > cutoff]

    def check_limit(self, ip: str) -> Tuple[bool, str]:
        """Check if IP has exceeded rate limits"""
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

        # Check limits
        hour_count = len(self.store[ip]["hour"])
        day_count = len(self.store[ip]["day"])

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

    def record_request(self, ip: str) -> None:
        """Record a request for rate limiting"""
        now = datetime.now()
        self.store[ip]["hour"].append(now)
        self.store[ip]["day"].append(now)


class CostTracker:
    """Track daily API costs"""

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

    def record_cost(self, usage: Dict[str, int]) -> None:
        """Record estimated cost"""
        cost = self.estimate_cost(usage)
        self.cost += cost
        logger.info(f"Request cost: ${cost:.4f}, total today: ${self.cost:.4f}")

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


class ClaudeClient:
    """Client for Claude API with MCP integration"""

    def __init__(self):
        self.chat_config = get_chat_config()
        self.client = anthropic.Anthropic(api_key=self.chat_config.anthropic_api_key)

    def chat(self, user_message: str) -> Dict:
        """
        Send a message to Claude with MCP tools.

        Args:
            user_message: User's question

        Returns:
            Dict with 'response' (str) and 'usage' (dict)
        """
        # Load the production url or local host for within server communication
        mcp_url = self.chat_config.get_mcp_url()

        response = self.client.beta.messages.create(
            model=self.chat_config.claude_model,
            max_tokens=self.chat_config.max_tokens,
            system=CHAT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            mcp_servers=[{"type": "url", "url": mcp_url, "name": "napistu-mcp"}],
            extra_headers={"anthropic-beta": "mcp-client-2025-04-04"},
        )

        # Extract text from response
        response_text = ""
        for block in response.content:
            if block.type == "text":
                response_text += block.text

        return {
            "response": response_text,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        }


# ============================================================================
# Global instances (module-level singletons)
# ============================================================================

_chat_config: Optional[ChatConfig] = None
_claude_client: Optional[ClaudeClient] = None


def get_chat_config() -> ChatConfig:
    """Get the chat configuration singleton"""
    global _chat_config
    if _chat_config is None:
        try:
            _chat_config = ChatConfig()
            logger.info("✅ Chat configuration validated successfully")
        except Exception as e:
            logger.error(f"❌ Chat configuration validation failed: {e}")
            raise
    return _chat_config


def get_claude_client() -> ClaudeClient:
    """Get or create Claude client singleton"""
    global _claude_client
    if _claude_client is None:
        _claude_client = ClaudeClient()
    return _claude_client


# Initialize after function definitions
rate_limiter = RateLimiter()
cost_tracker = CostTracker()
