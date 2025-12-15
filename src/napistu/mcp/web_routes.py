"""
Web routes for chat interface using FastMCP's custom_route system.

Provides REST API endpoints that complement the MCP protocol.
These routes are registered directly on the FastMCP server.
"""

import logging
from typing import Dict

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from mcp.server import FastMCP
from pydantic import BaseModel

from napistu.mcp.chat_web import (
    ChatConfig,
    cost_tracker,
    get_claude_client,
    rate_limiter,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Models
# ============================================================================


class ChatMessage(BaseModel):
    """Request model for chat messages"""

    content: str


class ChatResponse(BaseModel):
    """Response model for chat messages"""

    response: str
    usage: Dict[str, int]


class ChatStats(BaseModel):
    """Stats model for monitoring"""

    daily_budget: float
    cost_today: float
    budget_remaining: float
    rate_limits: Dict[str, int]


# ============================================================================
# Route handlers (called by custom_route decorators)
# ============================================================================


async def handle_chat(request: Request) -> JSONResponse:
    """
    Handle chat requests.

    Rate limits:
    - 5 messages per hour per IP
    - 15 messages per day per IP

    Daily budget: $5
    """
    # Get client IP
    ip = request.client.host

    # Parse request body
    try:
        body = await request.json()
        message = ChatMessage(**body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")

    # Validate message
    if not message.content or not message.content.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    if len(message.content) > ChatConfig.MAX_MESSAGE_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Message must be under {ChatConfig.MAX_MESSAGE_LENGTH} characters",
        )

    # Check rate limits
    is_allowed, error_msg = rate_limiter.check_limit(ip)
    if not is_allowed:
        raise HTTPException(status_code=429, detail=error_msg)

    # Check daily budget
    if not cost_tracker.check_budget():
        raise HTTPException(
            status_code=503,
            detail="Daily budget exceeded. Service will be available tomorrow.",
        )

    # Call Claude
    try:
        claude_client = get_claude_client()
        result = claude_client.chat(message.content)

        # Record usage
        rate_limiter.record_request(ip)
        cost_tracker.record_cost(result["usage"])

        return JSONResponse(ChatResponse(**result).model_dump())

    except ValueError as e:
        # API key not configured
        logger.error(f"Chat API not configured: {e}")
        raise HTTPException(status_code=503, detail="Chat service not available")
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


async def handle_stats(request: Request) -> JSONResponse:
    """Get current usage statistics"""
    stats = cost_tracker.get_stats()
    return JSONResponse(
        ChatStats(
            daily_budget=ChatConfig.DAILY_BUDGET,
            cost_today=stats["cost_today"],
            budget_remaining=stats["budget_remaining"],
            rate_limits={
                "per_hour": ChatConfig.RATE_LIMIT_PER_HOUR,
                "per_day": ChatConfig.RATE_LIMIT_PER_DAY,
            },
        ).model_dump()
    )


async def handle_health(request: Request) -> JSONResponse:
    """Health check for chat API"""
    try:
        # Check if API key is configured
        get_claude_client()
        api_configured = True
    except ValueError:
        api_configured = False

    return JSONResponse(
        {
            "status": "healthy" if api_configured else "unavailable",
            "chat_api": "configured" if api_configured else "not_configured",
            "budget_ok": cost_tracker.check_budget(),
        }
    )


# functions


def enable_chat_web_interface(mcp: FastMCP) -> None:
    """
    Enable chat web interface with REST endpoints.

    Registers three REST endpoints for the landing page chat interface:
    - POST /api/chat - Main chat endpoint with rate limiting
    - GET /api/stats - Usage statistics and budget tracking
    - GET /api/health - Health check and API key validation

    Parameters
    ----------
    mcp : FastMCP
        FastMCP server instance to register endpoints on

    Examples
    --------
    >>> mcp = FastMCP("napistu-docs", host="0.0.0.0", port=8080)
    >>> enable_chat_web_interface(mcp)
    """
    logger.info("Enabling chat web interface")

    # Register endpoints using FastMCP's custom_route decorator
    @mcp.custom_route("/api/chat", methods=["POST"])
    async def chat_route(request):
        return await handle_chat(request)

    @mcp.custom_route("/api/stats", methods=["GET"])
    async def stats_route(request):
        return await handle_stats(request)

    @mcp.custom_route("/api/health", methods=["GET"])
    async def health_route(request):
        return await handle_health(request)

    logger.info("Registered chat endpoints at /api/*")
