"""
web_routes.py - Route handlers for Napistu chat web interface with CORS support
"""

import logging

from mcp.server import FastMCP
from pydantic import BaseModel, ValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse

from napistu.mcp.chat_web import (
    cost_tracker,
    get_claude_client,
    rate_limiter,
)
from napistu.mcp.constants import DEFAULT_ALLOWED_ORIGINS

logger = logging.getLogger(__name__)


def get_cors_headers(origin: str = None):
    """
    Get CORS headers for response.

    Only allows requests from specified origins for security.
    Returns appropriate CORS headers if origin is in allowed list.
    """
    # Check if origin is in allowed list
    if origin and origin in DEFAULT_ALLOWED_ORIGINS:
        allowed_origin = origin
    else:
        # Default to first allowed origin if no origin header or not in list
        allowed_origin = DEFAULT_ALLOWED_ORIGINS[0]

    return {
        "Access-Control-Allow-Origin": allowed_origin,
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Max-Age": "3600",
    }


class ChatMessage(BaseModel):
    content: str


async def handle_chat(request: Request) -> JSONResponse:
    """Handle chat requests with rate limiting and cost tracking"""

    # Handle preflight OPTIONS request
    if request.method == "OPTIONS":
        return JSONResponse(
            content={}, headers=get_cors_headers(request.headers.get("origin"))
        )

    try:
        # Get client IP
        ip = request.client.host

        # Parse request body
        body = await request.json()
        message = ChatMessage(**body)

        # Validate message length
        if not message.content or len(message.content) > 2000:
            return JSONResponse(
                content={"detail": "Message must be between 1 and 2000 characters."},
                status_code=400,
                headers=get_cors_headers(request.headers.get("origin")),
            )

        # Check rate limits
        is_allowed, error_msg = rate_limiter.check_rate_limit(ip)
        if not is_allowed:
            return JSONResponse(
                content={"detail": error_msg},
                status_code=429,
                headers=get_cors_headers(request.headers.get("origin")),
            )

        # Check daily budget
        if not cost_tracker.check_budget():
            return JSONResponse(
                content={
                    "detail": "Daily budget exceeded. Service will be available again tomorrow."
                },
                status_code=503,
                headers=get_cors_headers(request.headers.get("origin")),
            )

        # Call Claude with MCP tools
        client = get_claude_client()
        result = await client.call_claude_with_mcp(message.content)

        # Record usage
        rate_limiter.record_request(ip)
        cost_tracker.record_cost(result["usage"])

        return JSONResponse(
            content=result, headers=get_cors_headers(request.headers.get("origin"))
        )

    except ValidationError as e:
        return JSONResponse(
            content={"detail": str(e)},
            status_code=400,
            headers=get_cors_headers(request.headers.get("origin")),
        )
    except Exception as e:
        return JSONResponse(
            content={"detail": f"Internal error: {str(e)}"},
            status_code=500,
            headers=get_cors_headers(request.headers.get("origin")),
        )


async def handle_stats(request: Request) -> JSONResponse:
    """Get current usage stats"""

    # Handle preflight OPTIONS request
    if request.method == "OPTIONS":
        return JSONResponse(
            content={}, headers=get_cors_headers(request.headers.get("origin"))
        )

    try:
        # Import ChatConfig to get budget
        from napistu.mcp.chat_web import ChatConfig

        # Get cost stats
        cost_stats = cost_tracker.get_stats()

        # Build response
        stats = {
            "budget": {
                "daily_limit": ChatConfig.DAILY_BUDGET,
                "spent_today": cost_stats["cost_today"],
                "remaining": cost_stats["budget_remaining"],
            },
            "rate_limits": {
                "per_hour": ChatConfig.RATE_LIMIT_PER_HOUR,
                "per_day": ChatConfig.RATE_LIMIT_PER_DAY,
            },
        }

        return JSONResponse(
            content=stats, headers=get_cors_headers(request.headers.get("origin"))
        )
    except Exception as e:
        return JSONResponse(
            content={"detail": f"Error getting stats: {str(e)}"},
            status_code=500,
            headers=get_cors_headers(request.headers.get("origin")),
        )


async def handle_health(request: Request) -> JSONResponse:
    """Health check for chat API"""

    # Handle preflight OPTIONS request
    if request.method == "OPTIONS":
        return JSONResponse(
            content={}, headers=get_cors_headers(request.headers.get("origin"))
        )

    try:
        client = get_claude_client()
        api_configured = client.client is not None

        return JSONResponse(
            content={
                "status": "healthy",
                "chat_api": "configured" if api_configured else "not_configured",
                "budget_ok": cost_tracker.check_budget(),
            },
            headers=get_cors_headers(request.headers.get("origin")),
        )
    except Exception as e:
        return JSONResponse(
            content={"status": "unhealthy", "error": str(e)},
            status_code=500,
            headers=get_cors_headers(request.headers.get("origin")),
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
