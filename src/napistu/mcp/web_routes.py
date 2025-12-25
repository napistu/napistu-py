"""
web_routes.py - Route handlers for Napistu chat web interface with CORS support
"""

import logging

import httpx
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse
from starlette.routing import Route

from napistu.mcp.chat_web import (
    cost_tracker,
    get_chat_config,
    get_claude_client,
    rate_limiter,
)
from napistu.mcp.constants import DEFAULT_ALLOWED_ORIGINS

logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    content: str


def create_chat_app() -> Starlette:
    """
    Create a Starlette app for chat routes with CORS middleware.

    This app is completely separate from the MCP server and only handles
    the /api/* chat endpoints. CORS is only applied to these routes.

    Returns
    -------
    Starlette
        Starlette app with chat routes and CORS middleware
    """

    chat_app = Starlette(
        routes=[
            Route("/api/chat", endpoint=handle_chat, methods=["POST", "OPTIONS"]),
            Route("/api/stats", endpoint=handle_stats, methods=["GET", "OPTIONS"]),
            Route("/api/health", endpoint=handle_health, methods=["GET", "OPTIONS"]),
            Route("/api/test-mcp", endpoint=handle_mcp_test, methods=["GET"]),
        ]
    )

    # Add CORS middleware ONLY to this app
    chat_app.add_middleware(
        CORSMiddleware,
        allow_origins=DEFAULT_ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    logger.info("Created chat API with CORS middleware at /api/*")

    return chat_app


async def handle_chat(request: Request) -> JSONResponse:
    """Handle chat requests with rate limiting and cost tracking"""
    try:
        # Get client IP
        ip = request.client.host

        # Parse request body
        body = await request.json()
        message = ChatMessage(**body)

        # Get config for validation
        config = get_chat_config()

        # Validate message length using config
        if not message.content or len(message.content) > config.max_message_length:
            return JSONResponse(
                content={
                    "detail": f"Message must be between 1 and {config.max_message_length} characters."
                },
                status_code=400,
            )

        # Check rate limits
        is_allowed, error_msg = rate_limiter.check_limit(ip)
        if not is_allowed:
            return JSONResponse(
                content={"detail": error_msg},
                status_code=429,
            )

        # Check daily budget
        if not cost_tracker.check_budget():
            return JSONResponse(
                content={
                    "detail": "Daily budget exceeded. Service will be available again tomorrow."
                },
                status_code=503,
            )

        # Call Claude with MCP tools
        client = get_claude_client()
        result = client.chat(message.content)

        # Record usage
        rate_limiter.record_request(ip)
        cost_tracker.record_cost(result["usage"])

        return JSONResponse(content=result)

    except ValidationError as e:
        return JSONResponse(
            content={"detail": str(e)},
            status_code=400,
        )
    except Exception as e:
        return JSONResponse(
            content={"detail": f"Internal error: {str(e)}"},
            status_code=500,
        )


async def handle_health(request: Request) -> JSONResponse:
    """Health check for chat API"""
    try:
        client = get_claude_client()
        api_configured = client.client is not None

        return JSONResponse(
            content={
                "status": "healthy",
                "chat_api": "configured" if api_configured else "not_configured",
                "budget_ok": cost_tracker.check_budget(),
            },
        )
    except Exception as e:
        return JSONResponse(
            content={"status": "unhealthy", "error": str(e)},
            status_code=500,
        )


async def handle_mcp_test(request: Request) -> JSONResponse:
    """Test if MCP server is reachable from this container"""
    from napistu.mcp.constants import MCP_PRODUCTION_URL

    mcp_url = MCP_PRODUCTION_URL + "/mcp/"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                mcp_url,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "test", "version": "1.0"},
                    },
                },
            )
            return JSONResponse(
                content={
                    "status": "reachable",
                    "code": response.status_code,
                    "can_reach_external_mcp": True,
                    "url_tested": mcp_url,
                }
            )
    except Exception as e:
        return JSONResponse(
            content={
                "status": "unreachable",
                "error": str(e),
                "can_reach_external_mcp": False,
                "url_tested": mcp_url,
            },
            status_code=500,
        )


async def handle_stats(request: Request) -> JSONResponse:
    """Get current usage stats"""
    try:
        chat_config = get_chat_config()
        cost_stats = cost_tracker.get_stats()

        stats = {
            "budget": {
                "daily_limit": chat_config.daily_budget,
                "spent_today": cost_stats["cost_today"],
                "remaining": cost_stats["budget_remaining"],
            },
            "rate_limits": {
                "per_hour": chat_config.rate_limit_per_hour,
                "per_day": chat_config.rate_limit_per_day,
            },
        }

        return JSONResponse(content=stats)
    except Exception as e:
        return JSONResponse(
            content={"detail": f"Error getting stats: {str(e)}"},
            status_code=500,
        )


async def redirect_to_mcp(request: Request) -> RedirectResponse:
    """Redirect /mcp to /mcp/ for trailing slash compatibility"""
    return RedirectResponse(url="/mcp/", status_code=307)
