"""
web_routes.py - Route handlers for Napistu chat web interface with CORS support
"""

import logging
from typing import Optional, Tuple

import httpx
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse
from starlette.routing import Route

from napistu.mcp.chat_web import (
    get_chat_config,
    get_claude_client,
    get_cost_tracker,
    get_rate_limiter,
)
from napistu.mcp.constants import DEFAULT_ALLOWED_ORIGINS, MCP_PRODUCTION_URL

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
    # debugging
    try:
        # Get client IP
        ip = request.client.host
        logger.info(f"Chat request from IP: {ip}")

        # Parse and validate request body
        message, error = await _parse_chat_message(request)
        if error:
            return error

        # Initialize all components (lazy init with fail-fast)
        success, error = _initialize_chat_components()
        if not success:
            return error

        # Get initialized components
        config = get_chat_config()
        rate_limiter = get_rate_limiter()
        cost_tracker = get_cost_tracker()
        client = get_claude_client()

        # Validate request against all constraints
        error = _validate_chat_request(message, ip, config, rate_limiter, cost_tracker)
        if error:
            return error

        # Call Claude with MCP tools
        logger.info(f"Calling Claude API for message: {message.content[:50]}...")
        try:
            result = await client.chat(message.content)
            logger.info(f"Claude API response received: {result['usage']}")
        except Exception as e:
            logger.error(f"Claude API call failed: {type(e).__name__}: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return JSONResponse(
                content={
                    "detail": "Failed to get response from Claude API",
                    "error": str(e),
                },
                status_code=500,
            )

        # Record usage
        rate_limiter.record_request(ip)
        cost_tracker.record_cost(result["usage"])
        logger.info(f"Request completed successfully for IP {ip}")

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Unexpected error in handle_chat: {type(e).__name__}: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return JSONResponse(
            content={
                "detail": f"Internal error: {str(e)}",
                "error_type": type(e).__name__,
            },
            status_code=500,
        )


async def handle_health(request: Request) -> JSONResponse:
    """Health check for chat API"""
    try:
        client = get_claude_client()
        api_configured = client.client is not None
        cost_tracker = get_cost_tracker()

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
        # Get client IP for rate limit calculations
        ip = request.client.host

        chat_config = get_chat_config()
        cost_tracker = get_cost_tracker()
        rate_limiter = get_rate_limiter()
        cost_stats = cost_tracker.get_stats()

        # Get remaining requests for this IP
        remaining = rate_limiter.get_remaining_requests(ip)

        stats = {
            "budget": {
                "daily_limit": chat_config.daily_budget,
                "spent_today": cost_stats["cost_today"],
                "remaining": cost_stats["budget_remaining"],
            },
            "rate_limits": remaining,  # Now just uses the method directly
        }

        logger.info(
            f"Stats for IP {ip}: {remaining['per_hour']} remaining/hour, "
            f"{remaining['per_day']} remaining/day"
        )

        return JSONResponse(content=stats)
    except Exception as e:
        return JSONResponse(
            content={"detail": f"Error getting stats: {str(e)}"},
            status_code=500,
        )


async def redirect_to_mcp(request: Request) -> RedirectResponse:
    """Redirect /mcp to /mcp/ for trailing slash compatibility"""
    return RedirectResponse(url="/mcp/", status_code=307)


# private utils


def _initialize_chat_components() -> Tuple[bool, Optional[JSONResponse]]:
    """
    Initialize all chat components with lazy loading and error handling.

    Returns
    -------
    Tuple[bool, Optional[JSONResponse]]
        (success, error_response)
    """
    try:
        _ = get_chat_config()
        _ = get_rate_limiter()
        _ = get_cost_tracker()
        _ = get_claude_client()
        return True, None
    except Exception as e:
        logger.error(f"Failed to initialize chat components: {e}")
        error_response = JSONResponse(
            content={
                "detail": "Chat service configuration error. Please check server logs.",
                "error_type": type(e).__name__,
            },
            status_code=503,
        )
        return False, error_response


async def _parse_chat_message(
    request: Request,
) -> Tuple[Optional[ChatMessage], Optional[JSONResponse]]:
    """
    Parse and validate chat message from request body.

    Returns
    -------
    Tuple[Optional[ChatMessage], Optional[JSONResponse]]
        (message, error_response) where one is always None
    """
    try:
        body = await request.json()
        message = ChatMessage(**body)
        return message, None
    except Exception as e:
        logger.error(f"Failed to parse request body: {e}")
        error_response = JSONResponse(
            content={"detail": f"Invalid request body: {str(e)}"},
            status_code=400,
        )
        return None, error_response


def _validate_chat_request(
    message: ChatMessage,
    ip: str,
    config,
    rate_limiter,
    cost_tracker,
) -> Optional[JSONResponse]:
    """
    Validate chat request against all constraints.

    Returns
    -------
    Optional[JSONResponse]
        Error response if validation fails, None if all checks pass
    """
    # Validate message length
    if not message.content or len(message.content) > config.max_message_length:
        logger.warning(
            f"Message length validation failed: {len(message.content)} chars"
        )
        return JSONResponse(
            content={
                "detail": f"Message must be between 1 and {config.max_message_length} characters."
            },
            status_code=400,
        )

    # Check rate limits
    is_allowed, error_msg = rate_limiter.check_limit(ip)
    if not is_allowed:
        logger.warning(f"Rate limit exceeded for IP {ip}: {error_msg}")
        return JSONResponse(
            content={"detail": error_msg},
            status_code=429,
        )

    # Check daily budget
    if not cost_tracker.check_budget():
        logger.warning("Daily budget exceeded")
        return JSONResponse(
            content={
                "detail": "Daily budget exceeded. Service will be available again tomorrow."
            },
            status_code=503,
        )

    return None
