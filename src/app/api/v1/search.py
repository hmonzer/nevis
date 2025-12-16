"""Search API endpoints for unified search across clients and documents."""
from typing import Annotated
from fastapi import APIRouter, Depends, Query
from dependency_injector.wiring import Provide, inject

from src.app.containers import Container
from src.app.config import Settings
from src.app.core.domain.models import SearchRequest
from src.app.core.services.search_service import SearchService
from src.client.schemas import SearchResultResponse
from src.app.api.mappers import to_search_result_response

router = APIRouter(prefix="/search", tags=["search"])


@router.get("/", response_model=list[SearchResultResponse])
@inject
async def search(
    q: Annotated[str, Query(min_length=1, description="Search query string")],
    top_k: Annotated[int | None, Query(gt=0, le=100, description="Maximum number of results")] = None,
    service: SearchService = Depends(Provide[Container.search_service]),
    config: Settings = Depends(Provide[Container.config]),
) -> list[SearchResultResponse]:
    """
    Search across clients and documents using hybrid search.

    This endpoint performs a unified search across all clients and documents,
    combining vector similarity search with keyword search for improved results.
    Similarity thresholds are configured per-component in application settings.

    Args:
        q: The search query string
        top_k: Maximum number of results to return (default from config, max: 100)

    Returns:
        List of search results containing matched clients and documents,
        ranked by relevance score
    """
    # Use config default if top_k not specified
    effective_top_k = top_k if top_k is not None else config.search.default_top_k

    request = SearchRequest(query=q, top_k=effective_top_k)
    results = await service.search(request)

    return [to_search_result_response(result) for result in results]
