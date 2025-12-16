"""Search API endpoints for unified search across clients and documents."""
from typing import Annotated
from fastapi import APIRouter, Depends, Query
from dependency_injector.wiring import Provide, inject

from src.app.containers import Container
from src.app.core.domain.models import SearchRequest
from src.app.core.services.search_service import SearchService
from src.client.schemas import SearchResultResponse
from src.app.api.mappers import to_search_result_response

router = APIRouter(prefix="/search", tags=["search"])


@router.get("/", response_model=list[SearchResultResponse])
@inject
async def search(
    q: Annotated[str, Query(min_length=1, description="Search query string")],
    top_k: Annotated[int, Query(gt=0, le=100, description="Maximum number of results")] = 3,
    threshold: Annotated[float, Query(ge=-1.0, le=1.0, description="Minimum similarity threshold")] = 0.5,
    service: SearchService = Depends(Provide[Container.search_service])
) -> list[SearchResultResponse]:
    """
    Search across clients and documents using hybrid search.

    This endpoint performs a unified search across all clients and documents,
    combining vector similarity search with keyword search for improved results.

    Args:
        q: The search query string
        top_k: Maximum number of results to return (default: 10, max: 100)
        threshold: Minimum similarity threshold for results (default: 0.5)

    Returns:
        List of search results containing matched clients and documents,
        ranked by relevance score
    """
    request = SearchRequest(query=q, top_k=top_k, threshold=threshold)
    results = await service.search(request)

    return [to_search_result_response(result) for result in results]
