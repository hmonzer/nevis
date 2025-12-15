from src.app.core.services.document_search_service import DocumentSearchService


class EngineAdapter:
    def __init__(self, document_search_service: DocumentSearchService):
        self.document_search_service = document_search_service

    async def search(self, query_text: str) -> list[str]:
        """
        Searches for documents and returns a list of document IDs.
        """
        search_results = await self.document_search_service.search(query=query_text)
        return [result.document.id for result in search_results]
