import json
import asyncio
import uuid
from pathlib import Path
from ranx import Qrels, Run, evaluate

from evals.chunk_search.engine_adapter import EngineAdapter
from src.app.core.domain.models import Document, DocumentSearchResult, Client
from src.app.core.services.document_search_service import DocumentSearchService


class MockDocumentRepository:
    def __init__(self, documents):
        self._documents = {doc.id: doc for doc in documents}

    async def get_by_ids(self, ids: list[uuid.UUID]) -> list[Document]:
        return [self._documents[id] for id in ids if id in self._documents]


class MockDocumentChunkSearchService:
    async def search(self, query: str, top_k: int, similarity_threshold: float):
        # This is a mock and will not be used directly in this script
        return []


async def main():
    # 1. Load Data
    data_path = Path(__file__).parent / "data" / "synthetic_wealth_data.json"
    with open(data_path, "r") as f:
        test_suites = json.load(f)

    qrels_dict = {}
    run_dict = {}

    for suite in test_suites:
        for use_case in suite["use_cases"]:
            doc_id_map = {}
            corpus_docs = []
            clients = []
            client_id_map = {}

            for item in use_case["corpus"]:
                if "document_id" in item:
                    old_id = item["document_id"]
                    new_id = uuid.uuid4()
                    doc_id_map[old_id] = new_id
                    corpus_docs.append(
                        Document(
                            id=new_id,
                            client_id=uuid.uuid4(),  # Assign a random client_id for now
                            title=item["metadata"].get("title", "Default Title"),
                            s3_key=f"s3://bucket/{new_id}.pdf",
                        )
                    )
                elif "client_id" in item:
                    old_id = item["client_id"]
                    new_id = uuid.uuid4()
                    client_id_map[old_id] = new_id
                    # The client object has a description field, let's use the content for that.
                    clients.append(
                        Client(
                            id=new_id,
                            first_name=item["metadata"]["client_name"].split(" ")[0],
                            last_name=item["metadata"]["client_name"].split(" ")[1],
                            email=f"{item['metadata']['client_name'].replace(' ', '.')}@example.com",
                            description=item["content"]
                        )
                    )
                    # We also need to create a document for the client, as the search is document based
                    # and the expected results can contain client ids.
                    doc_id_map[old_id] = new_id
                    corpus_docs.append(
                        Document(
                            id=new_id,
                            client_id=new_id,
                            title=item["metadata"]["client_name"],
                            s3_key=f"s3://bucket/{new_id}.pdf",
                        )
                    )

            doc_repo = MockDocumentRepository(corpus_docs)
            chunk_search_service = MockDocumentChunkSearchService()
            
            # A mock search service that returns results based on the query
            class MockDocumentSearchService(DocumentSearchService):
                async def search(self, query: str, top_k: int = 10, threshold: float = 0.5) -> list[DocumentSearchResult]:
                    results = []
                    query_words = set(query.lower().split())
                    for doc in corpus_docs:
                        title_words = set(doc.title.lower().split())
                        if query_words.intersection(title_words):
                            results.append(DocumentSearchResult(document=doc, score=0.9))
                    return results

            doc_search_service = MockDocumentSearchService(chunk_search_service, doc_repo)
            engine = EngineAdapter(doc_search_service)

            for test in use_case["tests"]:
                query_id = test["query_id"]
                query_text = test["query_text"]

                # 2. Construct Qrels
                qrels_dict[query_id] = {}
                for i, doc_id in enumerate(test["expected_result_ids"]):
                    new_doc_id = str(doc_id_map[doc_id])
                    if i == 0:
                        score = 3  # Gold
                    elif i == 1:
                        score = 2  # Silver
                    else:
                        score = 1  # Bronze
                    qrels_dict[query_id][new_doc_id] = score

                # 3. Execute Search and Construct Run
                engine_results = await engine.search(query_text)
                run_dict[query_id] = {}
                for i, doc_id in enumerate(engine_results):
                    run_dict[query_id][str(doc_id)] = len(engine_results) - i

    # 4. Evaluation with Ranx
    qrels = Qrels(qrels_dict)
    run = Run(run_dict)

    results = evaluate(qrels, run, metrics=["mrr", "recall@5", "ndcg@5"])
    print(results)


if __name__ == "__main__":
    asyncio.run(main())