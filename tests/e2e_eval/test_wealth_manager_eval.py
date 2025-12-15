import pytest
import pytest_asyncio
from pathlib import Path
from ranx import Qrels, Run, evaluate
from sentence_transformers import SentenceTransformer

from src.app.core.services.chunks_search_service import DocumentChunkSearchService
from src.app.core.services.document_search_service import DocumentSearchService
from src.app.core.services.embedding import SentenceTransformerEmbedding
from src.app.infrastructure.document_search_repository import DocumentSearchRepository
from src.app.infrastructure.mappers.document_chunk_mapper import DocumentChunkMapper
from src.app.infrastructure.document_repository import DocumentRepository
from src.app.infrastructure.mappers.document_mapper import DocumentMapper

from tests.e2e_eval.data_setup import load_eval_suite, setup_eval_data

# --- Fixtures reusing conftest and dependencies ---

@pytest_asyncio.fixture(scope="module")
def sentence_transformer_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@pytest_asyncio.fixture(scope="module")
def embedding_service(sentence_transformer_model):
    return SentenceTransformerEmbedding(sentence_transformer_model)

@pytest_asyncio.fixture
def search_repository(clean_database):
    return DocumentSearchRepository(clean_database, DocumentChunkMapper())

@pytest_asyncio.fixture
def chunk_search_service(embedding_service, search_repository):
    return DocumentChunkSearchService(embedding_service=embedding_service, search_repository=search_repository)

@pytest_asyncio.fixture
def document_repository(clean_database):
    return DocumentRepository(clean_database, DocumentMapper())

@pytest_asyncio.fixture
def document_search_service(chunk_search_service, document_repository):
    return DocumentSearchService(chunk_search_service=chunk_search_service, document_repository=document_repository)


@pytest.mark.asyncio
async def test_wealth_manager_eval(
    nevis_client, # Used for data setup via API
    document_search_service, # Used for running search directly
    clean_database, # Ensures fresh DB
    s3_storage # Ensures S3
):
    # 1. Load the Evaluation Suite
    data_path = Path(__file__).parent / "data" / "synthetic_wealth_data.json"
    suite = load_eval_suite(data_path)
    
    print(f"\nStarting Evaluation Suite: {suite.suite_name}")

    for use_case in suite.use_cases:
        print(f"\nRunning Use Case: {use_case.title}")
        
        # 2. Setup Data (Ingestion via API)
        id_map = await setup_eval_data(use_case, nevis_client)
        
        # 3. Run Tests
        qrels_dict = {}
        run_dict = {}
        
        for test in use_case.tests:
            # Construct Qrels (Ground Truth)
            qrels_dict[test.query_id] = {}
            for rank, exp_id in enumerate(test.expected_result_ids):
                if exp_id in id_map:
                    real_id = str(id_map[exp_id])
                    # Score depends on length of expected results (first is highest)
                    score = len(test.expected_result_ids) - rank
                    qrels_dict[test.query_id][real_id] = score
                else:
                    # If an expected ID isn't in our map, it might be that the synthetic data 
                    # referenced a non-existent ID or an ID that failed to create.
                    print(f"Warning: Expected ID {exp_id} not found in creation map.")

            # Execute Search
            # Note: We use the actual query text from the test case
            top_k = 10
            results = await document_search_service.search(test.query_text, top_k=top_k)
            
            # Construct Run (System Output)
            run_dict[test.query_id] = {}
            for rank, result in enumerate(results):
                # Score decreases with rank based on top_k
                run_score = max(top_k - rank, 1)
                run_dict[test.query_id][str(result.document.id)] = run_score

        # 4. Evaluate using Ranx
        if qrels_dict and run_dict:
            qrels = Qrels(qrels_dict)
            run = Run(run_dict)
            
            metrics = evaluate(qrels, run, metrics=["mrr", "recall@5", "ndcg@5"])
            print(f"Metrics for {use_case.title}:")
            print(metrics)
            
            # Optional: Basic assertions to ensure we aren't completely failing
            # We expect at least some recall given the synthetic data is designed to match
            assert metrics["recall@5"] > 0, f"Recall@5 should be > 0 for {use_case.title}"
        else:
            print("No valid qrels or run data to evaluate.")