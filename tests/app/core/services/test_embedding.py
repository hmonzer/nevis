"""Tests for embedding service."""
import pytest
from sentence_transformers import SentenceTransformer

from src.app.core.services.embedding import SentenceTransformerEmbedding


@pytest.fixture
def sentence_transformer_model():
    """Create a SentenceTransformer model instance."""
    return SentenceTransformer("all-MiniLM-L6-v2")


@pytest.fixture
def embedding_service(sentence_transformer_model):
    """Create an embedding service instance with injected model."""
    return SentenceTransformerEmbedding(sentence_transformer_model)


@pytest.mark.asyncio
async def test_embed_single_text(embedding_service):
    """Test embedding a single text."""
    text = "This is a test sentence for embedding."
    result = await embedding_service.embed_document(text)

    assert result.text == text
    assert isinstance(result.embedding, list)
    assert len(result.embedding) == 384
    assert all(isinstance(val, float) for val in result.embedding)


@pytest.mark.asyncio
async def test_embed_text_empty_raises_error(embedding_service):
    """Test that embedding empty text raises ValueError."""
    with pytest.raises(ValueError, match="Text cannot be empty"):
        await embedding_service.embed_document("")

    with pytest.raises(ValueError, match="Text cannot be empty"):
        await embedding_service.embed_document("   ")


@pytest.mark.asyncio
async def test_embed_batch(embedding_service):
    """Test embedding multiple texts in a batch."""
    texts = [
        "First test sentence.",
        "Second test sentence.",
        "Third test sentence."
    ]
    results = await embedding_service.embed_document_batch(texts)

    assert isinstance(results, list)
    assert len(results) == 3

    for i, result in enumerate(results):
        assert result.text == texts[i]
        assert len(result.embedding) == 384
        assert all(isinstance(val, float) for val in result.embedding)


@pytest.mark.asyncio
async def test_embed_batch_empty_list_raises_error(embedding_service):
    """Test that embedding empty list raises ValueError."""
    with pytest.raises(ValueError, match="Texts list cannot be empty"):
        await embedding_service.embed_document_batch([])


@pytest.mark.asyncio
async def test_embed_batch_with_empty_text_raises_error(embedding_service):
    """Test that embedding batch with empty text raises ValueError."""
    texts = ["Valid text", "", "Another valid text"]
    with pytest.raises(ValueError, match="All texts must be non-empty"):
        await embedding_service.embed_document_batch(texts)


@pytest.mark.asyncio
async def test_embed_text_similarity_address_proof(embedding_service):
    """
    Test semantic similarity for document search use case.

    Validates that searching for 'address proof' returns documents
    containing utility bills (which serve as address proof).
    """
    import numpy as np

    # Mock utility bill content (contains address information)
    utility_bill = """
    ELECTRICITY BILL
    Account Number: 1234567890
    Service Address: 123 Main Street, Apartment 4B, New York, NY 10001
    Billing Period: January 1 - January 31, 2024
    Amount Due: $125.50
    Due Date: February 15, 2024
    """

    # Search query
    query = "I want a proof of address for my customer"

    driving_license_chunk = """
        DRIVER'S LICENSE
        License Number: D1234567
        Name: Jane Doe
        Date of Birth: 01/15/1985
        Expiration Date: 01/15/2028
        Class: C - Passenger Vehicle
        Restrictions: Corrective Lenses Required
        """

    # Generate embeddings
    bill_result = await embedding_service.embed_document(utility_bill)
    query_result = await embedding_service.embed_query(query)
    unrelated_result = await embedding_service.embed_document(driving_license_chunk)

    # Convert to numpy arrays
    bill_emb = np.array(bill_result.embedding)
    query_emb = np.array(query_result.embedding)
    unrelated_emb = np.array(unrelated_result.embedding)

    # Compute cosine similarities
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    sim_query_bill = cosine_similarity(query_emb, bill_emb)
    sim_query_unrelated = cosine_similarity(query_emb, unrelated_emb)

    # Utility bill should be more similar to "address proof" than unrelated text
    assert sim_query_bill > sim_query_unrelated
    # Utility bill should have reasonable similarity to "address proof" query
    assert sim_query_bill > 0.3  # Semantic similarity threshold


@pytest.mark.asyncio
async def test_embed_batch_consistency(embedding_service):
    """Test that batch embedding produces same results as individual embeddings."""
    import numpy as np

    texts = ["First sentence", "Second sentence"]

    # Embed individually
    result1_single = await embedding_service.embed_document(texts[0])
    result2_single = await embedding_service.embed_document(texts[1])

    # Embed as batch
    results_batch = await embedding_service.embed_document_batch(texts)

    # Results should be very close (allowing for minor floating point differences)
    np.testing.assert_allclose(result1_single.embedding, results_batch[0].embedding, rtol=1e-5)
    np.testing.assert_allclose(result2_single.embedding, results_batch[1].embedding, rtol=1e-5)

    # Verify text is preserved correctly
    assert results_batch[0].text == texts[0]
    assert results_batch[1].text == texts[1]


@pytest.mark.asyncio
async def test_embed_batch_semantic_search(embedding_service):
    """
    Test batch embedding for semantic document search.

    Validates that searching for 'address proof' correctly ranks:
    1. Utility bill (relevant - contains address) - highest similarity
    2. Driving license (less relevant - has address but not proof of address)
    """
    import numpy as np

    # Document chunks
    utility_bill_chunk = """
    WATER UTILITY BILL
    Customer Name: John Smith
    Service Address: 456 Oak Avenue, Suite 12, Boston, MA 02101
    Billing Period: December 2023
    Total Amount: $45.30
    Please pay by January 20, 2024
    """

    driving_license_chunk = """
    DRIVER'S LICENSE
    License Number: D1234567
    Name: Jane Doe
    Date of Birth: 01/15/1985
    Expiration Date: 01/15/2028
    Class: C - Passenger Vehicle
    Restrictions: Corrective Lenses Required
    """

    # Search query
    query = "I want a proof of address for my customer"

    # Batch embed all documents
    texts = [utility_bill_chunk, driving_license_chunk]
    doc_results = await embedding_service.embed_document_batch(texts)
    query_result = await embedding_service.embed_query(query)

    # Convert to numpy arrays
    query_emb = np.array(query_result.embedding)
    bill_emb = np.array(doc_results[0].embedding)
    license_emb = np.array(doc_results[1].embedding)

    # Compute cosine similarities
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    sim_query_bill = cosine_similarity(query_emb, bill_emb)
    sim_query_license = cosine_similarity(query_emb, license_emb)

    # Utility bill should be more semantically similar to "address proof"
    # than driving license (though both contain addresses)
    assert sim_query_bill > sim_query_license

    # Utility bill should have reasonable semantic similarity
    assert sim_query_bill > 0.25 # TODO: This should be reviewed as performance drops in batch.

    # Verify batch maintained correct text-embedding pairing
    assert doc_results[0].text == utility_bill_chunk
    assert doc_results[1].text == driving_license_chunk
