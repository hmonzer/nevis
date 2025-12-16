"""Integration tests for end-to-end document search functionality."""
import pytest
from uuid import uuid4

from src.app.core.domain.models import Client, SearchRequest


@pytest.mark.asyncio
async def test_end_to_end_document_search_proof_of_address(
    clean_database,
    document_service,
        search_service_no_rerank,
    unit_of_work_fixture,
    localstack_container
):
    """
    Integration test for document search with proof of address use case.

    This test verifies:
    1. Multiple documents are uploaded and processed (utility bill, passport, driving license)
    2. Semantic search for "proof of address" returns relevant documents
    3. Utility bill ranks highest as it's the best proof of address
    """
    # 1. Create a client
    client = Client(
        id=uuid4(),
        first_name="John",
        last_name="Doe",
        email="john.doe@test.com",
        description="Test client for proof of address search"
    )

    async with unit_of_work_fixture:
        unit_of_work_fixture.add(client)

    # 2. Create utility bill document (best proof of address)
    utility_bill_content = """
    ELECTRICITY BILL

    Account Holder: John Doe
    Service Address: 123 Main Street, Apartment 4B, Springfield, IL 62701
    Billing Period: October 1, 2024 - October 31, 2024

    This bill serves as proof of residence at the above address.
    The account holder is responsible for electricity usage at this residential address.

    Current Charges:
    - Energy Usage: $45.20
    - Distribution Charge: $12.50
    - Total Amount Due: $57.70

    Payment Due Date: November 15, 2024

    This utility bill can be used as proof of address for official purposes.
    Please retain this document for your records.
    """

    utility_bill = await document_service.create_document(
        client_id=client.id,
        title="Electricity Bill - October 2024",
        content=utility_bill_content
    )
    await document_service.process_document(utility_bill.id, utility_bill_content)

    # 3. Create passport document (proof of identity, NOT proof of address)
    passport_content = """
    PASSPORT

    Type: P (Regular Passport)
    Country Code: USA
    Passport No.: 123456789

    Surname: DOE
    Given Names: JOHN MICHAEL
    Nationality: UNITED STATES OF AMERICA
    Date of Birth: 15 MAR 1985
    Place of Birth: Chicago, Illinois
    Sex: M

    Date of Issue: 01 JAN 2020
    Date of Expiry: 01 JAN 2030
    Authority: U.S. Department of State

    This document is the property of the United States Government.
    It serves as proof of identity and citizenship.
    This passport is valid for international travel.
    """

    passport = await document_service.create_document(
        client_id=client.id,
        title="US Passport",
        content=passport_content
    )
    await document_service.process_document(passport.id, passport_content)

    # 4. Create driving license document (has address, but weaker proof than utility bill)
    license_content = """
    DRIVER LICENSE

    State of Illinois
    License Number: D123-4567-8901

    Name: JOHN MICHAEL DOE
    Address: 123 Main Street, Apt 4B
             Springfield, IL 62701

    Date of Birth: 03/15/1985
    Sex: M
    Height: 5'10"
    Eyes: Brown

    Class: D (Passenger Vehicles)
    Issue Date: 06/01/2023
    Expiration Date: 03/15/2029

    This license authorizes the holder to operate motor vehicles.
    Restrictions: Must wear corrective lenses
    """

    license = await document_service.create_document(
        client_id=client.id,
        title="Illinois Driver License",
        content=license_content
    )
    await document_service.process_document(license.id, license_content)

    # 5. Search for "proof of address"
    request = SearchRequest(query="proof of address", top_k=5)
    results = await search_service_no_rerank.search(request)

    # 6. Verify results
    assert len(results) > 0, "Should find at least one relevant chunk"

    # Verify scores are in descending order
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score, "Results should be ordered by score descending"

    # Verify all scores are valid (RRF scores are typically small positive values)
    for result in results:
        assert result.score > 0, f"Score {result.score} should be positive"

    # 7. Verify utility bill chunks rank highest
    # Get the top result's document_id
    top_result_doc_id = results[0].chunk.document_id

    # The top result should be from the utility bill (which has explicit "proof of address" language)
    assert top_result_doc_id == utility_bill.id, (
        f"Top result should be from utility bill (best proof of address), "
        f"but got document {top_result_doc_id}"
    )


@pytest.mark.asyncio
async def test_reranking_produces_different_scores(
    clean_database,
    document_service,
        search_service_no_rerank,
        search_service,
    unit_of_work_fixture,
    localstack_container
):
    """
    Test that reranking produces different (typically better) relevance scores.

    This verifies that the reranker is being applied and produces cross-encoder
    scores that differ from the initial cosine similarity scores.
    """
    # 1. Create a client
    client = Client(
        id=uuid4(),
        first_name="Alice",
        last_name="Johnson",
        email="alice.johnson@test.com",
        description="Test client for reranking comparison"
    )

    async with unit_of_work_fixture:
        unit_of_work_fixture.add(client)

    # 2. Create utility bill (best proof of address)
    utility_bill_content = """
    WATER UTILITY BILL

    Account Holder: Alice Johnson
    Service Address: 789 Pine Street, Unit 5, Seattle, WA 98101
    Billing Period: November 2024

    This utility bill serves as official proof of residence at the service address.
    This document can be used as proof of address for banking and government purposes.
    Please present this bill along with photo identification when proof of residence is required.

    Current Charges: $32.50
    Due Date: December 15, 2024
    Account Active Since: January 2020
    """

    utility_bill = await document_service.create_document(
        client_id=client.id,
        title="Water Bill - November 2024",
        content=utility_bill_content
    )
    await document_service.process_document(utility_bill.id, utility_bill_content)

    # 3. Search WITHOUT reranking
    request = SearchRequest(query="proof of address", top_k=5)
    results_no_rerank = await search_service_no_rerank.search(request)

    # 4. Search WITH reranking
    results_with_rerank = await search_service.search(request)

    # 5. Verify both return results
    assert len(results_no_rerank) > 0, "Should find results without reranking"
    assert len(results_with_rerank) > 0, "Should find results with reranking"

    # 6. Verify utility bill ranks first in both cases
    assert results_no_rerank[0].chunk.document_id == utility_bill.id, (
        "Utility bill should rank first without reranking"
    )
    assert results_with_rerank[0].chunk.document_id == utility_bill.id, (
        "Utility bill should rank first with reranking"
    )

    # 7. Verify scores are different (reranking changes the scoring)
    # The score without reranking is cosine similarity (typically -1 to 1)
    # The score with reranking is cross-encoder logits (unbounded)
    cosine_score = results_no_rerank[0].score
    reranked_score = results_with_rerank[0].score

    # Scores should be different due to different scoring methods
    assert cosine_score != reranked_score, (
        f"Reranking should produce different scores. "
        f"Cosine: {cosine_score:.4f}, Reranked: {reranked_score:.4f}"
    )

    print(f"\n=== Reranking Impact ===")
    print(f"Top result cosine similarity score: {cosine_score:.4f}")
    print(f"Top result cross-encoder score: {reranked_score:.4f}")
    print(f"Score difference: {abs(reranked_score - cosine_score):.4f}")
    print(f"Both methods ranked utility bill as most relevant ✓")


@pytest.mark.asyncio
async def test_search_with_similarity_threshold(
    clean_database,
    document_service,
        search_service_no_rerank,
    unit_of_work_fixture,
    localstack_container
):
    """
    Test that similarity threshold filters out low-relevance results.
    """
    # 1. Create a client
    client = Client(
        id=uuid4(),
        first_name="Jane",
        last_name="Smith",
        email="jane.smith@test.com",
        description="Test client for threshold testing"
    )

    async with unit_of_work_fixture:
        unit_of_work_fixture.add(client)

    # 2. Create document with specific content
    document_content = """
    Bank Statement

    Account Holder: Jane Smith
    Account Number: 9876543210
    Statement Period: September 2024

    This bank statement shows the address on file for the account holder.
    Service Address: 456 Oak Avenue, Unit 12, Portland, OR 97201

    Opening Balance: $1,250.00
    Deposits: $2,500.00
    Withdrawals: $1,800.00
    Closing Balance: $1,950.00

    This document can serve as proof of residence and financial standing.
    """

    document = await document_service.create_document(
        client_id=client.id,
        title="Bank Statement - September 2024",
        content=document_content
    )
    await document_service.process_document(document.id, document_content)

    # 3. Search with high similarity threshold (0.5) - filters vector search before RRF
    request_high = SearchRequest(query="proof of address", top_k=10, threshold=0.5)
    results_high_threshold = await search_service_no_rerank.search(request_high)

    # 4. Search with low similarity threshold (0.3)
    request_low = SearchRequest(query="proof of address", top_k=10, threshold=0.3)
    results_low_threshold = await search_service_no_rerank.search(request_low)

    # Assert - High threshold should return fewer or equal results
    assert len(results_high_threshold) <= len(results_low_threshold), (
        f"High threshold returned {len(results_high_threshold)} results, "
        f"low threshold returned {len(results_low_threshold)} results"
    )

    # All results should have positive scores (RRF scores)
    for result in results_high_threshold:
        assert result.score > 0, f"Result with score {result.score} should be positive"
