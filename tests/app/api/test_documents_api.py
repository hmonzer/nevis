from src.app.core.domain.models import Client
from src.client.schemas import CreateDocumentRequest


async def test_get_documents_by_ids(nevis_client, unit_of_work):

    # given
    client = Client(first_name="john", last_name="doe", email="john.doe@nevis.com", description="Test client")
    async with unit_of_work:
        unit_of_work.add(client)

    docs = [await nevis_client.upload_document(client_id=client.id, request=CreateDocumentRequest(title=f"Test Doc {i}", content=f"Test content for retrieval {i}"))
           for i in range(3)]

    doc_ids = [d.id for d in docs]
    # when
    result = await nevis_client.get_documents(doc_ids)

    # then
    assert len(result) == 3


async def test_get_documents_by_ids_for_subset(nevis_client, unit_of_work):

    # given
    client = Client(first_name="john", last_name="doe", email="john.doe@nevis.com", description="Test client")
    async with unit_of_work:
        unit_of_work.add(client)

    docs = [await nevis_client.upload_document(client_id=client.id, request=CreateDocumentRequest(title=f"Test Doc {i}", content=f"Test content for retrieval {i}"))
           for i in range(3)]

    # when
    result = await nevis_client.get_documents([docs[0].id])

    # then
    assert len(result) == 1
    assert result[0].title == docs[0].title


