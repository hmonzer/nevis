from typing import List, Optional
from pydantic import BaseModel
from src.client.schemas import CreateClientRequest, CreateDocumentRequest

class ClientRecord(CreateClientRequest):
    id: str

class DocumentRecord(CreateDocumentRequest):
    id: str
    client_id: str

class Corpus(BaseModel):
    clients: List[ClientRecord]
    documents: List[DocumentRecord]

class TestItem(BaseModel):
    query_id: str
    query_text: str
    expected_result_ids: List[str]

class UseCase(BaseModel):
    id: str
    title: str
    description: Optional[str] = None
    tests: List[TestItem]

class EvalSuite(BaseModel):
    suite_name: str
    version: str
    description: Optional[str] = None
    corpus: Corpus
    use_cases: List[UseCase]