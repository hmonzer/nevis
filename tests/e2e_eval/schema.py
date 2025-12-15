from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from src.client.schemas import CreateClientRequest, CreateDocumentRequest

class ClientRecord(CreateClientRequest):
    id: str

class DocumentRecord(CreateDocumentRequest):
    id: str
    client_id: str

class Corpus(BaseModel):
    client_records: List[ClientRecord]
    documents: List[DocumentRecord]

class TestItem(BaseModel):
    query_id: str
    query_text: str
    expected_result_ids: List[str]

class UseCase(BaseModel):
    id: str
    title: str
    description: Optional[str] = None
    corpus: Corpus
    tests: List[TestItem]

class EvalSuite(BaseModel):
    suite_name: str
    version: str
    use_cases: List[UseCase]