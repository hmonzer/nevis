from src.app.core.services.chunking import RecursiveChunkingStrategy


def test_recursive_chunking_strategy_chunks_text():
    # Arrange
    strategy = RecursiveChunkingStrategy(chunk_size=100, chunk_overlap=20)
    text = "This is a long sentence that needs to be chunked. " * 5

    # Act
    chunks = strategy.chunk_text(text)

    # Assert
    assert isinstance(chunks, list)
    assert len(chunks) > 1
    for chunk in chunks:
        assert isinstance(chunk, str)
        assert len(chunk) <= 100


def test_recursive_chunking_with_overlap():
    # Arrange
    strategy = RecursiveChunkingStrategy(chunk_size=35, chunk_overlap=15)
    text = "This is a test text to see how the chunking and overlap works."

    # Act
    chunks = strategy.chunk_text(text)

    # Assert
    assert len(chunks) > 1
    # Check that there is an overlap between consecutive chunks
    for i in range(len(chunks) - 1):
        assert chunks[i+1][:5] in chunks[i]


def test_recursive_chunking_empty_text():
    # Arrange
    strategy = RecursiveChunkingStrategy()
    text = ""

    # Act
    chunks = strategy.chunk_text(text)

    # Assert
    assert chunks == []


def test_recursive_chunking_whitespace_text():
    # Arrange
    strategy = RecursiveChunkingStrategy()
    text = "     \n\n   "

    # Act
    chunks = strategy.chunk_text(text)

    # Assert
    assert chunks == []


def test_recursive_chunking_small_text():
    # Arrange
    strategy = RecursiveChunkingStrategy(chunk_size=100, chunk_overlap=20)
    text = "This is a short text."

    # Act
    chunks = strategy.chunk_text(text)

    # Assert
    assert len(chunks) == 1
    assert chunks[0] == text


def test_recursive_chunking_custom_separators():
    # Arrange
    strategy = RecursiveChunkingStrategy(chunk_size=10, chunk_overlap=5, separators=["|"])
    text = "chunk1|chunk2|chunk3"

    # Act
    chunks = strategy.chunk_text(text)

    # Assert
    assert len(chunks) == 3
    assert chunks[0] == "chunk1"
    assert chunks[1] == "|chunk2"
    assert chunks[2] == "|chunk3"

