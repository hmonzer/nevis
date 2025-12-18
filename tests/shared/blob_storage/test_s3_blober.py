"""Unit tests for S3BlobStorage."""
import pytest

from src.shared.blob_storage.s3_blober import S3BlobStorage


@pytest.mark.asyncio
async def test_generate_presigned_url(s3_storage: S3BlobStorage):
    """Test generating a pre-signed URL for an existing object."""
    # Upload a test object
    test_key = "test/presigned_url_test.txt"
    test_content = "This is test content for pre-signed URL generation."
    await s3_storage.upload_text_content(test_key, test_content)

    # Generate pre-signed URL
    url = await s3_storage.generate_presigned_url(test_key, expiration=3600)

    # Verify URL is returned and contains expected components
    assert url is not None
    assert isinstance(url, str)
    assert test_key in url
    assert "X-Amz-Signature" in url or "Signature" in url


@pytest.mark.asyncio
async def test_generate_presigned_url_custom_expiration(s3_storage: S3BlobStorage):
    """Test that custom expiration is accepted."""
    test_key = "test/presigned_url_expiration_test.txt"
    test_content = "Content for expiration test."
    await s3_storage.upload_text_content(test_key, test_content)

    # Generate with custom expiration (5 minutes)
    url = await s3_storage.generate_presigned_url(test_key, expiration=300)

    assert url is not None
    assert isinstance(url, str)


@pytest.mark.asyncio
async def test_generate_presigned_url_nonexistent_key(s3_storage: S3BlobStorage):
    """Test that pre-signed URL can be generated for non-existent keys.

    Note: S3 generates pre-signed URLs without checking if the object exists.
    The URL will be valid but will return 404 when accessed.
    """
    nonexistent_key = "test/nonexistent_key_12345.txt"

    # S3 will generate a URL even for non-existent objects
    url = await s3_storage.generate_presigned_url(nonexistent_key, expiration=3600)

    assert url is not None
    assert isinstance(url, str)
    assert nonexistent_key in url
