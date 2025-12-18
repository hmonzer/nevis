"""S3 Blob Storage adapter for file storage operations."""
import asyncio
from typing import Optional

import boto3
from botocore.exceptions import ClientError
from pydantic import BaseModel, Field


class S3BlobStorageSettings(BaseModel):
    """Settings for S3 Blob Storage."""
    bucket_name: str = Field(..., description="S3 bucket name")
    endpoint_url: Optional[str] = Field(None, description="S3 endpoint URL (for LocalStack)")
    region_name: str = Field(default="us-east-1", description="AWS region")
    aws_access_key_id: Optional[str] = Field(None, description="AWS access key ID")
    aws_secret_access_key: Optional[str] = Field(None, description="AWS secret access key")


class S3BlobStorage:
    """
    Robust S3 Blob Storage adapter compatible with both AWS S3 and LocalStack.

    This adapter provides methods for uploading and downloading content from S3.
    It uses boto3 for S3 operations and wraps synchronous calls in async context.
    """

    def __init__(self, settings: S3BlobStorageSettings):
        """
        Initialize S3 Blob Storage adapter.

        Args:
            settings: S3 storage configuration
        """
        self.settings = settings
        self._client = None
        self._resource = None

    @property
    def client(self):
        """Get or create S3 client (lazy initialization)."""
        if self._client is None:
            client_kwargs = {
                "region_name": self.settings.region_name,
            }

            if self.settings.endpoint_url:
                client_kwargs["endpoint_url"] = self.settings.endpoint_url

            if self.settings.aws_access_key_id and self.settings.aws_secret_access_key:
                client_kwargs["aws_access_key_id"] = self.settings.aws_access_key_id
                client_kwargs["aws_secret_access_key"] = self.settings.aws_secret_access_key

            self._client = boto3.client("s3", **client_kwargs)  # type: ignore[call-overload]

        return self._client

    @property
    def resource(self):
        """Get or create S3 resource (lazy initialization)."""
        if self._resource is None:
            resource_kwargs = {
                "region_name": self.settings.region_name,
            }

            if self.settings.endpoint_url:
                resource_kwargs["endpoint_url"] = self.settings.endpoint_url

            if self.settings.aws_access_key_id and self.settings.aws_secret_access_key:
                resource_kwargs["aws_access_key_id"] = self.settings.aws_access_key_id
                resource_kwargs["aws_secret_access_key"] = self.settings.aws_secret_access_key

            self._resource = boto3.resource("s3", **resource_kwargs)  # type: ignore[call-overload]

        return self._resource

    async def ensure_bucket_exists(self) -> None:
        """
        Ensure the configured bucket exists, create if it doesn't.

        Raises:
            ClientError: If bucket creation fails
        """
        await asyncio.to_thread(self._ensure_bucket_exists_sync)

    def _ensure_bucket_exists_sync(self) -> None:
        """Synchronous helper to ensure bucket exists."""
        try:
            self.client.head_bucket(Bucket=self.settings.bucket_name)
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "404":
                # Bucket doesn't exist, create it
                try:
                    if self.settings.region_name == "us-east-1":
                        # us-east-1 doesn't support LocationConstraint
                        self.client.create_bucket(Bucket=self.settings.bucket_name)
                    else:
                        self.client.create_bucket(
                            Bucket=self.settings.bucket_name,
                            CreateBucketConfiguration={"LocationConstraint": self.settings.region_name}
                        )
                except ClientError as create_error:
                    raise RuntimeError(
                        f"Failed to create bucket {self.settings.bucket_name}: {str(create_error)}"
                    ) from create_error
            else:
                raise RuntimeError(
                    f"Failed to access bucket {self.settings.bucket_name}: {str(e)}"
                ) from e

    async def upload_text_content(self, key: str, content: str) -> str:
        """
        Upload text content to S3.

        Args:
            key: S3 object key (path) for the content
            content: Text content to upload

        Returns:
            The S3 key where content was uploaded

        Raises:
            RuntimeError: If upload fails
        """
        try:
            # Ensure bucket exists before upload
            await self.ensure_bucket_exists()

            # Upload content as text
            await asyncio.to_thread(
                self.client.put_object,
                Bucket=self.settings.bucket_name,
                Key=key,
                Body=content.encode("utf-8"),
                ContentType="text/plain"
            )

            return key
        except ClientError as e:
            raise RuntimeError(
                f"Failed to upload content to S3 key {key}: {str(e)}"
            ) from e

    async def download_text_content(self, key: str) -> str:
        """
        Download text content from S3.

        Args:
            key: S3 object key (path) to download

        Returns:
            Text content from S3

        Raises:
            RuntimeError: If download fails or key doesn't exist
        """
        try:
            response = await asyncio.to_thread(
                self.client.get_object,
                Bucket=self.settings.bucket_name,
                Key=key
            )

            # Read and decode content
            content_bytes = await asyncio.to_thread(response["Body"].read)
            return content_bytes.decode("utf-8")
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "NoSuchKey":
                raise RuntimeError(f"S3 key {key} does not exist") from e
            raise RuntimeError(
                f"Failed to download content from S3 key {key}: {str(e)}"
            ) from e

    async def delete_object(self, key: str) -> None:
        """
        Delete an object from S3.

        Args:
            key: S3 object key (path) to delete

        Raises:
            RuntimeError: If deletion fails
        """
        try:
            await asyncio.to_thread(
                self.client.delete_object,
                Bucket=self.settings.bucket_name,
                Key=key
            )
        except ClientError as e:
            raise RuntimeError(
                f"Failed to delete S3 key {key}: {str(e)}"
            ) from e

    async def object_exists(self, key: str) -> bool:
        """
        Check if an object exists in S3.

        Args:
            key: S3 object key (path) to check

        Returns:
            True if object exists, False otherwise
        """
        try:
            await asyncio.to_thread(
                self.client.head_object,
                Bucket=self.settings.bucket_name,
                Key=key
            )
            return True
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "404":
                return False
            # Re-raise other errors
            raise RuntimeError(
                f"Failed to check existence of S3 key {key}: {str(e)}"
            ) from e

    async def generate_presigned_url(self, key: str, expiration: int = 3600) -> str:
        """
        Generate a pre-signed URL for downloading an object from S3.

        Args:
            key: S3 object key (path) to generate URL for
            expiration: URL expiration time in seconds (default: 1 hour)

        Returns:
            Pre-signed URL for downloading the object

        Raises:
            RuntimeError: If URL generation fails
        """
        try:
            url = await asyncio.to_thread(
                self.client.generate_presigned_url,
                "get_object",
                Params={"Bucket": self.settings.bucket_name, "Key": key},
                ExpiresIn=expiration,
            )
            return url
        except ClientError as e:
            raise RuntimeError(
                f"Failed to generate pre-signed URL for S3 key {key}: {str(e)}"
            ) from e
