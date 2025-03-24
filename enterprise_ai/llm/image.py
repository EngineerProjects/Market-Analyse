"""
Image handling utilities for LLM module.

This module provides a comprehensive set of utilities for detecting, processing,
and encoding images for use with vision-capable LLMs. It supports multiple input
formats including file paths, URLs, bytes, and already encoded base64 strings.
"""

import base64
import mimetypes
import os
import re
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import httpx

from enterprise_ai.llm.exceptions import ImageProcessingError
from enterprise_ai.logger import get_logger

# Initialize logger
logger = get_logger("llm.image")


class ImageHandler:
    """Intelligent image handler for automatic detection and encoding."""

    # Supported image formats and their content types
    SUPPORTED_FORMATS = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp",
        "bmp": "image/bmp",
    }

    # Regular expression to detect if a string is already base64 encoded
    BASE64_PATTERN = re.compile(r"^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$")

    # Pattern to detect data URLs
    DATA_URL_PATTERN = re.compile(r"^data:image/([a-zA-Z]+);base64,([a-zA-Z0-9+/]+=*)$")

    @classmethod
    def process_image(
        cls,
        image: Any,
        detect_format: bool = True,
        validate: bool = True,
        default_format: str = "jpeg",
    ) -> Tuple[str, str]:
        """Process an image into base64 encoding with automatic format detection.

        Args:
            image: The image to process (str path, URL, bytes, or base64 string)
            detect_format: Whether to try to detect the image format
            validate: Whether to validate the processed image
            default_format: Default format to use if detection fails

        Returns:
            Tuple of (base64_encoded_image, media_type)

        Raises:
            ImageProcessingError: If the image cannot be processed
        """
        try:
            # Case 1: Image is already a valid base64 string
            if isinstance(image, str) and cls._is_base64(image):
                logger.debug("Image is already base64 encoded")
                media_type = f"image/{default_format}"
                return image, media_type

            # Case 2: Image is a data URL (e.g., data:image/jpeg;base64,...)
            if isinstance(image, str) and image.startswith("data:image/"):
                logger.debug("Image is a data URL")
                match = cls.DATA_URL_PATTERN.match(image)
                if match:
                    format_name, base64_data = match.groups()
                    media_type = f"image/{format_name}"
                    return base64_data, media_type
                else:
                    # Invalid data URL format
                    raise ImageProcessingError("Invalid data URL format")

            # Case 3: Image is a URL
            if isinstance(image, str) and (
                image.startswith("http://")
                or image.startswith("https://")
                or image.startswith("ftp://")
            ):
                logger.debug("Image is a URL")
                base64_data = cls._encode_from_url(image)
                media_type = cls._detect_media_type_from_url(image, default_format)
                return base64_data, media_type

            # Case 4: Image is a file path
            if isinstance(image, (str, Path)) and not (
                isinstance(image, str)
                and (image.startswith("http://") or image.startswith("https://"))
            ):
                logger.debug("Image is a file path")
                image_path = Path(image)
                if not image_path.exists():
                    raise ImageProcessingError(f"Image file not found: {image_path}")

                base64_data = cls._encode_from_file(image_path)
                media_type = cls._detect_media_type_from_path(image_path, default_format)
                return base64_data, media_type

            # Case 5: Image is bytes
            if isinstance(image, bytes):
                logger.debug("Image is bytes")
                base64_data = cls._encode_from_bytes(image)
                # Can't detect media type from bytes alone, use default
                media_type = f"image/{default_format}"
                return base64_data, media_type

            # If we got here, we don't know how to handle this type
            raise ImageProcessingError(f"Unsupported image type: {type(image)}")

        except Exception as e:
            if isinstance(e, ImageProcessingError):
                # Re-raise existing ImageProcessingError
                raise
            else:
                # Wrap other exceptions
                logger.error(f"Error processing image: {str(e)}")
                raise ImageProcessingError(f"Failed to process image: {str(e)}")

    @classmethod
    def _is_base64(cls, s: str) -> bool:
        """Check if a string is likely a base64 encoded image.

        Args:
            s: String to check

        Returns:
            True if the string is likely a base64 encoded image
        """
        # First, check if the string contains only valid base64 characters
        if not cls.BASE64_PATTERN.match(s):
            return False

        # Check length (must be multiple of 4)
        if len(s) % 4 != 0:
            return False

        # Try to decode a small sample to confirm it's valid base64
        try:
            sample = s[:20] + "=" * (4 - len(s[:20]) % 4)  # Pad to multiple of 4
            base64.b64decode(sample)
            return True
        except Exception:
            return False

    @classmethod
    def _encode_from_file(cls, path: Path) -> str:
        """Encode an image file as base64.

        Args:
            path: Path to the image file

        Returns:
            Base64-encoded image

        Raises:
            ImageProcessingError: If the image file cannot be encoded
        """
        try:
            with open(path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Error encoding image file {path}: {str(e)}")
            raise ImageProcessingError(f"Failed to encode image file: {str(e)}", str(path))

    @classmethod
    def _encode_from_bytes(cls, image_data: bytes) -> str:
        """Encode image bytes as base64.

        Args:
            image_data: Image bytes

        Returns:
            Base64-encoded image

        Raises:
            ImageProcessingError: If the image data cannot be encoded
        """
        try:
            return base64.b64encode(image_data).decode("utf-8")
        except Exception as e:
            logger.error(f"Error encoding image bytes: {str(e)}")
            raise ImageProcessingError(f"Failed to encode image bytes: {str(e)}")

    @classmethod
    def _encode_from_url(cls, url: str) -> str:
        """Download and encode an image from a URL.

        Args:
            url: URL of the image

        Returns:
            Base64-encoded image

        Raises:
            ImageProcessingError: If the image cannot be downloaded or encoded
        """
        try:
            response = httpx.get(url, timeout=30.0, follow_redirects=True)
            response.raise_for_status()
            return cls._encode_from_bytes(response.content)
        except Exception as e:
            logger.error(f"Error downloading image from {url}: {str(e)}")
            raise ImageProcessingError(f"Failed to download image: {str(e)}", url)

    @classmethod
    def _detect_media_type_from_path(cls, path: Path, default_format: str = "jpeg") -> str:
        """Detect the media type of an image from its file path.

        Args:
            path: Path to the image file
            default_format: Default format to use if detection fails

        Returns:
            Media type string (e.g., "image/jpeg")
        """
        # Get extension without the dot
        ext = path.suffix.lower().lstrip(".")

        # Check if it's a supported format
        if ext in cls.SUPPORTED_FORMATS:
            return cls.SUPPORTED_FORMATS[ext]

        # Try using mimetypes
        mime_type, _ = mimetypes.guess_type(str(path))
        if mime_type and mime_type.startswith("image/"):
            return mime_type

        # Use default
        return f"image/{default_format}"

    @classmethod
    def _detect_media_type_from_url(cls, url: str, default_format: str = "jpeg") -> str:
        """Detect the media type of an image from its URL.

        Args:
            url: URL of the image
            default_format: Default format to use if detection fails

        Returns:
            Media type string (e.g., "image/jpeg")
        """
        # Extract path from URL
        path = url.split("?")[0]  # Remove query parameters

        # Get extension from the path
        ext = os.path.splitext(path)[1].lower().lstrip(".")

        # Check if it's a supported format
        if ext in cls.SUPPORTED_FORMATS:
            return cls.SUPPORTED_FORMATS[ext]

        # Try using mimetypes
        mime_type, _ = mimetypes.guess_type(url)
        if mime_type and mime_type.startswith("image/"):
            return mime_type

        # Use default
        return f"image/{default_format}"
