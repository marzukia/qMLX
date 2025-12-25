# SPDX-License-Identifier: Apache-2.0
"""Tests for the OpenAI-compatible API server."""

import platform
import sys

import pytest


# Skip all tests if not on Apple Silicon
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Requires Apple Silicon"
)


# =============================================================================
# Unit Tests - Request/Response Models
# =============================================================================

class TestRequestModels:
    """Test Pydantic request models."""

    def test_chat_message_text_only(self):
        """Test chat message with text content."""
        from vllm_mlx.server import Message

        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_chat_message_multimodal(self):
        """Test chat message with multimodal content."""
        from vllm_mlx.server import Message, ContentPart

        content = [
            {"type": "text", "text": "What's this?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}}
        ]
        msg = Message(role="user", content=content)

        assert msg.role == "user"
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2

    def test_image_url_model(self):
        """Test ImageUrl model."""
        from vllm_mlx.server import ImageUrl

        img_url = ImageUrl(url="https://example.com/image.jpg")
        assert img_url.url == "https://example.com/image.jpg"
        assert img_url.detail is None

    def test_video_url_model(self):
        """Test VideoUrl model."""
        from vllm_mlx.server import VideoUrl

        video_url = VideoUrl(url="https://example.com/video.mp4")
        assert video_url.url == "https://example.com/video.mp4"

    def test_content_part_text(self):
        """Test ContentPart with text."""
        from vllm_mlx.server import ContentPart

        part = ContentPart(type="text", text="Hello world")
        assert part.type == "text"
        assert part.text == "Hello world"

    def test_content_part_image(self):
        """Test ContentPart with image_url."""
        from vllm_mlx.server import ContentPart

        part = ContentPart(
            type="image_url",
            image_url={"url": "https://example.com/img.jpg"}
        )
        assert part.type == "image_url"
        # image_url can be dict or ImageUrl object
        if isinstance(part.image_url, dict):
            assert part.image_url["url"] == "https://example.com/img.jpg"
        else:
            assert part.image_url.url == "https://example.com/img.jpg"

    def test_content_part_video(self):
        """Test ContentPart with video."""
        from vllm_mlx.server import ContentPart

        part = ContentPart(type="video", video="/path/to/video.mp4")
        assert part.type == "video"
        assert part.video == "/path/to/video.mp4"

    def test_content_part_video_url(self):
        """Test ContentPart with video_url."""
        from vllm_mlx.server import ContentPart

        part = ContentPart(
            type="video_url",
            video_url={"url": "https://example.com/video.mp4"}
        )
        assert part.type == "video_url"
        # video_url can be dict or VideoUrl object
        if isinstance(part.video_url, dict):
            assert part.video_url["url"] == "https://example.com/video.mp4"
        else:
            assert part.video_url.url == "https://example.com/video.mp4"


class TestChatCompletionRequest:
    """Test ChatCompletionRequest model."""

    def test_basic_request(self):
        """Test basic chat completion request."""
        from vllm_mlx.server import ChatCompletionRequest, Message

        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")]
        )

        assert request.model == "test-model"
        assert len(request.messages) == 1
        assert request.max_tokens is None  # uses _default_max_tokens when None
        assert request.temperature == 0.7  # default
        assert request.stream is False  # default

    def test_request_with_options(self):
        """Test request with custom options."""
        from vllm_mlx.server import ChatCompletionRequest, Message

        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            max_tokens=100,
            temperature=0.5,
            stream=True,
        )

        assert request.max_tokens == 100
        assert request.temperature == 0.5
        assert request.stream is True

    def test_request_with_video_params(self):
        """Test request with video parameters."""
        from vllm_mlx.server import ChatCompletionRequest, Message

        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Describe the video")],
            video_fps=2.0,
            video_max_frames=16,
        )

        assert request.video_fps == 2.0
        assert request.video_max_frames == 16


class TestCompletionRequest:
    """Test CompletionRequest model."""

    def test_basic_completion_request(self):
        """Test basic completion request."""
        from vllm_mlx.server import CompletionRequest

        request = CompletionRequest(
            model="test-model",
            prompt="Once upon a time"
        )

        assert request.model == "test-model"
        assert request.prompt == "Once upon a time"
        assert request.max_tokens is None  # uses _default_max_tokens when None


# =============================================================================
# Helper Function Tests
# =============================================================================

class TestHelperFunctions:
    """Test server helper functions."""

    def test_is_mllm_model_patterns(self):
        """Test MLLM model detection patterns."""
        from vllm_mlx.server import is_mllm_model

        # Should detect as MLLM
        assert is_mllm_model("mlx-community/Qwen3-VL-4B-Instruct-3bit")
        assert is_mllm_model("mlx-community/llava-1.5-7b-4bit")
        assert is_mllm_model("mlx-community/paligemma-3b-mix-224-4bit")
        assert is_mllm_model("mlx-community/pixtral-12b-4bit")
        assert is_mllm_model("mlx-community/Idefics3-8B-Llama3-4bit")
        assert is_mllm_model("mlx-community/deepseek-vl-7b-chat-4bit")

        # Should NOT detect as MLLM
        assert not is_mllm_model("mlx-community/Llama-3.2-1B-Instruct-4bit")
        assert not is_mllm_model("mlx-community/Mistral-7B-Instruct-4bit")
        assert not is_mllm_model("mlx-community/Qwen2-7B-Instruct-4bit")

    def test_extract_multimodal_content_text_only(self):
        """Test extracting content from text-only messages."""
        from vllm_mlx.server import extract_multimodal_content, Message

        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ]

        processed, images, videos = extract_multimodal_content(messages)

        assert len(processed) == 2
        assert processed[0]["content"] == "Hello"
        assert len(images) == 0
        assert len(videos) == 0

    def test_extract_multimodal_content_with_image(self):
        """Test extracting content with images."""
        from vllm_mlx.server import extract_multimodal_content, Message

        messages = [
            Message(role="user", content=[
                {"type": "text", "text": "What's this?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}}
            ])
        ]

        processed, images, videos = extract_multimodal_content(messages)

        assert len(processed) == 1
        assert processed[0]["content"] == "What's this?"
        assert len(images) == 1
        assert "https://example.com/img.jpg" in images[0]

    def test_extract_multimodal_content_with_video(self):
        """Test extracting content with videos."""
        from vllm_mlx.server import extract_multimodal_content, Message

        messages = [
            Message(role="user", content=[
                {"type": "text", "text": "Describe this video"},
                {"type": "video", "video": "/path/to/video.mp4"}
            ])
        ]

        processed, images, videos = extract_multimodal_content(messages)

        assert len(processed) == 1
        assert processed[0]["content"] == "Describe this video"
        assert len(videos) == 1
        assert videos[0] == "/path/to/video.mp4"

    def test_extract_multimodal_content_with_video_url(self):
        """Test extracting content with video_url format."""
        from vllm_mlx.server import extract_multimodal_content, Message

        messages = [
            Message(role="user", content=[
                {"type": "text", "text": "What happens?"},
                {"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}}
            ])
        ]

        processed, images, videos = extract_multimodal_content(messages)

        assert len(videos) == 1




# =============================================================================
# Integration Tests (require running server)
# =============================================================================

@pytest.mark.slow
@pytest.mark.integration
class TestServerIntegration:
    """Integration tests that require a running server.

    These tests are skipped by default. Run with:
        pytest -m integration --server-url http://localhost:8000
    """

    @pytest.fixture
    def server_url(self, request):
        """Get server URL from command line or use default."""
        return request.config.getoption("--server-url", default="http://localhost:8000")

    def test_health_endpoint(self, server_url):
        """Test /health endpoint."""
        import requests

        response = requests.get(f"{server_url}/health", timeout=5)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "model_name" in data

    def test_models_endpoint(self, server_url):
        """Test /v1/models endpoint."""
        import requests

        response = requests.get(f"{server_url}/v1/models", timeout=5)
        assert response.status_code == 200

        data = response.json()
        assert "data" in data
        assert len(data["data"]) > 0

    def test_chat_completion(self, server_url):
        """Test /v1/chat/completions endpoint."""
        import requests

        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 10,
        }

        response = requests.post(
            f"{server_url}/v1/chat/completions",
            json=payload,
            timeout=30,
        )
        assert response.status_code == 200

        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert data["choices"][0]["message"]["content"]


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--server-url",
        action="store",
        default="http://localhost:8000",
        help="URL of the vllm-mlx server for integration tests",
    )
