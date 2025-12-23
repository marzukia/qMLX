# SPDX-License-Identifier: Apache-2.0
"""
Engine Core for vllm-mlx continuous batching.

This module provides the EngineCore class that coordinates:
- Model loading and management
- Request scheduling via Scheduler
- Async request processing
- Output streaming

The design follows vLLM's engine architecture adapted for MLX.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Union

from .request import Request, RequestOutput, RequestStatus, SamplingParams
from .scheduler import Scheduler, SchedulerConfig, SchedulerOutput

logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    """Configuration for the engine."""

    model_name: str = ""
    scheduler_config: Optional[SchedulerConfig] = None
    step_interval: float = 0.001  # 1ms between steps


class EngineCore:
    """
    Core engine for vllm-mlx inference with continuous batching.

    This engine runs the generation loop and manages request lifecycle.
    It provides both sync and async interfaces for request handling.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: Optional[EngineConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or EngineConfig()

        # Create scheduler
        scheduler_config = self.config.scheduler_config or SchedulerConfig()
        self.scheduler = Scheduler(
            model=model,
            tokenizer=tokenizer,
            config=scheduler_config,
        )

        # Output queues for streaming
        self._output_queues: Dict[str, asyncio.Queue] = {}
        self._finished_events: Dict[str, asyncio.Event] = {}

        # Engine state
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._start_time: Optional[float] = None
        self._steps_executed = 0

    async def start(self) -> None:
        """Start the engine loop."""
        if self._running:
            return

        self._running = True
        self._start_time = time.time()
        self._task = asyncio.create_task(self._engine_loop())
        logger.info("Engine started")

    async def stop(self) -> None:
        """Stop the engine loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Engine stopped")

    def is_running(self) -> bool:
        """Check if engine is running."""
        return self._running

    async def _engine_loop(self) -> None:
        """Main engine loop."""
        while self._running:
            try:
                if self.scheduler.has_requests():
                    # Run one generation step
                    output = self.scheduler.step()
                    self._steps_executed += 1

                    # Distribute outputs to waiting clients
                    for req_output in output.outputs:
                        queue = self._output_queues.get(req_output.request_id)
                        if queue:
                            await queue.put(req_output)

                        if req_output.finished:
                            event = self._finished_events.get(req_output.request_id)
                            if event:
                                event.set()
                else:
                    # No work, yield control
                    await asyncio.sleep(self.config.step_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Engine loop error: {e}")
                await asyncio.sleep(0.1)

    async def add_request(
        self,
        prompt: Union[str, List[int]],
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
        images: Optional[List[Any]] = None,
        videos: Optional[List[Any]] = None,
    ) -> str:
        """
        Add a request for processing.

        Args:
            prompt: Input prompt (string or token IDs)
            sampling_params: Generation parameters
            request_id: Optional custom request ID
            images: Optional images for multimodal
            videos: Optional videos for multimodal

        Returns:
            The request ID
        """
        if request_id is None:
            request_id = str(uuid.uuid4())

        if sampling_params is None:
            sampling_params = SamplingParams()

        request = Request(
            request_id=request_id,
            prompt=prompt,
            sampling_params=sampling_params,
            images=images,
            videos=videos,
        )

        # Setup output queue
        self._output_queues[request_id] = asyncio.Queue()
        self._finished_events[request_id] = asyncio.Event()

        # Add to scheduler
        self.scheduler.add_request(request)

        return request_id

    async def abort_request(self, request_id: str) -> bool:
        """Abort a request."""
        result = self.scheduler.abort_request(request_id)
        self._cleanup_request(request_id)
        return result

    def _cleanup_request(self, request_id: str) -> None:
        """Clean up request tracking."""
        self._output_queues.pop(request_id, None)
        self._finished_events.pop(request_id, None)
        self.scheduler.remove_finished_request(request_id)

    async def stream_outputs(
        self,
        request_id: str,
        timeout: Optional[float] = None,
    ) -> AsyncIterator[RequestOutput]:
        """
        Stream outputs for a request.

        Args:
            request_id: The request ID
            timeout: Optional timeout in seconds

        Yields:
            RequestOutput objects as tokens are generated
        """
        queue = self._output_queues.get(request_id)
        if queue is None:
            # Request might not be added yet or already cleaned up
            return

        try:
            while True:
                try:
                    if timeout:
                        output = await asyncio.wait_for(queue.get(), timeout=timeout)
                    else:
                        output = await queue.get()

                    yield output

                    if output.finished:
                        break

                except asyncio.TimeoutError:
                    logger.warning(f"Timeout waiting for request {request_id}")
                    break

        finally:
            self._cleanup_request(request_id)

    async def generate(
        self,
        prompt: Union[str, List[int]],
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> RequestOutput:
        """
        Generate a complete response (non-streaming).

        Args:
            prompt: Input prompt
            sampling_params: Generation parameters
            request_id: Optional request ID

        Returns:
            Final RequestOutput with complete text
        """
        request_id = await self.add_request(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            **kwargs,
        )

        final_output = None
        async for output in self.stream_outputs(request_id, timeout=300):
            final_output = output

        if final_output is None:
            raise RuntimeError(f"No output for request {request_id}")

        return final_output

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        scheduler_stats = self.scheduler.get_stats()
        uptime = time.time() - self._start_time if self._start_time else 0

        return {
            "running": self._running,
            "uptime_seconds": uptime,
            "steps_executed": self._steps_executed,
            "active_requests": len(self._output_queues),
            **scheduler_stats,
        }

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get prefix cache statistics."""
        return self.scheduler.get_cache_stats()


class AsyncEngineCore:
    """
    Async context manager wrapper for EngineCore.

    Usage:
        async with AsyncEngineCore(model, tokenizer) as engine:
            request_id = await engine.add_request("Hello")
            async for output in engine.stream_outputs(request_id):
                print(output.new_text)
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: Optional[EngineConfig] = None,
    ):
        self.engine = EngineCore(model, tokenizer, config)

    async def __aenter__(self) -> "AsyncEngineCore":
        await self.engine.start()
        return self

    async def __aexit__(self, *args) -> None:
        await self.engine.stop()

    def start(self) -> None:
        """Start engine (creates task in current loop)."""
        asyncio.create_task(self.engine.start())

    async def stop(self) -> None:
        """Stop the engine."""
        await self.engine.stop()

    async def add_request(
        self,
        prompt: Union[str, List[int]],
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Add a request."""
        return await self.engine.add_request(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            **kwargs,
        )

    async def abort_request(self, request_id: str) -> bool:
        """Abort a request."""
        return await self.engine.abort_request(request_id)

    async def stream_outputs(
        self,
        request_id: str,
        timeout: Optional[float] = None,
    ) -> AsyncIterator[RequestOutput]:
        """Stream outputs."""
        async for output in self.engine.stream_outputs(request_id, timeout):
            yield output

    async def generate(
        self,
        prompt: Union[str, List[int]],
        sampling_params: Optional[SamplingParams] = None,
        **kwargs,
    ) -> RequestOutput:
        """Generate complete response."""
        return await self.engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            **kwargs,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get engine stats."""
        return self.engine.get_stats()

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get prefix cache statistics."""
        return self.engine.get_cache_stats()
