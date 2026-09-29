

import asyncio
import inspect
import logging

from omegaconf import DictConfig
from transformers import AutoTokenizer

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register as register_manager
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.ray_utils import get_event_loop
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register as register_manager_legacy

logger = logging.getLogger(__file__)


class AsyncTokenBucket:
    """Async token bucket for rate limiting with variable token consumption.

    The token bucket algorithm is a classic rate limiting technique that allows
    for burst traffic while maintaining an average rate limit. This implementation
    is async-first and thread-safe, designed for use in concurrent environments.

    The bucket starts full and refills at a constant rate (rate_limit tokens/second).
    When tokens are acquired, they are consumed from the bucket. If insufficient
    tokens are available, the acquire() method will sleep until enough tokens
    have been refilled.

    This implementation supports variable token consumption, making it suitable
    for rate limiting based on request size (e.g., API token usage).

    Args:
        rate_limit (float): The rate at which tokens are added to the bucket,
            in tokens per second. For example, rate_limit=10.0 means 10 tokens
            are added per second (or 600 per minute).
        max_tokens (float, optional): The maximum capacity of the token bucket.
            Defaults to rate_limit if not specified. This value determines the
            maximum burst size allowed.

    Attributes:
        rate_limit (float): Tokens added per second.
        max_tokens (float): Maximum bucket capacity.
        tokens (float): Current number of available tokens.
        last_update (float | None): Timestamp of last token update (from event loop).
        lock (asyncio.Lock): Async lock for thread-safe token operations.

    Example:
        >>> # Limit to 60 requests per minute (1 request per second)
        >>> rpm_limiter = AsyncTokenBucket(rate_limit=1.0, max_tokens=1.0)
        >>> await rpm_limiter.acquire(1.0)  # Consumes 1 token
        >>>
        >>> # Limit to 10000 tokens per minute (~166.67 tokens per second)
        >>> tpm_limiter = AsyncTokenBucket(rate_limit=166.67, max_tokens=166.67)
        >>> await tpm_limiter.acquire(100.0)  # Consumes 100 tokens

    Thread Safety:
        All operations are protected by an asyncio.Lock, making this class safe
        for concurrent use across multiple coroutines.

    Algorithm Details:
        1. On each acquire(), calculate elapsed time since last update
        2. Refill tokens: tokens += elapsed * rate_limit (capped at max_tokens)
        3. If tokens >= num_tokens: consume tokens and return
        4. Otherwise: calculate wait_time = tokens_needed / rate_limit, then sleep
        5. Retry after sleep (loop back to step 1)
    """

    def __init__(self, rate_limit: float, max_tokens: float = None):
        self.rate_limit = rate_limit
        self.max_tokens = max_tokens or rate_limit
        self.tokens = self.max_tokens
        self.last_update = None
        self.lock = asyncio.Lock()

    async def acquire(self, num_tokens: float = 1.0) -> None:
        """Acquire tokens from the bucket, waiting if necessary.

        This method will block (using asyncio.sleep) until sufficient tokens
        are available. It automatically refills tokens based on elapsed time
        and the configured rate_limit.

        For requests exceeding max_tokens, the method will wait for enough time
        to accumulate the required tokens at the configured rate_limit, allowing
        tokens to temporarily go negative.

        Args:
            num_tokens (float): Number of tokens to consume. Defaults to 1.0.
                Can be fractional for fine-grained rate limiting.

        Returns:
            None: Returns when tokens have been successfully acquired.

        Raises:
            No exceptions are raised. This method will wait indefinitely until
            tokens become available.

        Example:
            >>> bucket = AsyncTokenBucket(rate_limit=10.0)
            >>> await bucket.acquire(5.0)  # Acquire 5 tokens
            >>> await bucket.acquire(1.0)  # Acquire 1 more token

        Implementation Notes:
            - Uses event loop's time() for high-precision timestamps
            - Lock is released during sleep to allow other coroutines to proceed
            - Tokens are refilled continuously based on elapsed time
            - For requests > max_tokens, allows temporary negative balance
        """
        # Handle requests larger than max_tokens separately
        if num_tokens > self.max_tokens:
            wait_time = 0.0
            async with self.lock:
                loop = get_event_loop()
                now = loop.time()
                if self.last_update is None:
                    self.last_update = now

                elapsed = now - self.last_update
                new_tokens = elapsed * self.rate_limit
                self.tokens = min(self.max_tokens, self.tokens + new_tokens)

                tokens_needed = num_tokens - self.tokens
                if tokens_needed > 0:
                    wait_time = tokens_needed / self.rate_limit

                self.tokens -= num_tokens
                self.last_update = now

            if wait_time > 0:
                await asyncio.sleep(wait_time)
            return

        # Standard case: request <= max_tokens
        while True:
            wait_time = 0.0
            async with self.lock:
                loop = get_event_loop()
                now = loop.time()
                if self.last_update is None:
                    self.last_update = now

                elapsed = now - self.last_update
                new_tokens = elapsed * self.rate_limit
                self.tokens = min(self.max_tokens, self.tokens + new_tokens)
                self.last_update = now

                if self.tokens >= num_tokens:
                    self.tokens -= num_tokens
                    return

                tokens_needed = num_tokens - self.tokens
                wait_time = tokens_needed / self.rate_limit

            if wait_time > 0:
                await asyncio.sleep(wait_time)


@register_manager("rate_limited")
@register_manager_legacy("rate_limited")
class RateLimitedRewardManager(RewardManagerBase):


    # Class-level state for global rate limiting
    _semaphore = None
    _max_concurrent = None
    _rpm_limiter = None
    _max_rpm = None
    _tpm_limiter = None
    _max_tpm = None
    _estimated_tokens_per_request = None
    _class_initialized = False

    @classmethod
    def init_class(cls, config: DictConfig, tokenizer: AutoTokenizer):
        """Initialize class state shared across all instances."""
        # Check if already initialized before calling parent.
        #
        # NOTE: This class owns a *global*, class-level set of rate limiters. Once the class has been
        # initialized, subsequent instantiations cannot change the shared limiters. This is by design,
        # but it can be surprising (and dangerous) when the first initialization happens with default
        # values (often "unlimited") and later code tries to apply limits.
        if cls._class_initialized:
            rm_cfg = config.get("reward") or {}
            incoming_max_rpm = rm_cfg.get("max_rpm", None)
            incoming_max_tpm = rm_cfg.get("max_tpm", None)

            # Warn when a caller is trying to change the global RPM/TPM limits after initialization.
            # This commonly happens if the first instance was created without a config (legacy signature),
            # which initializes the global limiters to their defaults and locks them in.
            if (incoming_max_rpm != cls._max_rpm) or (incoming_max_tpm != cls._max_tpm):
                if (
                    incoming_max_rpm is not None
                    or incoming_max_tpm is not None
                    or cls._max_rpm is not None
                    or cls._max_tpm is not None
                ):
                    logger.warning(
                        "RateLimitedRewardManager has already been initialized and its rate limiters are shared "
                        "globally across instances. The incoming (max_rpm/max_tpm) settings will be ignored. "
                        "This can lead to unexpected behavior (e.g., exceeding API rate limits) if the first "
                        "initialization used defaults (often unlimited). "
                        f"Existing: max_rpm={cls._max_rpm}, max_tpm={cls._max_tpm}. "
                        f"Incoming: max_rpm={incoming_max_rpm}, max_tpm={incoming_max_tpm}. "
                        "To apply different limits, ensure the first RateLimitedRewardManager created in this "
                        "process uses the desired configuration (or restart/reset the process)."
                    )
            return

        super().init_class(config, tokenizer)

        rm_cfg = config.get("reward") or {}

        # Concurrency limiter
        cls._max_concurrent = rm_cfg.get("max_concurrent", 1)
        cls._semaphore = asyncio.Semaphore(cls._max_concurrent)

        # Request rate limiter (RPM)
        cls._max_rpm = rm_cfg.get("max_rpm", None)
        if cls._max_rpm is not None:
            requests_per_second = cls._max_rpm / 60.0
            cls._rpm_limiter = AsyncTokenBucket(rate_limit=requests_per_second, max_tokens=requests_per_second)
        else:
            cls._rpm_limiter = None

        # Token rate limiter (TPM)
        cls._max_tpm = rm_cfg.get("max_tpm", None)
        cls._estimated_tokens_per_request = rm_cfg.get("estimated_tokens_per_request", 2000)
        if cls._max_tpm is not None:
            tokens_per_second = cls._max_tpm / 60.0
            cls._tpm_limiter = AsyncTokenBucket(rate_limit=tokens_per_second, max_tokens=tokens_per_second)
        else:
            cls._tpm_limiter = None

        log_msg = "Rate limiting configuration:\n"
        log_msg += f"  - Concurrency limit: {cls._max_concurrent}\n"
        if cls._max_rpm is not None:
            log_msg += f"  - Request rate limit: {cls._max_rpm} RPM ({cls._max_rpm / 60.0:.2f} RPS)\n"
        else:
            log_msg += "  - Request rate limit: unlimited\n"
        if cls._max_tpm is not None:
            log_msg += f"  - Token rate limit: {cls._max_tpm} TPM ({cls._max_tpm / 60.0:.2f} TPS)\n"
            log_msg += f"  - Estimated tokens per request: {cls._estimated_tokens_per_request}\n"
        else:
            log_msg += "  - Token rate limit: unlimited\n"
        log_msg += "All limiters are shared globally across all workers."
        logger.info(log_msg)

        cls._class_initialized = True

    def __init__(
        self,
        config,
        tokenizer,
        compute_score,
        reward_router_address=None,
        reward_model_tokenizer=None,
        # Legacy (AbstractRewardManager) kwargs for compatibility. Not used.
        num_examine: int | None = None,
        reward_fn_key: str | None = None,
        **kwargs,
    ):
        # When called via the legacy AbstractRewardManager signature, `config` may be absent.
        # In that case we fall back to an empty config so training can proceed.
        if config is None:
            config = DictConfig({"reward": {}})
        if tokenizer is None:
            raise TypeError("RateLimitedRewardManager requires `tokenizer`.")

        super().__init__(config, tokenizer, compute_score)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer
        self.timeout = config.reward.get("timeout", 300.0)

    async def _compute_reward(
        self, data_source: str, solution_str: str, ground_truth: str, extra_info: dict
    ) -> dict | float:
        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )
        if self.is_async_reward_score:
            return await self.compute_score(
                data_source=data_source,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **extra_reward_kwargs,
            )
        else:
            return await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_source=data_source,
                    solution_str=solution_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **extra_reward_kwargs,
                ),
            )

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]

        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)
        if tool_extra_fields is not None:
            extra_info.update(tool_extra_fields.items())

        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        reward_extra_info = {}

        # Apply rate limiting layers
        if self._rpm_limiter is not None:
            await self._rpm_limiter.acquire(1.0)

        if self._tpm_limiter is not None:
            estimated_tokens = self._estimated_tokens_per_request
            await self._tpm_limiter.acquire(estimated_tokens)

        async with self._semaphore:
            try:
                result = await asyncio.wait_for(
                    self._compute_reward(
                        data_source=data_source,
                        solution_str=response_str,
                        ground_truth=ground_truth,
                        extra_info=extra_info,
                    ),
                    timeout=self.timeout,
                )

                score: float
                if isinstance(result, dict):
                    score = result["score"]
                    for key, value in result.items():
                        reward_extra_info[key] = value
                else:
                    score = result
                    reward_extra_info["acc"] = score

                reward = score

            except asyncio.TimeoutError:
                logger.warning(
                    f"Reward computation timed out after {self.timeout}s for data_source={data_source}. "
                    f"Response preview: {response_str[:100]}..."
                )
                reward = 0.0
                reward_extra_info["timeout"] = True
                reward_extra_info["acc"] = 0.0

            except Exception as e:
                logger.error(
                    f"Reward computation failed for data_source={data_source}: {e}. "
                    f"Response preview: {response_str[:100]}..."
                )
                reward = 0.0
                reward_extra_info["error"] = str(e)
                reward_extra_info["acc"] = 0.0

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}

    def __call__(self, data: DataProto, return_dict: bool = False):
        from collections import defaultdict

        import torch

        # If there are pre-computed rm_scores, return them directly
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        # Initialize reward tensor
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        # Process each data item through the async event loop
        async def process_batch():
            tasks = []
            for i in range(len(data)):
                data_item = data[i : i + 1]  # Get single item as DataProto slice
                tasks.append(self.run_single(data_item))

            results = await asyncio.gather(*tasks)
            return results

        # Run the async processing using self.loop property which lazily gets/creates event loop
        # This ensures rate limiters and semaphores work correctly by using the same loop
        results = self.loop.run_until_complete(process_batch())

        # Aggregate results into reward tensor and extra info
        for i, result in enumerate(results):
            data_item = data[i]
            response_ids = data_item.batch["responses"]
            response_length = response_ids.shape[-1]
            valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()

            reward = result["reward_score"]
            reward_tensor[i, valid_response_length - 1] = reward

            # Collect extra info
            if "reward_extra_info" in result:
                for key, value in result["reward_extra_info"].items():
                    reward_extra_info[key].append(value)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
