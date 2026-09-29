

from dataclasses import dataclass, field
from typing import Any, Optional

from verl.base_config import BaseConfig

__all__ = ["AlgoConfig", "FilterGroupsConfig", "KLControlConfig", "RolloutCorrectionConfig"]


@dataclass
class KLControlConfig(BaseConfig):
    """Configuration for KL control.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        type (str): Type of KL control. Can be "fixed" or "adaptive".
        kl_coef (float): Initial coefficient for KL penalty.
        horizon (int): Horizon value for adaptive controller.
        target_kl (float): Target KL divergence for adaptive controller.
    """

    type: str = "fixed"
    kl_coef: float = 0.001
    horizon: int = 10000
    target_kl: float = 0.1


@dataclass
class FilterGroupsConfig(BaseConfig):


    enable: bool = False
    metric: Optional[str] = None
    max_num_gen_batches: int = 0


@dataclass
class RolloutCorrectionConfig(BaseConfig):


    rollout_is: Optional[str] = "sequence"
    rollout_is_threshold: str | float = 2.0
    rollout_is_batch_normalize: bool = False
    rollout_rs: Optional[str] = None
    rollout_rs_threshold: Optional[str | float] = None
    bypass_mode: bool = False
    loss_type: str = "ppo_clip"

    @classmethod
    def decoupled_token_is(cls, threshold: float = 2.0) -> "RolloutCorrectionConfig":

        return cls(rollout_is="token", rollout_is_threshold=threshold, rollout_rs=None)

    @classmethod
    def decoupled_seq_is(cls, threshold: float = 2.0) -> "RolloutCorrectionConfig":
        """Decoupled Mode with Sequence-level Importance Sampling.

        IS weight correction at sequence level in decoupled mode (three policies).

        Args:
            threshold (float): Upper threshold for IS weights. Default: 2.0

        Returns:
            RolloutCorrectionConfig configured for decoupled mode with sequence-level IS
        """
        return cls(rollout_is="sequence", rollout_is_threshold=threshold, rollout_rs=None)

    @classmethod
    def decoupled_token_icepop(
        cls,
        threshold: float = 5.0,
        threshold_lower: float = 0.5,
    ) -> "RolloutCorrectionConfig":
        """Decoupled Mode with exact token-level IcePop.

        Keeping response_mask unchanged and
        zeroing token IS weights outside
        [threshold_lower, threshold].

        Args:
            threshold (float): Upper IcePop bound. Default: 5.0
            threshold_lower (float): Lower IcePop bound. Default: 0.5

        Returns:
            RolloutCorrectionConfig configured for decoupled mode with token-level IcePop
        """
        return cls(rollout_is="token", rollout_is_threshold=f"{threshold_lower}_{threshold}", rollout_rs=None)

    @classmethod
    def decoupled_seq_is_rs(
        cls,
        is_threshold: float = 2.0,
        rs_threshold: Optional[str | float] = "0.5_2.0",
    ) -> "RolloutCorrectionConfig":
        """Decoupled Mode with Sequence-level IS + Rejection Sampling.

        Sequence-level IS with sequence-level rejection sampling in decoupled mode.
        Rejects entire sequences based on sequence-level IS weight.

        Args:
            is_threshold (float): Upper threshold for IS weights. Default: 2.0
            rs_threshold (Optional[Union[str, float]]): Upper threshold for rejection sampling. Default: 0.5_2.0

        Returns:
            RolloutCorrectionConfig configured for decoupled mode with sequence IS + RS
        """
        return cls(
            rollout_is="sequence",
            rollout_is_threshold=is_threshold,
            rollout_rs="seq_sum_k1",
            rollout_rs_threshold=rs_threshold,
        )

    @classmethod
    def decoupled_geo_rs(
        cls,
        rs_threshold: Optional[str | float] = "0.999_1.001",
    ) -> "RolloutCorrectionConfig":
        """Decoupled Mode with Geometric Mean Rejection Sampling (ratio-based).

        Uses geometric mean IS ratio E[log(r)] for rejection sampling at sequence level.
        This is a ratio-based mode (ideal = 0.0) with [lower, upper] threshold bounds.
        Length-normalized but still uses IS ratio semantics.

        Args:
            rs_threshold (Optional[Union[str, float]]): Geometric RS threshold (upper). Default: 0.999_1.001 (±0.1%)

        Returns:
            RolloutCorrectionConfig configured for decoupled mode with Geo-RS
        """
        return cls(
            rollout_is=None,
            rollout_rs="seq_mean_k1",
            rollout_rs_threshold=rs_threshold,
        )

    @classmethod
    def bypass_ppo_clip(cls) -> "RolloutCorrectionConfig":
        """Bypass mode with PPO-clip loss.

        PPO clipped objective in bypass mode. The PPO ratio = π_θ/π_rollout
        already handles IS correction, so no explicit IS weights are applied.

        Skips old_log_prob computation for faster execution (2 policies instead of 3).

        Returns:
            RolloutCorrectionConfig configured for bypass mode with PPO-clip
        """
        return cls(
            rollout_is=None,
            rollout_rs=None,
            bypass_mode=True,
            loss_type="ppo_clip",
        )

    @classmethod
    def bypass_ppo_clip_geo_rs(
        cls,
        rs_threshold: Optional[str | float] = "0.999_1.001",
    ) -> "RolloutCorrectionConfig":
        """Bypass mode with PPO-clip loss and Geometric Mean RS (ratio-based).

        PPO clipped objective in bypass mode with geometric mean IS ratio RS.
        Uses E[log(r)] (ideal = 0.0) with [lower, upper] threshold bounds.

        Args:
            rs_threshold (Optional[Union[str, float]]): Geometric RS threshold (upper). Default: 0.999_1.001 (±0.1%)

        Returns:
            RolloutCorrectionConfig configured for bypass mode with PPO-clip + Geo-RS
        """
        return cls(
            rollout_is=None,
            rollout_rs="seq_mean_k1",
            rollout_rs_threshold=rs_threshold,
            bypass_mode=True,
            loss_type="ppo_clip",
        )

    @classmethod
    def bypass_ppo_clip_k3_rs(
        cls,
        rs_threshold: float = 0.01,
    ) -> "RolloutCorrectionConfig":
        """Bypass mode with PPO-clip loss and K3 Rejection Sampling.

        PPO clipped objective in bypass mode with K3 KL estimator RS to mask outliers.
        K3 is more stable than K1 for small KL values.
        The PPO ratio = π_θ/π_rollout already handles IS correction.

        Args:
            rs_threshold (float): Max allowed K3 divergence. Default: 0.01

        Returns:
            RolloutCorrectionConfig configured for bypass mode with PPO-clip + K3-RS
        """
        return cls(
            rollout_is=None,
            rollout_rs="seq_mean_k3",
            rollout_rs_threshold=rs_threshold,
            bypass_mode=True,
            loss_type="ppo_clip",
        )

    @classmethod
    def bypass_pg_is(cls, threshold: float = 2.0) -> "RolloutCorrectionConfig":
        """Bypass mode with REINFORCE loss and IS Correction.

        Uses REINFORCE loss with explicit IS correction in bypass mode.
        No PPO clipping.

        Args:
            threshold (float): Upper threshold for IS weights. Default: 2.0

        Returns:
            RolloutCorrectionConfig configured for bypass mode with REINFORCE + IS
        """
        return cls(
            rollout_is="sequence",
            rollout_is_threshold=threshold,
            rollout_rs=None,
            bypass_mode=True,
            loss_type="reinforce",
        )

    @classmethod
    def bypass_pg_token_icepop(
        cls,
        threshold: float = 5.0,
        threshold_lower: float = 0.5,
    ) -> "RolloutCorrectionConfig":
        """Bypass mode with REINFORCE loss and exact token-level IcePop.

        Uses explicit IS weights in bypass mode and zeroes out token weights
        outside [threshold_lower, threshold] without modifying response_mask.

        Args:
            threshold (float): Upper IcePop bound. Default: 5.0
            threshold_lower (float): Lower IcePop bound. Default: 0.5

        Returns:
            RolloutCorrectionConfig configured for bypass mode with REINFORCE + token-level IcePop
        """
        return cls(
            rollout_is="token",
            rollout_is_threshold=f"{threshold_lower}_{threshold}",
            rollout_rs=None,
            bypass_mode=True,
            loss_type="reinforce",
        )

    @classmethod
    def bypass_pg_geo_rs(
        cls,
        rs_threshold: Optional[str | float] = "0.999_1.001",
    ) -> "RolloutCorrectionConfig":
        """Bypass mode with REINFORCE loss and Geometric Mean RS (ratio-based).

        REINFORCE with geometric mean IS ratio rejection sampling in bypass mode.
        Uses E[log(r)] (ideal = 0.0) with [lower, upper] threshold bounds.

        Args:
            rs_threshold (Optional[Union[str, float]]): Geometric RS threshold (upper). Default: 0.999_1.001 (±0.1%)

        Returns:
            RolloutCorrectionConfig configured for bypass mode with REINFORCE + Geo-RS
        """
        return cls(
            rollout_is=None,
            rollout_rs="seq_mean_k1",
            rollout_rs_threshold=rs_threshold,
            bypass_mode=True,
            loss_type="reinforce",
        )

    @classmethod
    def decoupled_geo_rs_seq_tis(
        cls,
        is_threshold: float = 2.0,
        rs_threshold: Optional[str | float] = "0.999_1.001",
    ) -> "RolloutCorrectionConfig":

        return cls(
            rollout_is="sequence",
            rollout_is_threshold=is_threshold,
            rollout_rs="seq_mean_k1",
            rollout_rs_threshold=rs_threshold,
        )

    @classmethod
    def decoupled_geo_rs_token_tis(
        cls,
        is_threshold: float = 2.0,
        rs_threshold: Optional[str | float] = "0.999_1.001",
    ) -> "RolloutCorrectionConfig":
        """Decoupled mode with Geometric Mean RS and Token-level Truncated IS (ratio-based).

        Combines the Geometric Mean Filter (ratio-based validity check) with
        Token-level IS weights. Uses E[log(r)] (ideal = 0.0).

        Args:
            is_threshold (float): Upper threshold for token IS weights. Default: 2.0
            rs_threshold (Optional[Union[str, float]]): Geometric RS threshold (upper). Default: 0.999_1.001 (±0.1%)

        Returns:
            RolloutCorrectionConfig configured for Geo-RS-Token-TIS
        """
        return cls(
            rollout_is="token",
            rollout_is_threshold=is_threshold,
            rollout_rs="seq_mean_k1",
            rollout_rs_threshold=rs_threshold,
        )

    @classmethod
    def bypass_pg_geo_rs_seq_tis(
        cls,
        is_threshold: float = 2.0,
        rs_threshold: Optional[str | float] = "0.999_1.001",
    ) -> "RolloutCorrectionConfig":
        """Bypass mode with REINFORCE loss, Geo-RS, and Sequence-level IS.

        Combines geometric mean IS ratio rejection with sequence-level IS
        in bypass mode with REINFORCE loss (no PPO clipping).
        Uses E[log(r)] (ideal = 0.0) with [lower, upper] threshold bounds.

        Args:
            is_threshold (float): Upper threshold for sequence IS weights. Default: 2.0
            rs_threshold (Optional[Union[str, float]]): Geometric RS threshold (upper). Default: 0.999_1.001 (±0.1%)

        Returns:
            RolloutCorrectionConfig configured for bypass mode with REINFORCE + Geo-RS + Seq-TIS
        """
        return cls(
            rollout_is="sequence",
            rollout_is_threshold=is_threshold,
            rollout_rs="seq_mean_k1",
            rollout_rs_threshold=rs_threshold,
            bypass_mode=True,
            loss_type="reinforce",
        )

    @classmethod
    def bypass_pg_geo_rs_token_tis(
        cls,
        is_threshold: float = 2.0,
        rs_threshold: Optional[str | float] = "0.999_1.001",
    ) -> "RolloutCorrectionConfig":
        """Bypass mode with REINFORCE loss, Geo-RS, and Token-level IS.

        Combines geometric mean IS ratio rejection with token-level IS weights
        in bypass mode with REINFORCE loss (no PPO clipping).
        Uses E[log(r)] (ideal = 0.0) with [lower, upper] threshold bounds.

        Token-level IS has lower variance but introduces bias.

        Args:
            is_threshold (float): Upper threshold for token IS weights. Default: 2.0
            rs_threshold (Optional[Union[str, float]]): Geometric RS threshold (upper). Default: 0.999_1.001 (±0.1%)

        Returns:
            RolloutCorrectionConfig configured for bypass mode with REINFORCE + Geo-RS + Token-TIS
        """
        return cls(
            rollout_is="token",
            rollout_is_threshold=is_threshold,
            rollout_rs="seq_mean_k1",
            rollout_rs_threshold=rs_threshold,
            bypass_mode=True,
            loss_type="reinforce",
        )

    @classmethod
    def decoupled_k3_rs(
        cls,
        rs_threshold: float = 0.01,
    ) -> "RolloutCorrectionConfig":
        """Decoupled mode with K3 KL Estimator Rejection Sampling.

        Uses K3 KL estimator at sequence level for rejection sampling.
        K3 = E[r - log(r) - 1] where r = π_train/π_rollout.
        More stable than geometric mean for small KL values.

        K3 >= 0 always (equals 0 when policies match exactly).

        Args:
            rs_threshold (float): Max allowed K3 divergence. Default: 0.01
                Typical range: 0.001-0.1

        Returns:
            RolloutCorrectionConfig configured for K3 RS
        """
        return cls(
            rollout_is=None,
            rollout_rs="seq_mean_k3",
            rollout_rs_threshold=rs_threshold,
        )

    @classmethod
    def decoupled_k3_rs_seq_tis(
        cls,
        is_threshold: float = 2.0,
        rs_threshold: float = 0.01,
    ) -> "RolloutCorrectionConfig":
        """Decoupled mode with K3 RS and Sequence-level Truncated IS.

        Combines K3 KL estimator rejection with sequence-level IS weights.
        K3 provides more stable outlier detection than geometric mean.

        Args:
            is_threshold (float): Upper threshold for sequence IS weights. Default: 2.0
            rs_threshold (float): Max allowed K3 divergence. Default: 0.01

        Returns:
            RolloutCorrectionConfig configured for K3-RS-Seq-TIS
        """
        return cls(
            rollout_is="sequence",
            rollout_is_threshold=is_threshold,
            rollout_rs="seq_mean_k3",
            rollout_rs_threshold=rs_threshold,
        )

    @classmethod
    def decoupled_k3_rs_token_tis(
        cls,
        is_threshold: float = 2.0,
        rs_threshold: float = 0.01,
    ) -> "RolloutCorrectionConfig":
        """Decoupled mode with K3 RS and Token-level Truncated IS.

        Combines K3 KL estimator rejection with token-level IS weights.
        K3 provides more stable outlier detection than geometric mean.
        Token-level IS has lower variance but introduces bias.

        Args:
            is_threshold (float): Upper threshold for token IS weights. Default: 2.0
            rs_threshold (float): Max allowed K3 divergence. Default: 0.01

        Returns:
            RolloutCorrectionConfig configured for K3-RS-Token-TIS
        """
        return cls(
            rollout_is="token",
            rollout_is_threshold=is_threshold,
            rollout_rs="seq_mean_k3",
            rollout_rs_threshold=rs_threshold,
        )

    @classmethod
    def disabled(cls) -> "RolloutCorrectionConfig":
        """Disabled - Metrics Only Mode.

        Computes and logs off-policy metrics without applying correction.

        Returns:
            RolloutCorrectionConfig with all correction disabled
        """
        return cls(rollout_is=None, rollout_rs=None)


@dataclass
class AlgoConfig(BaseConfig):
    """Configuration for the algorithm.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        gamma (float): Discount factor for future rewards.
        lam (float): Trade-off between bias and variance in the GAE estimator.
        adv_estimator (str): Advantage estimator type: "gae", "grpo", "reinforce_plus_plus", etc.
        norm_adv_by_std_in_grpo (bool): Whether to normalize advantages by std (specific to GRPO).
        use_kl_in_reward (bool): Whether to enable in-reward KL penalty.
        kl_penalty (str): How to estimate KL divergence: "kl", "abs", "mse", "low_var_kl", or "full".
        kl_ctrl (KLControlConfig): KL control configuration.
        use_pf_ppo (bool): Whether to enable preference feedback PPO.
        pf_ppo (dict[str, Any]): Preference feedback PPO settings.
        filter_groups (Optional[FilterGroupsConfig]): Filter groups configuration, used in DAPO and Entropy
        rollout_correction (Optional[RolloutCorrectionConfig]): Rollout Correction configuration.
            Addresses off-policy issues from policy mismatch, model staleness, and general distribution shifts.

            Set to None to disable entirely. Use factory methods for common presets:
            - RolloutCorrectionConfig.decoupled_token_is() - Decoupled mode with token-level IS
            - RolloutCorrectionConfig.decoupled_seq_is() - Decoupled mode with sequence-level IS
            - RolloutCorrectionConfig.decoupled_seq_is_rs() - Decoupled mode with sequence IS + RS
            - RolloutCorrectionConfig.decoupled_k1_rs() - Decoupled mode with K1-RS (divergence)
            - RolloutCorrectionConfig.decoupled_geo_rs() - Decoupled mode with Geo-RS (ratio)
            - RolloutCorrectionConfig.bypass_ppo_clip() - Bypass mode with PPO-clip
            - RolloutCorrectionConfig.bypass_ppo_clip_k1_rs() - Bypass mode with PPO-clip + K1-RS
            - RolloutCorrectionConfig.bypass_pg_is() - Bypass mode with REINFORCE + IS
            - RolloutCorrectionConfig.bypass_pg_k1_rs() - Bypass mode with REINFORCE + K1-RS

            For backward compatibility, you can still pass a dict, which will be converted to
            RolloutCorrectionConfig automatically.
    """

    gamma: float = 1.0
    lam: float = 1.0
    adv_estimator: str = "gae"
    norm_adv_by_std_in_grpo: bool = True
    use_kl_in_reward: bool = False
    kl_penalty: str = "kl"
    kl_ctrl: KLControlConfig = field(default_factory=KLControlConfig)
    use_pf_ppo: bool = False
    pf_ppo: dict[str, Any] = field(default_factory=dict)
    filter_groups: Optional[FilterGroupsConfig] = None
    # Rollout Correction: corrects off-policy issues (policy mismatch, model staleness, distribution shifts)
    # Set to None to disable, use RolloutCorrectionConfig presets (e.g., .tis(), .mis()), or pass dict
    rollout_correction: Optional[RolloutCorrectionConfig] = None
    # GDPO (Group reward-Decoupled Normalization Policy Optimization) settings.
    # gdpo_reward_keys: keys in non_tensor_batch (from compute_score's return dict) that
    #   correspond to individual reward dimensions, e.g. ["format_reward", "accuracy_reward"].
    # gdpo_reward_weights: per-dimension weights for aggregation (default: equal weights).
    gdpo_reward_keys: Optional[list[str]] = None
    gdpo_reward_weights: Optional[list[float]] = None
