
try:
    import hydra
except ImportError as e:
    raise ImportError("Please install hydra-core via 'pip install hydra-core' and retry.") from e


@hydra.main(config_path="../verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    print(config)
    from verl.utils.config import omega_conf_to_dataclass

    profiler_config = omega_conf_to_dataclass(config.critic.profiler)
    print(profiler_config)


if __name__ == "__main__":
    main()
