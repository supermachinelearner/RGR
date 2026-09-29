

from .base_model_merger import generate_config_from_args, parse_args


def main():
    args = parse_args()
    config = generate_config_from_args(args)
    print(f"config: {config}")

    if config.backend == "fsdp":
        from .fsdp_model_merger import FSDPModelMerger

        merger = FSDPModelMerger(config)
    elif config.backend == "megatron":
        from .megatron_model_merger import MegatronModelMerger

        merger = MegatronModelMerger(config)
    else:
        raise NotImplementedError(f"Unknown backend: {config.backend}")

    merger.merge_and_save()
    merger.cleanup()


if __name__ == "__main__":
    main()
