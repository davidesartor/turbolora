"""TuRBO training of TinyLoRA: train_bo's objective, searched by turbo.search instead of global Thompson sampling."""

from turbolora import train_bo, turbo


def main() -> None:
    args = turbo.add_arguments(train_bo.argument_parser()).parse_args()
    train_bo.run(args, search=turbo.search, loss="turbo")


if __name__ == "__main__":
    main()
