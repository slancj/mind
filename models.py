import litellm


def main() -> None:
    models = sorted(litellm.model_list)
    for model in models:
        print(model)
    print(f"\n{len(models)} models")


if __name__ == "__main__":
    main()