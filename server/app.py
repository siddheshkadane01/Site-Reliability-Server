from main import app


def main() -> None:
    """Entry point required by OpenEnv validator scripts."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
