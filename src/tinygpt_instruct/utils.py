from pathlib import Path


def get_models_dir() -> Path:
    """Возвращает путь к директории models.

    Returns:
        Path: Путь к директории models.
    """
    root_dir = Path(__file__).parent.parent.parent
    model_dir = root_dir / "models"
    model_dir.mkdir(exist_ok=True)
    return model_dir
