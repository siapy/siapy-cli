from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

DOTENV_PATH = ".env"

BASE_DIR = Path(__file__).parent.parent.parent.absolute()
SAVED_DIR = BASE_DIR / "saved"

SAVED_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(override=True, dotenv_path=DOTENV_PATH)


class Settings(BaseSettings):
    images_dir: str = ""
    debug: bool = False

    model_config = SettingsConfigDict(
        env_file=DOTENV_PATH,
        env_file_encoding="utf-8",
        cli_parse_args=False,
    )


settings = Settings()
