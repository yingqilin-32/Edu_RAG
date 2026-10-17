import pytest

from edu_rag.config import ConfigurationError, Settings


def test_missing_settings_report_names_without_credentials(monkeypatch):
    for name in ("DB_HOST", "DB_USER", "DB_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ConfigurationError, match="DB_HOST, DB_USER, DB_PASSWORD"):
        Settings(_env_file=None).database_url()


def test_password_special_characters_are_preserved_but_redacted():
    password = "demo@pass:#/? with spaces"
    settings = Settings(
        _env_file=None,
        db_host="example.invalid",
        db_user="postgres.demo",
        db_password=password,
    )
    url = settings.database_url()
    assert url.password == password
    assert password not in str(url)
    assert password not in repr(settings)

