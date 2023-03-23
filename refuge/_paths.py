# Refuge, to inspire every business owner with the Gospel and empower them with AI.
# Copyright (C) Alex Carpenter and Simon Biggs

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import pathlib

LIB = pathlib.Path(__file__).parent

STORE = pathlib.Path.home().joinpath(".refuge")
CONFIG = STORE.joinpath("config")
SECRETS = CONFIG.joinpath("secrets")

USERS = STORE.joinpath("users")
EMAIL_MAPPING = USERS.joinpath("email-mapping")
USER_DETAILS = USERS.joinpath("details")
AGENT_MAPPING = USERS.joinpath("agent-mapping")
FORM_DATA = USERS.joinpath("forms")

RECORDS = STORE.joinpath("records")

PROMPTS = RECORDS.joinpath("prompts")
COMPLETIONS = RECORDS.joinpath("completions")
ARTICLE_METADATA = RECORDS.joinpath("article-metadata")
DOWNLOADED_ARTICLES = RECORDS.joinpath("downloaded-articles")
EMAILS = RECORDS.joinpath("emails")
COMPLETION_CACHE = RECORDS.joinpath("completion-cache")

PIPELINES = STORE.joinpath("pipelines")

GOOGLE_ALERTS_PIPELINES = PIPELINES.joinpath("google-alerts")
NEW_GOOGLE_ALERTS = GOOGLE_ALERTS_PIPELINES.joinpath("new")

EMAIL_PIPELINES = PIPELINES.joinpath("emails")
NEW_EMAILS = EMAIL_PIPELINES.joinpath("new")

LOGS = STORE.joinpath("server", "logs")
PHIRHO_LOGS = LOGS.joinpath("phirho")

TEST_DIR = LIB.joinpath("tests")
TESTS_DATA = TEST_DIR.joinpath("data")

FORM_TEMPLATES = CONFIG.joinpath("form-templates")
FAQ_DATA = CONFIG.joinpath("faq")

AI_DIR = LIB.joinpath("_ai")
AI_REGISTRY_DIR = AI_DIR.joinpath("registry")


def get_article_metadata_path(hash_digest: str, create_parent: bool = False):
    path = _get_record_path(ARTICLE_METADATA, hash_digest, create_parent)

    return path


def get_downloaded_article_path(hash_digest: str, create_parent: bool = False):
    path = _get_record_path(DOWNLOADED_ARTICLES, hash_digest, create_parent)

    return path


def get_emails_path(hash_digest: str, create_parent: bool = False):
    path = _get_record_path(EMAILS, hash_digest, create_parent)

    return path


def get_completion_cache_path(hash_digest: str, create_parent: bool = False):
    path = _get_record_path(COMPLETION_CACHE, hash_digest, create_parent)

    return path


def _get_record_path(root: pathlib.Path, hash_digest: str, create_parent: bool):
    path = root / _get_relative_json_path(hash_digest)

    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)

    return path


def _get_relative_json_path(hash_digest: str):
    return pathlib.Path(hash_digest[0:4]) / hash_digest[4:8] / f"{hash_digest}.json"
