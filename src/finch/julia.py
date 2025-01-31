import os

import juliapkg


def add_package(name: str, hash: str, version: str) -> None:
    deps = juliapkg.deps.load_cur_deps()
    if deps.get("packages", {}).get(name, {}).get("version", None) != version:
        juliapkg.add(name, hash, version=version)
        juliapkg.resolve()


_FINCH_NAME = "Finch"
_FINCH_VERSION = "1.2.0"
_FINCH_HASH = "9177782c-1635-4eb9-9bfb-d9dfa25e6bce"
_FINCH_REPO_PATH = os.environ.get("FINCH_REPO_PATH", default=None)
_FINCH_REPO_URL = os.environ.get("FINCH_URL_PATH", default=None)

if _FINCH_REPO_PATH and _FINCH_REPO_URL:
    raise ValueError(
        "FINCH_REPO_PATH and FINCH_URL_PATH can't be set at the same time."
    )

if _FINCH_REPO_PATH:  # Also account for empty string
    juliapkg.add(_FINCH_NAME, _FINCH_HASH, path=_FINCH_REPO_PATH, dev=True)
elif _FINCH_REPO_URL:
    juliapkg.add(_FINCH_NAME, _FINCH_HASH, url=_FINCH_REPO_URL, dev=True)
else:
    add_package(_FINCH_NAME, _FINCH_HASH, _FINCH_VERSION)

import juliacall as jc  # noqa

juliapkg.resolve()
from juliacall import Main as jl  # noqa

jl.seval("using Finch")
jl.seval("using Random")
