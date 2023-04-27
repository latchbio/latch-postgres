@default:
  just --list --unsorted

local_install:
  python -m pip install -e .

build:
  poetry build

publish:
  poetry publish --build


