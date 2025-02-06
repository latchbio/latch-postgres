@default:
  just --list --unsorted

publish:
  #!/usr/bin/env bash
  uv publish \
    --token "$(cat credentials/pypi.txt)"
