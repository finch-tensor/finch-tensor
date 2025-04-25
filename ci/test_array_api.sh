#!/usr/bin/env bash
set -euxo pipefail

source ci/clone_array_api_tests.sh

python -c 'import finch'
ARRAY_API_TESTS_MODULE="finch" pytest "$ARRAY_API_TESTS_DIR/array_api_tests/" -v -c "$ARRAY_API_TESTS_DIR/pytest.ini" --ci --max-examples=2 --derandomize --disable-deadline --disable-warnings -n auto --skips-file ci/array-api-skips.txt
