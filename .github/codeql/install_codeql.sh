#!/usr/bin/env bash
set -euo pipefail

# Install the CodeQL CLI locally for developer workflows.
# Environment variables:
# - CODEQL_VERSION (default: 2.24.2)
# - CODEQL_INSTALL_DIR (default: /tmp/codeql)

CODEQL_VERSION="${CODEQL_VERSION:-2.24.2}"
CODEQL_INSTALL_DIR="${CODEQL_INSTALL_DIR:-/tmp/codeql}"

if [[ -x "${CODEQL_INSTALL_DIR}/codeql" ]]; then
    echo "CodeQL already installed at ${CODEQL_INSTALL_DIR}/codeql"
    exit 0
fi

OS="$(uname -s)"
case "${OS}" in
    Linux)
        ASSET="codeql-linux64.zip"
        ;;
    Darwin)
        # CodeQL distributes an osx64 bundle that works on current macOS setups.
        ASSET="codeql-osx64.zip"
        ;;
    MINGW*|MSYS*|CYGWIN*)
        ASSET="codeql-win64.zip"
        ;;
    *)
        echo "Unsupported OS: ${OS}" >&2
        exit 1
        ;;
esac

URL="https://github.com/github/codeql-cli-binaries/releases/download/v${CODEQL_VERSION}/${ASSET}"
TMP_DIR="$(mktemp -d)"
ARCHIVE_PATH="${TMP_DIR}/${ASSET}"
EXTRACT_DIR="${TMP_DIR}/extract"

echo "Downloading CodeQL ${CODEQL_VERSION} from ${URL}"
curl -fsSL "${URL}" -o "${ARCHIVE_PATH}"

mkdir -p "${EXTRACT_DIR}"
unzip -q "${ARCHIVE_PATH}" -d "${EXTRACT_DIR}"

mkdir -p "${CODEQL_INSTALL_DIR}"
rm -rf "${CODEQL_INSTALL_DIR:?}"/*
cp -R "${EXTRACT_DIR}/codeql/." "${CODEQL_INSTALL_DIR}"

echo "Installed CodeQL to ${CODEQL_INSTALL_DIR}"
"${CODEQL_INSTALL_DIR}/codeql" version
