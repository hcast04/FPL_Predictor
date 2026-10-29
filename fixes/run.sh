#!/usr/bin/env bash
# Convenience shim: the real entry point lives in the project root.
exec "$(dirname "$0")/../run.sh" "$@"
