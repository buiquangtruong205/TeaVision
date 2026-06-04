#!/usr/bin/env sh

uvicorn src.deploy.api:app --host 0.0.0.0 --port "${PORT:-8000}"
