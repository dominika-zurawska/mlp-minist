#!/bin/bash

if [ ! -f /app/data/.model_registered ]; then
    echo register model
    python src/register_local_model.py

    touch /app/data/.model_registered
fi

exec "$@"
