#!/bin/bash

GIT_ROOT=$(git rev-parse --show-toplevel)
export DATABASE_PATH="${GIT_ROOT}/data/database/users.db"
export DATABASE_SALT='$2b$12$l0U9KMzQ52as6nnm5W6XJu'
export DATA_PATH="${GIT_ROOT}/data"

poetry run python main.py
