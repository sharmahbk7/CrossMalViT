#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME=crossmal_vit
SERIALIZED_FILE=checkpoints/crossmal_vit_best.pth
HANDLER=deployment/torchserve/handler.py
EXTRA_FILES="configs/experiment/main_experiment.yaml"

mkdir -p model_store

torch-model-archiver   --model-name ${MODEL_NAME}   --version 1.0   --serialized-file ${SERIALIZED_FILE}   --handler ${HANDLER}   --extra-files ${EXTRA_FILES}   --export-path model_store   --force
