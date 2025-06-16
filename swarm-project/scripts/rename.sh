#!/usr/bin/env bash

# Script to ensure each JPEG filename is prefixed with its class directory name.
# For example, ILSVRC2012_val_xxx.JPEG → n019328574_val_xxx.JPEG when in class folder n019328574.

TRAIN_DIR="data/imagenette/train"
VAL_DIR="data/imagenette/val"

find "$VAL_DIR" -type f -name "*.JPEG" | while read -r filepath; do
    dirpath=$(dirname "$filepath")
    classdir=$(basename "$dirpath")
    filename=$(basename "$filepath")
    # If filename doesn't already start with classdir + underscore, rename it
    if [[ "$filename" != ${classdir}_* ]]; then
        mv "$filepath" "$dirpath/${classdir}_${filename}"
        echo "Renamed $filename → ${classdir}_${filename}"
    fi
done