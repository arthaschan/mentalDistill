#!/usr/bin/env bash
# Wait for all training to finish, then move base models from EasyEdit3 to mentalDistill/models/
# and update setup.env to point to new paths.
set -e

PROJ="/home/student/arthas/mentalDistill"
SRC="/home/student/arthas/EasyEdit3"
DST="$PROJ/models"
SETUP_ENV="$PROJ/setup.env"
EXPERIMENT_PID=981344  # run_full_experiment.sh PID

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for training PID $EXPERIMENT_PID to finish..."

# Wait for the experiment pipeline to complete
while kill -0 "$EXPERIMENT_PID" 2>/dev/null; do
    sleep 60
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training finished. Starting model migration..."

mkdir -p "$DST"

for model_dir in \
    "$SRC/Qwen2.5-7B-Instruct" \
    "$SRC/Qwen2.5-14B-Instruct" \
    "$SRC/Qwen2.5-32B-Instruct" \
    "$SRC/Qwen3-14B"; do

    name=$(basename "$model_dir")
    if [[ -d "$model_dir" ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Moving $name ($(du -sh "$model_dir" | cut -f1))..."
        mv "$model_dir" "$DST/$name"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done: $name"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP: $model_dir not found"
    fi
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All models moved. Verifying..."
du -sh "$DST"/*/

# Update setup.env to point to new model locations
if [[ -f "$SETUP_ENV" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Updating setup.env model paths..."
    sed -i "s|$SRC|$DST|g" "$SETUP_ENV"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] setup.env updated."
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Migration complete."

# Clean up empty EasyEdit3 if nothing left
remaining=$(ls -A "$SRC" 2>/dev/null | wc -l)
if [[ "$remaining" -eq 0 ]]; then
    rmdir "$SRC"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Removed empty $SRC"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $SRC still has $remaining items, not removing."
fi
