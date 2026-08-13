#!/bin/bash

set -euo pipefail

PY_SCRIPT="/mnt/petrelfs/liukehui/projects/Depth-Anything-3/gen_our_droid_depth/droid_pick_the_pot_lid_from_the_towel_and_put_it_on_the_pot_1.0.0_lerobot/video_depth_gen_from_lerobot_droid_version_mp.py"
MODEL_DIR="/mnt/petrelfs/liukehui/evla1_t/liukehui/projects/Depth-Anything-3/models/DA3NESTED-GIANT-LARGE-1.1"
DEVICES="0,1,2,3"

VIDEO_ROOT="/mnt/petrelfs/liukehui/evla1_t/liukehui/data/droid_put_the_pot_lid_on_the_towel_1.0.0_lerobot/droid_1.0.0_lerobot/videos"
OUT_ROOT="/mnt/petrelfs/liukehui/evla1_t/liukehui/data/droid_put_the_pot_lid_on_the_towel_1.0.0_lerobot/droid_1.0.0_lerobot/depth_videos"

for chunk in chunk-{000..000}; do
  VIDEO_DIR="$VIDEO_ROOT/$chunk/observation.images.exterior_image_1_left"
  OUT_DIR="$OUT_ROOT/$chunk/observation.images.exterior_image_1_left"
  LOG_PATH="$OUT_DIR/run.log"

  mkdir -p "$OUT_DIR"
  echo "================================================================================" | tee -a "$LOG_PATH"
  echo "[`date '+%F %T'`] chunk=$chunk" | tee -a "$LOG_PATH"
  echo "video_dir=$VIDEO_DIR" | tee -a "$LOG_PATH"
  echo "output_dir=$OUT_DIR" | tee -a "$LOG_PATH"
  echo "devices=$DEVICES" | tee -a "$LOG_PATH"

  python "$PY_SCRIPT" \
    --video-dir "$VIDEO_DIR" \
    --output-dir "$OUT_DIR" \
    --devices "$DEVICES" \
    --model-dir "$MODEL_DIR" \
    --fps 15.0 \
    --export-format depth_vis \
    --process-res 320 2>&1 | tee -a "$LOG_PATH"

  echo "[`date '+%F %T'`] chunk=$chunk 执行完毕" | tee -a "$LOG_PATH"
  echo "================================================================================" | tee -a "$LOG_PATH"
done

for chunk in chunk-{000..000}; do
  VIDEO_DIR="$VIDEO_ROOT/$chunk/observation.images.exterior_image_2_left"
  OUT_DIR="$OUT_ROOT/$chunk/observation.images.exterior_image_2_left"
  LOG_PATH="$OUT_DIR/run.log"

  mkdir -p "$OUT_DIR"
  echo "================================================================================" | tee -a "$LOG_PATH"
  echo "[`date '+%F %T'`] chunk=$chunk" | tee -a "$LOG_PATH"
  echo "video_dir=$VIDEO_DIR" | tee -a "$LOG_PATH"
  echo "output_dir=$OUT_DIR" | tee -a "$LOG_PATH"
  echo "devices=$DEVICES" | tee -a "$LOG_PATH"

  python "$PY_SCRIPT" \
    --video-dir "$VIDEO_DIR" \
    --output-dir "$OUT_DIR" \
    --devices "$DEVICES" \
    --model-dir "$MODEL_DIR" \
    --fps 15.0 \
    --export-format depth_vis \
    --process-res 320 2>&1 | tee -a "$LOG_PATH"

  echo "[`date '+%F %T'`] chunk=$chunk 执行完毕" | tee -a "$LOG_PATH"
  echo "================================================================================" | tee -a "$LOG_PATH"
done

for chunk in chunk-{000..000}; do
  VIDEO_DIR="$VIDEO_ROOT/$chunk/observation.images.wrist_image_left"
  OUT_DIR="$OUT_ROOT/$chunk/observation.images.wrist_image_left"
  LOG_PATH="$OUT_DIR/run.log"

  mkdir -p "$OUT_DIR"
  echo "================================================================================" | tee -a "$LOG_PATH"
  echo "[`date '+%F %T'`] chunk=$chunk" | tee -a "$LOG_PATH"
  echo "video_dir=$VIDEO_DIR" | tee -a "$LOG_PATH"
  echo "output_dir=$OUT_DIR" | tee -a "$LOG_PATH"
  echo "devices=$DEVICES" | tee -a "$LOG_PATH"

  python "$PY_SCRIPT" \
    --video-dir "$VIDEO_DIR" \
    --output-dir "$OUT_DIR" \
    --devices "$DEVICES" \
    --model-dir "$MODEL_DIR" \
    --fps 15.0 \
    --export-format depth_vis \
    --process-res 320 2>&1 | tee -a "$LOG_PATH"

  echo "[`date '+%F %T'`] chunk=$chunk 执行完毕" | tee -a "$LOG_PATH"
  echo "================================================================================" | tee -a "$LOG_PATH"
done
