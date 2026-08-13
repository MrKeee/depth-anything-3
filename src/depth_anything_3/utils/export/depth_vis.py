# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import imageio
import numpy as np

from depth_anything_3.specs import Prediction
from depth_anything_3.utils.visualize import visualize_depth


def export_to_depth_vis(
    prediction: Prediction,
    export_dir: str,
    fps: float = 20,
    macro_block_size: int = 2,
    output_name: str = "depth_vis.mp4",
):
    out_dir = os.path.join(export_dir, "depth_vis")
    os.makedirs(out_dir, exist_ok=True)

    filename = os.path.basename(output_name)
    if not filename.lower().endswith(".mp4"):
        filename = f"{filename}.mp4"
    mp4_path = os.path.join(out_dir, filename)

    # imageio 的 ffmpeg writer 默认 macro_block_size=16，会把非 16 倍数分辨率自动 resize（例如 322x182 -> 336x192）。
    # 这里使用 macro_block_size=2：既能保持原始分辨率，又满足 yuv420p 的偶数宽高要求。
    writer = imageio.get_writer(
        mp4_path,
        fps=fps,
        codec="libx264",
        macro_block_size=macro_block_size,
        # 注意：imageio/ffmpeg writer 本身可能也会设置一次 pix_fmt，
        # 如果这里再通过 ffmpeg_params 传 "-pix_fmt yuv420p" 会导致日志提示：
        # "Multiple -pix_fmt options specified ... only the last option ... will be used."
        # 用 pixelformat 参数更干净，避免重复传参。
        pixelformat="yuv420p",
    )

    try:
        for idx in range(prediction.depth.shape[0]):
            depth_vis = np.asarray(visualize_depth(prediction.depth[idx]))
            writer.append_data(depth_vis)
    finally:
        writer.close()