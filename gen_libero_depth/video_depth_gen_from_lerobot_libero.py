import argparse
import glob
import os
import shutil
import time
import traceback
import multiprocessing as mp

from tqdm import tqdm

from depth_anything_3.api import DepthAnything3
from depth_anything_3.services.input_handlers import VideoHandler


VIEW_TO_COLS = {
    "exterior_image_1_left": (
        "exterior_image_1_left_cam_to_base_extrinsics_matrix",
        "exterior_image_1_left_intrinsics_matrix",
    ),
    "exterior_image_2_left": (
        "exterior_image_2_left_cam_to_base_extrinsics_matrix",
        "exterior_image_2_left_intrinsics_matrix",
    ),
}


def load_extrinsics_and_intrinsics(video_path: str, n_frames: int):
    """按 video_path 自动找到对应 parquet，并读取内外参，返回 (N,4,4)/(N,3,3).

    注意：parquet 里是 cam_to_base（更像 c2w），而模型内部 cam_enc 会对输入 ext 做一次 affine_inverse
    来得到 c2w，因此这里先把 c2w 取逆成 w2c 再喂给模型。
    """
    import numpy as np
    import pandas as pd

    p = os.path.abspath(video_path)
    parts = p.split(os.sep)
    if "videos" not in parts:
        raise ValueError(f"video_path 不包含 videos：{p}")
    i = parts.index("videos")
    dataset_root = os.sep.join(parts[:i])
    chunk = parts[i + 1]
    view_dir = parts[i + 2]  # observation.images.exterior_image_1_left
    view = view_dir.split("observation.images.", 1)[1]
    if view not in VIEW_TO_COLS:
        raise ValueError(f"不支持的 view：{view}")

    episode = os.path.splitext(os.path.basename(p))[0]

    col_extr, col_intr = VIEW_TO_COLS[view]
    parquet_path = os.path.join(dataset_root, "data_with_cam", chunk, f"{episode}.parquet")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(parquet_path)

    def as_mat(v, shape, name):
        a = np.asarray(v)
        # 常见存法：object array，shape=(4,), 每个元素是 shape=(4,) 的数组/列表
        if getattr(a, "dtype", None) == object:
            try:
                a = np.stack(list(a), axis=0)
            except Exception:
                a = np.asarray(a.tolist())
        a = np.asarray(a, dtype=np.float32)
        if a.shape == shape:
            return a
        raise ValueError(f"{name} shape={a.shape}, expect {shape}")

    df = pd.read_parquet(parquet_path, columns=[col_extr, col_intr])
    extr = df.iloc[0][col_extr]
    intr = df.iloc[0][col_intr]
    if extr is None or intr is None:
        return None, None
    else:
        extr_4x4 = as_mat(extr, (4, 4), col_extr)
        intr_3x3 = as_mat(intr, (3, 3), col_intr)
        # parquet 里是 cam_to_base（更像 c2w），而 cam_enc 内部会对输入 ext 做 affine_inverse 得到 c2w；
        # 因此这里先用同一个 affine_inverse 把 c2w 转成 w2c，保证求逆方式与模型一致。
        from depth_anything_3.utils.geometry import affine_inverse
        import torch

        extr_4x4 = affine_inverse(torch.from_numpy(extr_4x4)).cpu().numpy()  # c2w -> w2c
        return (
            np.repeat(extr_4x4[None, ...], n_frames, axis=0),
            np.repeat(intr_3x3[None, ...], n_frames, axis=0),
        )


def run_single_video(
    video_path: str,
    output_dir: str,
    model_dir: str,
    device: str = "cuda:0",
    export_format: str = "mini_npz",
    fps: float = 15.0,
    process_res: int = 504,
    model: DepthAnything3 | None = None,
) -> None:
    """对单个视频运行深度估计并保存结果。"""
    os.makedirs(output_dir, exist_ok=True)

    image_dir_path = os.path.join(output_dir, "input_images")
    try:
        # 1. 从视频中提取帧
        frame_paths = VideoHandler.process(
            video_path=video_path,
            output_dir=output_dir,
            fps=fps,
        )

        # wrist 视角没有内外参，直接用 None
        if "observation.images.wrist_image_left" in video_path:
            extrinsics, intrinsics = None, None
        else:
            extrinsics, intrinsics = load_extrinsics_and_intrinsics(
                video_path, len(frame_paths)
            )

        # 2. 加载模型并进行深度估计
        if model is None:
            model = DepthAnything3.from_pretrained(model_dir).to(device).eval()

        start_time = time.time()
        export_kwargs = (
            {"depth_vis": {"output_name": os.path.basename(video_path), "fps": fps}}
            if export_format == "depth_vis"
            else {}
        )
        _ = model.inference(
            frame_paths,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            export_dir=output_dir,
            export_format=export_format,
            process_res=process_res,
            export_kwargs=export_kwargs,
        )
        end_time = time.time()

        print(f"[{os.path.basename(video_path)}] 深度估计时间：{end_time - start_time:.2f} 秒")
        print(
            f"[{os.path.basename(video_path)}] 深度估计速度：{len(frame_paths) / (end_time - start_time):.2f} FPS"
        )
    finally:
        # 无论成功/失败都清理临时帧，节省空间
        if os.path.exists(image_dir_path):
            shutil.rmtree(image_dir_path)
            print(f"[{os.path.basename(video_path)}] 已删除图片文件夹：{image_dir_path}")


def process_dir(
    video_dir: str,
    output_dir: str,
    model_dir: str,
    device: str,
    export_format: str,
    fps: float,
    process_res: int,
    model: DepthAnything3 | None = None,
) -> None:
    """遍历目录下所有 mp4 文件，逐个处理。"""
    video_dir = os.path.abspath(video_dir)
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "error_videos.log")
    mp4_files = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    if not mp4_files:
        print(f"未在 {video_dir} 下找到任何 mp4 文件")
        return

    print(f"在 {video_dir} 中共找到 {len(mp4_files)} 个视频。")

    for idx, mp4_file in enumerate(
        tqdm(mp4_files, desc=f"处理目录 {video_dir}", unit="video"), start=1
    ):
        print(f"\n[{idx}/{len(mp4_files)}] 处理: {mp4_file}")
        try:
            # 为每个视频创建独立输出子目录，避免 input_images 等临时目录冲突
            video_base = os.path.splitext(os.path.basename(mp4_file))[0]
            video_out_dir = os.path.join(output_dir, video_base)
            run_single_video(
                video_path=mp4_file,
                output_dir=video_out_dir,
                model_dir=model_dir,
                device=device,
                export_format=export_format,
                fps=fps,
                process_res=process_res,
                model=model,
            )
        except Exception as e:
            # 记录报错并继续下一个视频（包括 CUDA OOM）
            err = traceback.format_exc()
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write(f"video: {mp4_file}\n")
                f.write(f"error: {repr(e)}\n")
                f.write(err + "\n")
            print(f"[跳过] 处理失败：{mp4_file}\n  已写入日志：{log_path}")
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
    print(f"All jobs finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")


def _parse_devices(devices: str | None, fallback_device: str) -> list[str]:
    """
    Parse device list string like: "cuda:0,cuda:1" or "0,1".
    Returns a list of device strings like ["cuda:0", "cuda:1"].
    """
    if devices is None or devices.strip() == "":
        return [fallback_device]
    parts = [p.strip() for p in devices.split(",") if p.strip()]
    out: list[str] = []
    for p in parts:
        if p.isdigit():
            out.append(f"cuda:{p}")
        else:
            out.append(p)
    return out


def _device_index(device: str) -> int | None:
    """Return CUDA device index if device is like 'cuda:0', else None."""
    if device.startswith("cuda:"):
        try:
            return int(device.split(":", 1)[1])
        except Exception:
            return None
    return None


def _worker_process_videos(
    rank: int,
    device: str,
    mp4_files: list[str],
    output_dir: str,
    model_dir: str,
    export_format: str,
    fps: float,
    process_res: int,
) -> None:
    """
    Worker process: load model once on assigned device, then run inference on its shard of videos.
    Each video writes into output_dir/<video_basename>/...
    """
    # Import torch inside worker to reduce multiprocessing + CUDA edge cases in parent.
    import torch

    cuda_idx = _device_index(device)
    if cuda_idx is not None and torch.cuda.is_available():
        torch.cuda.set_device(cuda_idx)

    os.makedirs(output_dir, exist_ok=True)
    # 每个 worker 单独写一个日志文件，避免多进程同时 append 同一个文件导致内容交错
    device_tag = device.replace(":", "_").replace("/", "_")
    log_path = os.path.join(output_dir, f"error_videos_worker{rank}_{device_tag}.log")
    print(f"[worker {rank}] device={device} videos={len(mp4_files)} output_dir={output_dir}")

    model = DepthAnything3.from_pretrained(model_dir).to(device).eval()

    for idx, mp4_file in enumerate(mp4_files, start=1):
        print(f"[worker {rank}] ({idx}/{len(mp4_files)}) {mp4_file}")
        try:
            video_base = os.path.splitext(os.path.basename(mp4_file))[0]
            video_out_dir = os.path.join(output_dir, video_base)
            run_single_video(
                video_path=mp4_file,
                output_dir=video_out_dir,
                model_dir=model_dir,
                device=device,
                export_format=export_format,
                fps=fps,
                process_res=process_res,
                model=model,
            )
        except Exception as e:
            err = traceback.format_exc()
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write(f"worker: {rank}\n")
                f.write(f"device: {device}\n")
                f.write(f"video: {mp4_file}\n")
                f.write(f"error: {repr(e)}\n")
                f.write(err + "\n")
            print(f"[worker {rank}] [跳过] 处理失败：{mp4_file}\n  已写入日志：{log_path}")
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


def process_dir_multi_gpu(
    video_dir: str,
    output_dir: str,
    model_dir: str,
    devices: list[str],
    export_format: str,
    fps: float,
    process_res: int,
) -> None:
    """多 GPU 并行：按 devices 分片视频列表，每张卡一个进程。"""
    video_dir = os.path.abspath(video_dir)
    os.makedirs(output_dir, exist_ok=True)
    mp4_files = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    if not mp4_files:
        print(f"未在 {video_dir} 下找到任何 mp4 文件")
        return

    shards: list[list[str]] = [[] for _ in devices]
    for i, f in enumerate(mp4_files):
        shards[i % len(devices)].append(f)

    ctx = mp.get_context("spawn")
    procs: list[mp.Process] = []
    for rank, (device, shard) in enumerate(zip(devices, shards)):
        if not shard:
            continue
        p = ctx.Process(
            target=_worker_process_videos,
            args=(
                rank,
                device,
                shard,
                output_dir,
                model_dir,
                export_format,
                fps,
                process_res,
            ),
        )
        p.daemon = False
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    failed = [p.exitcode for p in procs if p.exitcode not in (0, None)]
    if failed:
        raise RuntimeError(f"Some worker processes failed, exit codes={failed}")
    print(f"All multi-GPU jobs finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="视频深度估计脚本：支持单视频或目录内批量视频。"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--video-path",
        type=str,
        help="单个视频路径，例如 /path/to/episode_000000.mp4",
    )
    group.add_argument(
        "--video-dir",
        type=str,
        help="包含多个 mp4 的目录（不递归），将依次处理其中所有 mp4",
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default="/home/liukehui/liukehui/projects/Depth-Anything-3/models/DA3NESTED-GIANT-LARGE-1.1",
        help="模型目录（包含 config.json 和 model.safetensors）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="推理设备，例如 cuda:0 / cuda:1 / cpu（默认 cuda:0）",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="多卡并行推理设备列表，例如 'cuda:0,cuda:1' 或 '0,1'；设置后会覆盖 --device（仅目录模式生效）",
    )
    parser.add_argument(
        "--export-format",
        type=str,
        default="depth_vis",
        help="导出格式（默认 mini_npz）",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=15.0,
        help="抽帧 FPS（默认 15.0）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="单视频模式下的输出目录；若不指定，则默认为与视频同目录、同名子目录",
    )
    parser.add_argument(
        "--process-res",
        type=int,
        default=320,
        help="处理分辨率504",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.video_path:
        # 单视频模式
        video_path = os.path.abspath(args.video_path)  # e.g. .../videos/chunk-000/.../episode_000004.mp4

        if args.output_dir is not None:
            output_dir = os.path.abspath(args.output_dir)
        else:
            video_dir = os.path.dirname(video_path)
            base = os.path.splitext(os.path.basename(video_path))[0]
            output_dir = os.path.join(video_dir, base)

        run_single_video(
            video_path=video_path,
            output_dir=output_dir,
            model_dir=args.model_dir,
            device=args.device,
            export_format=args.export_format,
            fps=args.fps,
            process_res=args.process_res,
        )
    else:
        # 目录批量模式
        video_dir = os.path.abspath(args.video_dir)
        output_dir = (
            os.path.abspath(args.output_dir)
            if args.output_dir is not None
            else os.path.join(video_dir, "depth_anything3_outputs")
        )
        devices = _parse_devices(args.devices, args.device)
        if len(devices) > 1:
            process_dir_multi_gpu(
                video_dir=video_dir,
                output_dir=output_dir,
                model_dir=args.model_dir,
                devices=devices,
                export_format=args.export_format,
                fps=args.fps,
                process_res=args.process_res,
            )
        else:
            # 单卡/单进程：复用一次模型即可
            model = DepthAnything3.from_pretrained(args.model_dir).to(devices[0]).eval()
            process_dir(
                video_dir=video_dir,
                output_dir=output_dir,
                model_dir=args.model_dir,
                device=devices[0],
                export_format=args.export_format,
                fps=args.fps,
                process_res=args.process_res,
                model=model,
            )


if __name__ == "__main__":
    main()
