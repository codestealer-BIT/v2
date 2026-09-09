from __future__ import annotations

"""RPC-based utility for extracting video frames and saving them as local JPEGs.

本模块直接基于 ``thriftpy2`` + ``euler.Client`` 调用
``LsContentInsightService.PackContentFeature``，通过
``pipeline_id=all_feature_v2`` + ``ItemFrameLoaderConfig`` 抽取视频帧，
不依赖任何本地的 ``TiktokDataFusedPipeline`` 封装。

安全与鉴权
-----------
- 本模块 **不内置任何 AK/SK**，也不会主动管理密钥。
- 调用方必须在字节内部环境中，确保已经按视频架构 IAM 规范
  为目标业务线申请了合规的抽帧相关权限（VL 等级、国内/海外等），
  并通过统一的鉴权链路（例如 ``euler.base_compat_middleware`` 注入
  统一头）正确下发 AK/SK 等凭据。
- 若 IAM 权限或网络环境不满足要求，RPC 可能返回非零的
  ``BaseResp.StatusCode``，或 ``frame_list`` 为空。

典型用法
--------
1. 在 Python 代码中调用::

    from tools.item_frame_extractor_rpc import extract_frames_by_item_id_via_rpc

    paths = extract_frames_by_item_id_via_rpc(
        item_id=7503539237917035783,
        output_dir="/tmp/video_frames",
        max_frames=5,
        width=360,
        height=640,
        fps=2,
        psm_name="tiktok.ls.content_insight",
        cluster="offline",
        idc="maliva",
    )

2. 从命令行使用::

    python -m tools.item_frame_extractor_rpc \
        --item_id 7503539237917035783 \
        --out /tmp/video_frames \
        --max_frames 5 \
        --width 360 \
        --height 640 \
        --fps 2 \
        --psm tiktok.ls.content_insight \
        --cluster offline \
        --idc maliva

返回的路径列表均为 **绝对路径**，便于后续管道处理。
"""

import argparse
import logging
import os
from io import BytesIO
from typing import List

import thriftpy2
from PIL import Image, ImageFile

import euler
from euler import base_compat_middleware
import bytedlogger

# 部分环境可能没有安装 whatimage/pyheif，这里做可选导入
try:  # type: ignore[unused-ignore]
    import whatimage  # type: ignore
except Exception:  # noqa: BLE001
    whatimage = None  # type: ignore

try:  # type: ignore[unused-ignore]
    import pyheif  # type: ignore
except Exception:  # noqa: BLE001
    pyheif = None  # type: ignore


# 允许加载部分截断的图片，增强鲁棒性
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 配置 bytedlogger 的默认日志格式
bytedlogger.config_default()
logger = bytedlogger.logging  # 统一使用 bytedlogger 的 logging 实例


# IDL 路径（固定挂载路径）
_IDL_ROOT = "/mnt/bn/jiny-ttls-i18n-fr1q/ttls_content/idl"
_CONTENT_INSIGHT_THRIFT = os.path.join(
    _IDL_ROOT,
    "tiktok_Is_content_insight_service.thrift",
)
_BASE_THRIFT = os.path.join(_IDL_ROOT, "base.thrift")


_feature_thrift = None
_base_thrift = None


def _load_idl_modules():
    """Lazily load thrift IDL modules used by the RPC client.

    为避免在导入时就触发 IO，这里采用惰性加载。
    """

    global _feature_thrift, _base_thrift
    if _feature_thrift is None or _base_thrift is None:
        _feature_thrift = thriftpy2.load(_CONTENT_INSIGHT_THRIFT)
        _base_thrift = thriftpy2.load(_BASE_THRIFT)
    return _feature_thrift, _base_thrift


def _build_client(
    psm_name: str,
    cluster: str,
    idc: str | None,
    timeout: int = 60,
) -> euler.Client:
    """构造并返回一个 LsContentInsightService 的 euler 客户端。

    不做任何全局缓存，调用方如需高频调用可以在外层自行缓存。
    """

    feature_thrift, _ = _load_idl_modules()
    service = feature_thrift.LsContentInsightService

    if idc is not None:
        endpoint = f"sd://{psm_name}?cluster={cluster}&idc={idc}"
    else:
        endpoint = f"sd://{psm_name}?cluster={cluster}"

    client = euler.Client(
        service,
        endpoint,
        timeout=timeout,
        transport="ttheader",
    )
    client.use(base_compat_middleware.client_middleware)
    return client


def _decode_image_bytes(data: bytes) -> Image.Image:
    """将原始二进制图片数据解码为 RGB PIL.Image。

    优先尝试 whatimage + pyheif 以兼容 HEIC；若不可用，则退化为
    直接交给 Pillow 处理。任何异常由调用方负责捕获。
    """

    # 优先尝试 whatimage 检测格式
    if whatimage is not None:
        try:
            fmt = whatimage.identify_image(data)  # type: ignore[attr-defined]
            if fmt == "heic" and pyheif is not None:
                heif_img = pyheif.read(data)  # type: ignore[attr-defined]
                return Image.frombytes(
                    mode=heif_img.mode,
                    size=heif_img.size,
                    data=heif_img.data,
                )
        except Exception:  # noqa: BLE001
            # 检测失败或非 HEIC，降级交给 Pillow
            pass

    # 默认路径：交给 Pillow 处理
    return Image.open(BytesIO(data)).convert("RGB")


def extract_frames_by_item_id_via_rpc(
    item_id: int,
    output_dir: str,
    max_frames: int = 5,
    width: int = 360,
    height: int = 640,
    fps: int = 2,
    psm_name: str = "tiktok.ls.content_insight",
    cluster: str = "offline",
    idc: str | None = "maliva",
) -> List[str]:
    """通过 LsContentInsightService 抽取指定 item 的视频帧并保存为 JPG。

    参数
    ----
    item_id:
        待抽帧的视频 item_id。
    output_dir:
        本地输出目录，若不存在会自动创建。
    max_frames:
        最多抽取的帧数，上限由后端服务与视频本身决定。
    width, height:
        抽帧分辨率，传给 ``ItemFrameLoaderConfig``。
    fps:
        抽帧帧率，传给 ``ItemFrameLoaderConfig``。
    psm_name:
        目标 RPC 服务的 PSM，默认 ``tiktok.ls.content_insight``。
    cluster:
        服务集群，例如 ``offline``、``default`` 等。
    idc:
        目标 IDC，例如 ``maliva``，为 ``None`` 时不在 endpoint 中附加
        ``idc`` 参数。

    返回
    ----
    list[str]
        实际成功保存的帧图像 **绝对路径** 列表。

    异常
    ----
    RuntimeError
        - ``BaseResp.StatusCode`` 非 0 时；
        - 找不到当前 item 对应的 ``content_features``；
        - ``frame_list`` 为空或不存在（通常意味着权限/环境/参数问题）。
    ValueError
        - ``max_frames <= 0``。
    其它异常
        - euler.Client 建立或 RPC 调用失败时原样抛出。

    说明
    ----
    - 函数对每一帧解码/保存分别 ``try/except``，单帧失败会记录日志
      并跳过，其余帧继续处理；
    - 若整体 ``frame_list`` 为空，则直接抛出异常，方便调用方排查
      IAM/环境/参数问题。
    """

    if max_frames <= 0:
        raise ValueError("max_frames must be a positive integer")

    os.makedirs(output_dir, exist_ok=True)

    feature_thrift, base_thrift = _load_idl_modules()

    logger.info(
        "[item_frame_rpc] calling PackContentFeature psm=%s cluster=%s idc=%s "
        "item_id=%s max_frames=%s width=%s height=%s fps=%s",
        psm_name,
        cluster,
        idc,
        item_id,
        max_frames,
        width,
        height,
        fps,
    )

    client = _build_client(psm_name=psm_name, cluster=cluster, idc=idc)

    # 构造请求
    content = feature_thrift.Content(
        id=item_id,
        type=3,
        item_id=item_id,
    )

    item_frame_loader_cfg = feature_thrift.ItemFrameLoaderConfig(
        width=width,
        height=height,
        frame_number=max_frames,
        fps=fps,
    )

    custom_cfg = {
        "item_frame_loader": feature_thrift.CustomNodeConfig(
            item_frame_loader_config=item_frame_loader_cfg,
        )
    }

    base = base_thrift.Base(Extra={"": "", "env": "prod"})

    req = feature_thrift.PackContentFeatureReq(
        content=[content],
        pipeline_id="all_feature_v2",
        custom_config=custom_cfg,
        Base=base,
    )

    # 调用 RPC
    rsp = client.PackContentFeature(req)

    # 校验返回码
    if not hasattr(rsp, "BaseResp"):
        raise RuntimeError("PackContentFeature response missing BaseResp")

    status_code = rsp.BaseResp.StatusCode
    if status_code != 0:
        raise RuntimeError(
            f"PackContentFeature failed for item_id={item_id}, "
            f"status_code={status_code}, message={getattr(rsp.BaseResp, 'Message', '')}"
        )

    # 读取 content_features
    content_features_map = getattr(rsp, "content_features", None)
    if content_features_map is None:
        raise RuntimeError(
            f"PackContentFeature response has no content_features for item_id={item_id}"
        )

    try:
        content_features = content_features_map[item_id]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"No content_features found for item_id={item_id}; "
            "check whether the item exists and permissions are correct."
        ) from exc

    frame_list = getattr(content_features, "frame_list", None)
    if not frame_list:
        raise RuntimeError(
            "Empty or missing frame_list from PackContentFeature for "
            f"item_id={item_id}. This is often caused by insufficient IAM "
            "permissions, incorrect environment (cluster/idc), or an invalid "
            "item_id/parameter combination."
        )

    saved_paths: List[str] = []

    for idx, frame_hex in enumerate(frame_list):
        if frame_hex is None:
            logger.warning(
                "[item_frame_rpc] frame %d for item_id=%s is None, skipping",
                idx,
                item_id,
            )
            continue

        try:
            frame_bytes = bytes.fromhex(frame_hex)
            img = _decode_image_bytes(frame_bytes)

            filename = f"{item_id}_frame_{idx}.jpg"
            file_path = os.path.join(output_dir, filename)
            img.save(file_path, format="JPEG", quality=95)
            saved_paths.append(os.path.abspath(file_path))
        except Exception as exc:  # noqa: BLE001
            # 单帧失败不影响其它帧
            logger.exception(
                "[item_frame_rpc] failed to decode/save frame %d for item_id=%s: %s",
                idx,
                item_id,
                exc,
            )

    return saved_paths


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "Extract video frames via LsContentInsightService.PackContentFeature "
            "and save them as local JPEG files."
        )
    )
    parser.add_argument(
        "--item_id",
        type=int,
        required=True,
        help="TikTok item_id whose frames will be extracted.",
    )
    parser.add_argument(
        "--out",
        "--output_dir",
        dest="output_dir",
        required=True,
        help="Directory to save extracted JPEG frames.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=5,
        help="Maximum number of frames to request (default: 5).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=360,
        help="Frame width passed to ItemFrameLoaderConfig (default: 360).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=640,
        help="Frame height passed to ItemFrameLoaderConfig (default: 640).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=2,
        help="FPS passed to ItemFrameLoaderConfig (default: 2).",
    )
    parser.add_argument(
        "--psm",
        "--psm_name",
        dest="psm_name",
        type=str,
        default="tiktok.ls.content_insight",
        help="PSM of LsContentInsightService (default: tiktok.ls.content_insight).",
    )
    parser.add_argument(
        "--cluster",
        type=str,
        default="offline",
        help="Service cluster name (default: offline).",
    )
    parser.add_argument(
        "--idc",
        type=str,
        default="maliva",
        help="IDC name (default: maliva). Use empty string to omit idc in endpoint.",
    )
    return parser.parse_args()


def main() -> None:
    """命令行入口：按参数调用 RPC 抽帧并打印保存路径。"""

    args = _parse_args()

    # 若显式传入空字符串，则视为不指定 idc
    idc: str | None
    if args.idc == "":
        idc = None
    else:
        idc = args.idc

    logger.info(
        "[item_frame_rpc_cli] item_id=%s output_dir=%s max_frames=%s width=%s "
        "height=%s fps=%s psm=%s cluster=%s idc=%s",
        args.item_id,
        args.output_dir,
        args.max_frames,
        args.width,
        args.height,
        args.fps,
        args.psm_name,
        args.cluster,
        idc,
    )

    try:
        saved_paths = extract_frames_by_item_id_via_rpc(
            item_id=args.item_id,
            output_dir=args.output_dir,
            max_frames=args.max_frames,
            width=args.width,
            height=args.height,
            fps=args.fps,
            psm_name=args.psm_name,
            cluster=args.cluster,
            idc=idc,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "[item_frame_rpc_cli] failed to extract frames for item_id=%s: %s",
            args.item_id,
            exc,
        )
        raise SystemExit(1) from exc

    if not saved_paths:
        logger.warning(
            "[item_frame_rpc_cli] no frames were saved for item_id=%s",
            args.item_id,
        )
    else:
        logger.info(
            "[item_frame_rpc_cli] saved %d frames to %s",
            len(saved_paths),
            args.output_dir,
        )
        for path in saved_paths:
            # 逐行打印，方便后续管道消费
            print(path)


if __name__ == "__main__":  # pragma: no cover
    main()
