"""
WBench Video Generation Script.

Generates multi-turn videos for all (or specified) cases using a registered model.

Usage:
    # Generate for all cases(为所有测试用例生成视频)
    python generate.py --model example --data_dir data

    # Generate specific cases(为指定的几个测试用例生成视频)
    python generate.py --model example --cases data/cases/case_1.json data/cases/case_2.json

    # Limit number of cases(限制生成的测试用例最大数量)
    python generate.py --model example --limit 10

    # Resume (skip existing)(断点续传，跳过已经生成过的视频)
    python generate.py --model example --resume
"""
import argparse
import glob
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models import get_model, list_models
from src.utils.case_loader import load_cases_raw
# 配置全局的日志记录格式和级别
logging.basicConfig(
    level=logging.INFO,   # 设置日志级别为 INFO，只输出 INFO 及以上级别的信息
    format="%(asctime)s [%(levelname)s] %(message)s",   #设置日志输出格式：时间 [日志级别] 消息内容
    datefmt="%H:%M:%S",   # 设置时间格式为：时:分:秒
)
logger = logging.getLogger(__name__)


def generate_case(model, case: dict, output_dir: str, data_root: str, **gen_kwargs) -> dict:
    """Generate multi-turn video for a single case."""
    case_id = case["id"]  # 从传入的测试用例字典中提取 case 的唯一 ID
    
    # 拼接出该测试用例对应的输出视频的完整文件路径
    output_path = os.path.join(output_dir, f"case_{case_id}_combined.mp4")
    # 具体的生成逻辑（如如何处理提示词、调用API等）由底层具体的模型类实现
    result = model.generate_multi_turn(
        case=case,
        output_path=output_path,
        data_root=data_root,
        **gen_kwargs,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="WBench video generation")
    parser.add_argument("--model", required=True, help=f"Model name. Available: {list_models()}")
    parser.add_argument("--data_dir", default="data", help="Path to data/ directory")
    parser.add_argument("--output_dir", default=None, help="Output dir (default: output_videos/<model>)")
    parser.add_argument("--cases", nargs="*", help="Specific case JSON files to process")
    parser.add_argument("--limit", type=int, default=None, help="Max cases to process")
    parser.add_argument("--resume", action="store_true", help="Skip cases with existing videos")
    parser.add_argument("--duration", type=float, default=4.0, help="Video duration per turn in seconds (default: 4.0)")
    parser.add_argument("--resolution", type=str, default="720P", help="Video resolution (default: 720P)")
    args = parser.parse_args()

    model = get_model(args.model)
    logger.info(f"Using model: {model}")
    
    # 确定最终的输出目录：如果用户指定了就用用户的，否则使用默认规范路径 "output_videos/模型名"
    output_dir = args.output_dir or os.path.join("output_videos", args.model)
    os.makedirs(output_dir, exist_ok=True)


    """ 加载数据 """
    
    if args.cases:   # 如果用户在命令行具体指定了哪些 case 文件（例如 --cases file1.json file2.json）
        cases = []
        for f in args.cases:
            with open(f) as fp:
                cases.append(json.load(fp))
    else:
        cases = load_cases_raw(args.data_dir)  # 如果用户没有指定具体文件，则调用内置工具函数，从 data_dir 目录下加载所有的测试用例

    if args.limit:
        cases = cases[:args.limit]

    logger.info(f"Processing {len(cases)} cases → {output_dir}")

    results = {"success": 0, "failed": 0, "skipped": 0}
    t0 = time.time()

    for i, case in enumerate(cases):
        case_id = case["id"]
        out_path = os.path.join(output_dir, f"case_{case_id}_combined.mp4")
        
        # 如果用户开启了 --resume 并且对应的视频文件已经存在于磁盘上了，那么就跳过这个测试用例的生成，直接进入下一个循环迭代
        if args.resume and os.path.exists(out_path):
            logger.info(f"[{i+1}/{len(cases)}] case_{case_id}: SKIP (exists)")
            results["skipped"] += 1
            continue
        # 打印正在生成的日志，例如 [1/289] case_1: generating...
        logger.info(f"[{i+1}/{len(cases)}] case_{case_id}: generating...")
        
        # 核心调用：执行当前用例的视频生成任务
        result = generate_case(model, case, output_dir, args.data_dir,
                               duration=args.duration, resolution=args.resolution)

        if result.get("code") == 0:
            results["success"] += 1
            logger.info(f"  → OK: {result['video_path']}")
        else:
            results["failed"] += 1
            logger.error(f"  → FAIL: {result.get('error')}")

    elapsed = time.time() - t0  # 循环结束，计算总耗时（当前时间减去起始时间）
    logger.info(
        f"\nDone in {elapsed:.1f}s — "
        f"success={results['success']}, failed={results['failed']}, skipped={results['skipped']}"
    )    # 打印最终的执行总结报告，包括总耗时、成功数、失败数、跳过数


if __name__ == "__main__":
    main()
