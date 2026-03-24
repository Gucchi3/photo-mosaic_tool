#!/usr/bin/env python3
"""
Photo Mosaic Generator - フォトモザイク生成ツール

指定フォルダ内の画像タイルを使い、ターゲット画像を再構成するフォトモザイクを生成します。
色空間はCIELAB（人間の知覚に近い）を使用し、複数の類似度指標を組み合わせて
最も適切なタイル画像を選択します。

Usage:
    python photo_mosaic.py --target <target_image> --tiles <tile_folder> [options]

Example:
    python photo_mosaic.py --target photo.jpg --tiles ./my_images/ --tile-size 48 --output mosaic.png
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter
from sklearn.neighbors import BallTree
from skimage.color import rgb2lab
from skimage.metrics import structural_similarity as ssim


# ─────────────────────────────────────────────
#  設定・データクラス
# ─────────────────────────────────────────────

@dataclass
class MosaicConfig:
    """モザイク生成の設定"""
    tile_size: int = 48            # 各タイルのピクセルサイズ
    output_scale: int = 1          # 出力倍率 (2 = タイルを2倍サイズで配置)
    color_weight: float = 0.6      # 平均色の重み
    histogram_weight: float = 0.25 # ヒストグラムの重み
    texture_weight: float = 0.15   # テクスチャ(エッジ)の重み
    blend_ratio: float = 0.0       # タイルとターゲット色のブレンド比率 (0=タイルそのまま, 0.3=少し色調補正)
    no_repeat_radius: int = 2      # 同一タイルの再利用禁止半径 (グリッド単位)
    top_k: int = 5                 # 候補タイル数 (この中からテクスチャ比較で最終選択)
    lab_hist_bins: int = 16        # LABヒストグラムのビン数
    max_workers: int = 0           # 並列ワーカー数 (0=自動)


@dataclass
class TileData:
    """タイル画像の特徴量"""
    path: str
    index: int
    avg_lab: np.ndarray           # LAB平均色 [L, a, b]
    histogram: np.ndarray          # LABヒストグラム（正規化済み）
    edge_density: float            # エッジ密度（テクスチャ指標）
    thumbnail: Optional[np.ndarray] = None  # リサイズ済みのRGB配列


# ─────────────────────────────────────────────
#  タイル画像の前処理
# ─────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def load_and_analyze_tile(args: tuple) -> Optional[dict]:
    """1枚のタイル画像を読み込み、特徴量を抽出する (並列処理用)"""
    path, index, tile_size, lab_hist_bins = args

    try:
        img = Image.open(path).convert('RGB')
    except Exception as e:
        print(f"  [SKIP] {path}: {e}")
        return None

    # 正方形にクロップ (中央切り出し)
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))

    # タイルサイズにリサイズ
    img = img.resize((tile_size, tile_size), Image.LANCZOS)
    rgb_array = np.array(img, dtype=np.float64) / 255.0

    # LAB色空間に変換
    lab_array = rgb2lab(rgb_array)

    # 1) 平均LAB色
    avg_lab = lab_array.mean(axis=(0, 1))

    # 2) LABヒストグラム (L, a, b 各チャンネル)
    hist_l = np.histogram(lab_array[:, :, 0], bins=lab_hist_bins, range=(0, 100))[0]
    hist_a = np.histogram(lab_array[:, :, 1], bins=lab_hist_bins, range=(-128, 127))[0]
    hist_b = np.histogram(lab_array[:, :, 2], bins=lab_hist_bins, range=(-128, 127))[0]
    histogram = np.concatenate([hist_l, hist_a, hist_b]).astype(np.float64)
    histogram = histogram / (histogram.sum() + 1e-10)  # 正規化

    # 3) エッジ密度 (テクスチャ指標)
    gray = img.convert('L')
    edges = gray.filter(ImageFilter.Kernel(
        size=(3, 3),
        kernel=[-1, -1, -1, -1, 8, -1, -1, -1, -1],
        scale=1, offset=128
    ))
    edge_array = np.array(edges, dtype=np.float64)
    edge_density = np.abs(edge_array - 128).mean() / 128.0

    return {
        'path': str(path),
        'index': index,
        'avg_lab': avg_lab.tolist(),
        'histogram': histogram.tolist(),
        'edge_density': float(edge_density),
        'thumbnail': (rgb_array * 255).astype(np.uint8).tolist(),
    }


def load_tile_database(tile_folder: str, config: MosaicConfig) -> list[TileData]:
    """タイルフォルダ内の全画像を読み込み、特徴量データベースを構築"""
    tile_folder = Path(tile_folder)
    if not tile_folder.is_dir():
        raise FileNotFoundError(f"タイルフォルダが見つかりません: {tile_folder}")

    # 対応画像ファイルを収集
    image_paths = []
    for ext in SUPPORTED_EXTENSIONS:
        image_paths.extend(tile_folder.rglob(f"*{ext}"))
        image_paths.extend(tile_folder.rglob(f"*{ext.upper()}"))
    image_paths = sorted(set(image_paths))

    if len(image_paths) == 0:
        raise ValueError(f"タイルフォルダに画像が見つかりません: {tile_folder}")

    print(f"\n📂 タイル画像: {len(image_paths)} 枚を検出")
    print(f"   特徴量を抽出中...")

    # 並列で特徴量を抽出
    args_list = [
        (str(p), i, config.tile_size, config.lab_hist_bins)
        for i, p in enumerate(image_paths)
    ]

    workers = config.max_workers if config.max_workers > 0 else min(os.cpu_count() or 4, 8)
    tiles = []

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(load_and_analyze_tile, a): a for a in args_list}
        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            if done_count % 100 == 0 or done_count == len(args_list):
                print(f"   ... {done_count}/{len(args_list)}")

            result = future.result()
            if result is not None:
                tiles.append(TileData(
                    path=result['path'],
                    index=result['index'],
                    avg_lab=np.array(result['avg_lab']),
                    histogram=np.array(result['histogram']),
                    edge_density=result['edge_density'],
                    thumbnail=np.array(result['thumbnail'], dtype=np.uint8),
                ))

    elapsed = time.time() - t0
    print(f"   ✅ {len(tiles)} 枚のタイルを処理完了 ({elapsed:.1f}秒)")

    if len(tiles) < 10:
        print(f"   ⚠️  タイル数が少ないため、モザイクの品質が低下する可能性があります")

    return tiles


# ─────────────────────────────────────────────
#  類似度計算
# ─────────────────────────────────────────────

def build_search_index(tiles: list[TileData]) -> tuple[BallTree, np.ndarray]:
    """BallTreeインデックスを構築 (平均LAB色による高速近傍探索)"""
    lab_matrix = np.array([t.avg_lab for t in tiles])
    tree = BallTree(lab_matrix, metric='euclidean')
    return tree, lab_matrix


def histogram_distance(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """ヒストグラム間のchi-square距離"""
    denom = hist1 + hist2 + 1e-10
    return 0.5 * np.sum((hist1 - hist2) ** 2 / denom)


def compute_combined_score(
    target_avg_lab: np.ndarray,
    target_histogram: np.ndarray,
    target_edge_density: float,
    tile: TileData,
    config: MosaicConfig
) -> float:
    """複合類似度スコアを計算 (小さいほど良い)"""
    # 1) LAB色差 (CIEDE簡易版 - ユークリッド距離)
    color_dist = np.linalg.norm(target_avg_lab - tile.avg_lab)
    # 正規化 (LAB空間の最大距離は約375程度)
    color_score = color_dist / 100.0

    # 2) ヒストグラム距離
    hist_score = histogram_distance(target_histogram, tile.histogram)

    # 3) テクスチャ差
    texture_score = abs(target_edge_density - tile.edge_density)

    # 重み付き合計
    total = (
        config.color_weight * color_score +
        config.histogram_weight * hist_score +
        config.texture_weight * texture_score
    )
    return total


# ─────────────────────────────────────────────
#  ターゲット画像の解析
# ─────────────────────────────────────────────

def analyze_target_region(
    lab_image: np.ndarray,
    rgb_image: np.ndarray,
    row: int, col: int,
    tile_size: int,
    lab_hist_bins: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """ターゲット画像の1グリッド領域の特徴量を計算"""
    y0, y1 = row * tile_size, (row + 1) * tile_size
    x0, x1 = col * tile_size, (col + 1) * tile_size

    region_lab = lab_image[y0:y1, x0:x1]
    region_rgb = rgb_image[y0:y1, x0:x1]

    # 平均LAB色
    avg_lab = region_lab.mean(axis=(0, 1))

    # LABヒストグラム
    hist_l = np.histogram(region_lab[:, :, 0], bins=lab_hist_bins, range=(0, 100))[0]
    hist_a = np.histogram(region_lab[:, :, 1], bins=lab_hist_bins, range=(-128, 127))[0]
    hist_b = np.histogram(region_lab[:, :, 2], bins=lab_hist_bins, range=(-128, 127))[0]
    histogram = np.concatenate([hist_l, hist_a, hist_b]).astype(np.float64)
    histogram = histogram / (histogram.sum() + 1e-10)

    # エッジ密度
    gray = np.mean(region_rgb, axis=2)
    gy, gx = np.gradient(gray)
    edge_density = np.sqrt(gx**2 + gy**2).mean() / 255.0

    return avg_lab, histogram, edge_density


# ─────────────────────────────────────────────
#  モザイク生成
# ─────────────────────────────────────────────

def generate_mosaic(
    target_path: str,
    tiles: list[TileData],
    config: MosaicConfig,
    output_path: str
) -> str:
    """フォトモザイクを生成"""

    # ターゲット画像の読み込み
    target_img = Image.open(target_path).convert('RGB')
    print(f"\n🎯 ターゲット画像: {target_path}")
    print(f"   元サイズ: {target_img.size[0]}x{target_img.size[1]}")

    # タイルサイズの倍数にリサイズ
    ts = config.tile_size
    grid_cols = max(1, target_img.size[0] // ts)
    grid_rows = max(1, target_img.size[1] // ts)
    new_w = grid_cols * ts
    new_h = grid_rows * ts
    target_img = target_img.resize((new_w, new_h), Image.LANCZOS)

    print(f"   グリッド: {grid_cols}列 x {grid_rows}行 = {grid_cols * grid_rows} セル")
    print(f"   調整後サイズ: {new_w}x{new_h}")

    # NumPy配列に変換
    target_rgb = np.array(target_img, dtype=np.float64) / 255.0
    target_lab = rgb2lab(target_rgb)

    # BallTreeインデックスの構築
    print(f"\n🔍 検索インデックスを構築中...")
    tree, lab_matrix = build_search_index(tiles)

    # top_k候補を先に取得するためのk
    k_search = min(config.top_k * 3, len(tiles))  # 余裕を持って検索

    # 出力画像の準備
    out_tile_size = ts * config.output_scale
    output_w = grid_cols * out_tile_size
    output_h = grid_rows * out_tile_size
    output_img = Image.new('RGB', (output_w, output_h))

    # 使用済みタイルの追跡 (重複回避用)
    used_tiles = {}  # (row, col) -> tile_index

    total_cells = grid_rows * grid_cols
    processed = 0

    print(f"\n🎨 モザイクを生成中... ({total_cells} セル)")
    t0 = time.time()

    for row in range(grid_rows):
        for col in range(grid_cols):
            processed += 1
            if processed % 200 == 0 or processed == total_cells:
                pct = processed / total_cells * 100
                elapsed = time.time() - t0
                eta = elapsed / processed * (total_cells - processed) if processed > 0 else 0
                print(f"   [{processed}/{total_cells}] {pct:.1f}% 完了 (残り約 {eta:.0f}秒)")

            # ターゲット領域の特徴量を計算
            avg_lab, histogram, edge_density = analyze_target_region(
                target_lab, target_rgb * 255, row, col, ts, config.lab_hist_bins
            )

            # Phase 1: BallTreeで平均色が近い候補をk個取得
            dists, indices = tree.query(avg_lab.reshape(1, -1), k=k_search)
            candidate_indices = indices[0]

            # 使用済みタイルを除外 (近隣セルとの重複回避)
            if config.no_repeat_radius > 0:
                excluded = set()
                for dr in range(-config.no_repeat_radius, config.no_repeat_radius + 1):
                    for dc in range(-config.no_repeat_radius, config.no_repeat_radius + 1):
                        if (dr, dc) == (0, 0):
                            continue
                        key = (row + dr, col + dc)
                        if key in used_tiles:
                            excluded.add(used_tiles[key])
                candidate_indices = [i for i in candidate_indices if tiles[i].index not in excluded]

            if len(candidate_indices) == 0:
                candidate_indices = indices[0].tolist()

            # Phase 2: 複合スコアで上位k個を厳密評価
            candidates_with_score = []
            for idx in candidate_indices[:config.top_k * 2]:
                tile = tiles[idx]
                score = compute_combined_score(
                    avg_lab, histogram, edge_density, tile, config
                )
                candidates_with_score.append((score, idx))

            candidates_with_score.sort(key=lambda x: x[0])

            # Phase 3: top_k候補からSSIMで最終選択 
            best_score = float('inf')
            best_idx = candidates_with_score[0][1]

            target_region_rgb = (target_rgb[
                row * ts:(row + 1) * ts,
                col * ts:(col + 1) * ts
            ] * 255).astype(np.uint8)

            for score, idx in candidates_with_score[:config.top_k]:
                tile = tiles[idx]
                if tile.thumbnail is not None:
                    # SSIMで構造的類似度を計算
                    try:
                        ssim_val = ssim(
                            target_region_rgb, tile.thumbnail,
                            channel_axis=2,
                            data_range=255
                        )
                        combined = score * (1.0 - 0.3 * ssim_val)  # SSIMを30%加味
                        if combined < best_score:
                            best_score = combined
                            best_idx = idx
                    except Exception:
                        pass

            chosen_tile = tiles[best_idx]
            used_tiles[(row, col)] = chosen_tile.index

            # タイル画像を配置
            if chosen_tile.thumbnail is not None:
                tile_img = Image.fromarray(chosen_tile.thumbnail)
            else:
                tile_img = Image.open(chosen_tile.path).convert('RGB')
                w, h = tile_img.size
                side = min(w, h)
                left = (w - side) // 2
                top = (h - side) // 2
                tile_img = tile_img.crop((left, top, left + side, top + side))
                tile_img = tile_img.resize((ts, ts), Image.LANCZOS)

            # 出力スケールに合わせてリサイズ
            if config.output_scale != 1:
                tile_img = tile_img.resize((out_tile_size, out_tile_size), Image.LANCZOS)

            # 色調ブレンド (オプション)
            if config.blend_ratio > 0:
                target_region = target_img.crop((
                    col * ts, row * ts,
                    (col + 1) * ts, (row + 1) * ts
                ))
                if config.output_scale != 1:
                    target_region = target_region.resize(
                        (out_tile_size, out_tile_size), Image.LANCZOS
                    )
                tile_img = Image.blend(tile_img, target_region, config.blend_ratio)

            output_img.paste(tile_img, (col * out_tile_size, row * out_tile_size))

    elapsed = time.time() - t0
    print(f"\n   ✅ モザイク生成完了! ({elapsed:.1f}秒)")

    # 保存
    output_img.save(output_path, quality=95)
    print(f"\n💾 出力: {output_path}")
    print(f"   サイズ: {output_w}x{output_h} px")

    return output_path


# ─────────────────────────────────────────────
#  メイン
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="🖼️ Photo Mosaic Generator - フォトモザイク生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本的な使い方
  python photo_mosaic.py --target photo.jpg --tiles ./images/

  # タイルサイズを小さくして精細に (処理時間は増加)
  python photo_mosaic.py --target photo.jpg --tiles ./images/ --tile-size 24

  # 色調ブレンドで元画像の色を少し混ぜる
  python photo_mosaic.py --target photo.jpg --tiles ./images/ --blend 0.2

  # 出力を2倍サイズに (タイルのディテールが見やすい)
  python photo_mosaic.py --target photo.jpg --tiles ./images/ --output-scale 2

  # 全パラメータ指定
  python photo_mosaic.py --target photo.jpg --tiles ./images/ \\
      --tile-size 32 --output-scale 2 --blend 0.15 \\
      --no-repeat 3 --top-k 8 --output mosaic_output.png
        """
    )

    parser.add_argument('--target', '-t', required=True,
                        help='ターゲット画像のパス')
    parser.add_argument('--tiles', '-d', required=True,
                        help='タイル画像が入ったフォルダのパス')
    parser.add_argument('--output', '-o', default='mosaic_output.png',
                        help='出力ファイルパス (default: mosaic_output.png)')
    parser.add_argument('--tile-size', type=int, default=48,
                        help='タイルサイズ (px, default: 48)')
    parser.add_argument('--output-scale', type=int, default=1,
                        help='出力倍率 (default: 1)')
    parser.add_argument('--blend', type=float, default=0.0,
                        help='色調ブレンド比率 0.0-0.5 (default: 0.0 = ブレンドなし)')
    parser.add_argument('--no-repeat', type=int, default=2,
                        help='同一タイル再利用禁止半径 (default: 2)')
    parser.add_argument('--top-k', type=int, default=5,
                        help='最終SSIM比較する候補数 (default: 5, 増やすと精度↑速度↓)')
    parser.add_argument('--color-weight', type=float, default=0.6,
                        help='平均色の重み (default: 0.6)')
    parser.add_argument('--hist-weight', type=float, default=0.25,
                        help='ヒストグラムの重み (default: 0.25)')
    parser.add_argument('--texture-weight', type=float, default=0.15,
                        help='テクスチャの重み (default: 0.15)')
    parser.add_argument('--workers', type=int, default=0,
                        help='並列ワーカー数 (default: 0 = 自動)')

    args = parser.parse_args()

    # 入力チェック
    if not os.path.isfile(args.target):
        print(f"❌ ターゲット画像が見つかりません: {args.target}")
        sys.exit(1)
    if not os.path.isdir(args.tiles):
        print(f"❌ タイルフォルダが見つかりません: {args.tiles}")
        sys.exit(1)

    print("=" * 60)
    print("  🖼️  Photo Mosaic Generator")
    print("  フォトモザイク生成ツール")
    print("=" * 60)

    config = MosaicConfig(
        tile_size=args.tile_size,
        output_scale=args.output_scale,
        blend_ratio=max(0.0, min(0.5, args.blend)),
        no_repeat_radius=args.no_repeat,
        top_k=args.top_k,
        color_weight=args.color_weight,
        histogram_weight=args.hist_weight,
        texture_weight=args.texture_weight,
        max_workers=args.workers,
    )

    print(f"\n⚙️  設定:")
    print(f"   タイルサイズ: {config.tile_size}px")
    print(f"   出力倍率: {config.output_scale}x")
    print(f"   色調ブレンド: {config.blend_ratio}")
    print(f"   重複回避半径: {config.no_repeat_radius}")
    print(f"   SSIM候補数: {config.top_k}")
    print(f"   重み [色:{config.color_weight} / ヒスト:{config.histogram_weight} / テクスチャ:{config.texture_weight}]")

    # タイルデータベースの構築
    tiles = load_tile_database(args.tiles, config)

    # モザイク生成
    result = generate_mosaic(args.target, tiles, config, args.output)

    print(f"\n{'=' * 60}")
    print(f"  ✨ 完了! → {result}")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()

"""
Photo Mosaic Generator - フォトモザイク生成ツール

指定フォルダ内の画像タイルを使い、ターゲット画像を再構成するフォトモザイクを生成します。
色空間はCIELAB（人間の知覚に近い）を使用し、複数の類似度指標を組み合わせて
最も適切なタイル画像を選択します。

Usage:
    python photo_mosaic.py --target <target_image> --tiles <tile_folder> [options]

Example:
    python photo_mosaic.py --target photo.jpg --tiles ./my_images/ --tile-size 48 --output mosaic.png
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter
from sklearn.neighbors import BallTree
from skimage.color import rgb2lab
from skimage.metrics import structural_similarity as ssim


# ─────────────────────────────────────────────
#  設定・データクラス
# ─────────────────────────────────────────────

@dataclass
class MosaicConfig:
    """モザイク生成の設定"""
    tile_size: int = 48            # 各タイルのピクセルサイズ
    output_scale: int = 1          # 出力倍率 (2 = タイルを2倍サイズで配置)
    color_weight: float = 0.6      # 平均色の重み
    histogram_weight: float = 0.25 # ヒストグラムの重み
    texture_weight: float = 0.15   # テクスチャ(エッジ)の重み
    blend_ratio: float = 0.0       # タイルとターゲット色のブレンド比率 (0=タイルそのまま, 0.3=少し色調補正)
    no_repeat_radius: int = 2      # 同一タイルの再利用禁止半径 (グリッド単位)
    top_k: int = 5                 # 候補タイル数 (この中からテクスチャ比較で最終選択)
    lab_hist_bins: int = 16        # LABヒストグラムのビン数
    max_workers: int = 0           # 並列ワーカー数 (0=自動)


@dataclass
class TileData:
    """タイル画像の特徴量"""
    path: str
    index: int
    avg_lab: np.ndarray           # LAB平均色 [L, a, b]
    histogram: np.ndarray          # LABヒストグラム（正規化済み）
    edge_density: float            # エッジ密度（テクスチャ指標）
    thumbnail: Optional[np.ndarray] = None  # リサイズ済みのRGB配列


# ─────────────────────────────────────────────
#  タイル画像の前処理
# ─────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def load_and_analyze_tile(args: tuple) -> Optional[dict]:
    """1枚のタイル画像を読み込み、特徴量を抽出する (並列処理用)"""
    path, index, tile_size, lab_hist_bins = args

    try:
        img = Image.open(path).convert('RGB')
    except Exception as e:
        print(f"  [SKIP] {path}: {e}")
        return None

    # 正方形にクロップ (中央切り出し)
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))

    # タイルサイズにリサイズ
    img = img.resize((tile_size, tile_size), Image.LANCZOS)
    rgb_array = np.array(img, dtype=np.float64) / 255.0

    # LAB色空間に変換
    lab_array = rgb2lab(rgb_array)

    # 1) 平均LAB色
    avg_lab = lab_array.mean(axis=(0, 1))

    # 2) LABヒストグラム (L, a, b 各チャンネル)
    hist_l = np.histogram(lab_array[:, :, 0], bins=lab_hist_bins, range=(0, 100))[0]
    hist_a = np.histogram(lab_array[:, :, 1], bins=lab_hist_bins, range=(-128, 127))[0]
    hist_b = np.histogram(lab_array[:, :, 2], bins=lab_hist_bins, range=(-128, 127))[0]
    histogram = np.concatenate([hist_l, hist_a, hist_b]).astype(np.float64)
    histogram = histogram / (histogram.sum() + 1e-10)  # 正規化

    # 3) エッジ密度 (テクスチャ指標)
    gray = img.convert('L')
    edges = gray.filter(ImageFilter.Kernel(
        size=(3, 3),
        kernel=[-1, -1, -1, -1, 8, -1, -1, -1, -1],
        scale=1, offset=128
    ))
    edge_array = np.array(edges, dtype=np.float64)
    edge_density = np.abs(edge_array - 128).mean() / 128.0

    return {
        'path': str(path),
        'index': index,
        'avg_lab': avg_lab.tolist(),
        'histogram': histogram.tolist(),
        'edge_density': float(edge_density),
        'thumbnail': (rgb_array * 255).astype(np.uint8).tolist(),
    }


def load_tile_database(tile_folder: str, config: MosaicConfig) -> list[TileData]:
    """タイルフォルダ内の全画像を読み込み、特徴量データベースを構築"""
    tile_folder = Path(tile_folder)
    if not tile_folder.is_dir():
        raise FileNotFoundError(f"タイルフォルダが見つかりません: {tile_folder}")

    # 対応画像ファイルを収集
    image_paths = []
    for ext in SUPPORTED_EXTENSIONS:
        image_paths.extend(tile_folder.rglob(f"*{ext}"))
        image_paths.extend(tile_folder.rglob(f"*{ext.upper()}"))
    image_paths = sorted(set(image_paths))

    if len(image_paths) == 0:
        raise ValueError(f"タイルフォルダに画像が見つかりません: {tile_folder}")

    print(f"\n📂 タイル画像: {len(image_paths)} 枚を検出")
    print(f"   特徴量を抽出中...")

    # 並列で特徴量を抽出
    args_list = [
        (str(p), i, config.tile_size, config.lab_hist_bins)
        for i, p in enumerate(image_paths)
    ]

    workers = config.max_workers if config.max_workers > 0 else min(os.cpu_count() or 4, 8)
    tiles = []

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(load_and_analyze_tile, a): a for a in args_list}
        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            if done_count % 100 == 0 or done_count == len(args_list):
                print(f"   ... {done_count}/{len(args_list)}")

            result = future.result()
            if result is not None:
                tiles.append(TileData(
                    path=result['path'],
                    index=result['index'],
                    avg_lab=np.array(result['avg_lab']),
                    histogram=np.array(result['histogram']),
                    edge_density=result['edge_density'],
                    thumbnail=np.array(result['thumbnail'], dtype=np.uint8),
                ))

    elapsed = time.time() - t0
    print(f"   ✅ {len(tiles)} 枚のタイルを処理完了 ({elapsed:.1f}秒)")

    if len(tiles) < 10:
        print(f"   ⚠️  タイル数が少ないため、モザイクの品質が低下する可能性があります")

    return tiles


# ─────────────────────────────────────────────
#  類似度計算
# ─────────────────────────────────────────────

def build_search_index(tiles: list[TileData]) -> tuple[BallTree, np.ndarray]:
    """BallTreeインデックスを構築 (平均LAB色による高速近傍探索)"""
    lab_matrix = np.array([t.avg_lab for t in tiles])
    tree = BallTree(lab_matrix, metric='euclidean')
    return tree, lab_matrix


def histogram_distance(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """ヒストグラム間のchi-square距離"""
    denom = hist1 + hist2 + 1e-10
    return 0.5 * np.sum((hist1 - hist2) ** 2 / denom)


def compute_combined_score(
    target_avg_lab: np.ndarray,
    target_histogram: np.ndarray,
    target_edge_density: float,
    tile: TileData,
    config: MosaicConfig
) -> float:
    """複合類似度スコアを計算 (小さいほど良い)"""
    # 1) LAB色差 (CIEDE簡易版 - ユークリッド距離)
    color_dist = np.linalg.norm(target_avg_lab - tile.avg_lab)
    # 正規化 (LAB空間の最大距離は約375程度)
    color_score = color_dist / 100.0

    # 2) ヒストグラム距離
    hist_score = histogram_distance(target_histogram, tile.histogram)

    # 3) テクスチャ差
    texture_score = abs(target_edge_density - tile.edge_density)

    # 重み付き合計
    total = (
        config.color_weight * color_score +
        config.histogram_weight * hist_score +
        config.texture_weight * texture_score
    )
    return total


# ─────────────────────────────────────────────
#  ターゲット画像の解析
# ─────────────────────────────────────────────

def analyze_target_region(
    lab_image: np.ndarray,
    rgb_image: np.ndarray,
    row: int, col: int,
    tile_size: int,
    lab_hist_bins: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """ターゲット画像の1グリッド領域の特徴量を計算"""
    y0, y1 = row * tile_size, (row + 1) * tile_size
    x0, x1 = col * tile_size, (col + 1) * tile_size

    region_lab = lab_image[y0:y1, x0:x1]
    region_rgb = rgb_image[y0:y1, x0:x1]

    # 平均LAB色
    avg_lab = region_lab.mean(axis=(0, 1))

    # LABヒストグラム
    hist_l = np.histogram(region_lab[:, :, 0], bins=lab_hist_bins, range=(0, 100))[0]
    hist_a = np.histogram(region_lab[:, :, 1], bins=lab_hist_bins, range=(-128, 127))[0]
    hist_b = np.histogram(region_lab[:, :, 2], bins=lab_hist_bins, range=(-128, 127))[0]
    histogram = np.concatenate([hist_l, hist_a, hist_b]).astype(np.float64)
    histogram = histogram / (histogram.sum() + 1e-10)

    # エッジ密度
    gray = np.mean(region_rgb, axis=2)
    gy, gx = np.gradient(gray)
    edge_density = np.sqrt(gx**2 + gy**2).mean() / 255.0

    return avg_lab, histogram, edge_density


# ─────────────────────────────────────────────
#  モザイク生成
# ─────────────────────────────────────────────

def generate_mosaic(
    target_path: str,
    tiles: list[TileData],
    config: MosaicConfig,
    output_path: str
) -> str:
    """フォトモザイクを生成"""

    # ターゲット画像の読み込み
    target_img = Image.open(target_path).convert('RGB')
    print(f"\n🎯 ターゲット画像: {target_path}")
    print(f"   元サイズ: {target_img.size[0]}x{target_img.size[1]}")

    # タイルサイズの倍数にリサイズ
    ts = config.tile_size
    grid_cols = max(1, target_img.size[0] // ts)
    grid_rows = max(1, target_img.size[1] // ts)
    new_w = grid_cols * ts
    new_h = grid_rows * ts
    target_img = target_img.resize((new_w, new_h), Image.LANCZOS)

    print(f"   グリッド: {grid_cols}列 x {grid_rows}行 = {grid_cols * grid_rows} セル")
    print(f"   調整後サイズ: {new_w}x{new_h}")

    # NumPy配列に変換
    target_rgb = np.array(target_img, dtype=np.float64) / 255.0
    target_lab = rgb2lab(target_rgb)

    # BallTreeインデックスの構築
    print(f"\n🔍 検索インデックスを構築中...")
    tree, lab_matrix = build_search_index(tiles)

    # top_k候補を先に取得するためのk
    k_search = min(config.top_k * 3, len(tiles))  # 余裕を持って検索

    # 出力画像の準備
    out_tile_size = ts * config.output_scale
    output_w = grid_cols * out_tile_size
    output_h = grid_rows * out_tile_size
    output_img = Image.new('RGB', (output_w, output_h))

    # 使用済みタイルの追跡 (重複回避用)
    used_tiles = {}  # (row, col) -> tile_index

    total_cells = grid_rows * grid_cols
    processed = 0

    print(f"\n🎨 モザイクを生成中... ({total_cells} セル)")
    t0 = time.time()

    for row in range(grid_rows):
        for col in range(grid_cols):
            processed += 1
            if processed % 200 == 0 or processed == total_cells:
                pct = processed / total_cells * 100
                elapsed = time.time() - t0
                eta = elapsed / processed * (total_cells - processed) if processed > 0 else 0
                print(f"   [{processed}/{total_cells}] {pct:.1f}% 完了 (残り約 {eta:.0f}秒)")

            # ターゲット領域の特徴量を計算
            avg_lab, histogram, edge_density = analyze_target_region(
                target_lab, target_rgb * 255, row, col, ts, config.lab_hist_bins
            )

            # Phase 1: BallTreeで平均色が近い候補をk個取得
            dists, indices = tree.query(avg_lab.reshape(1, -1), k=k_search)
            candidate_indices = indices[0]

            # 使用済みタイルを除外 (近隣セルとの重複回避)
            if config.no_repeat_radius > 0:
                excluded = set()
                for dr in range(-config.no_repeat_radius, config.no_repeat_radius + 1):
                    for dc in range(-config.no_repeat_radius, config.no_repeat_radius + 1):
                        if (dr, dc) == (0, 0):
                            continue
                        key = (row + dr, col + dc)
                        if key in used_tiles:
                            excluded.add(used_tiles[key])
                candidate_indices = [i for i in candidate_indices if tiles[i].index not in excluded]

            if len(candidate_indices) == 0:
                candidate_indices = indices[0].tolist()

            # Phase 2: 複合スコアで上位k個を厳密評価
            candidates_with_score = []
            for idx in candidate_indices[:config.top_k * 2]:
                tile = tiles[idx]
                score = compute_combined_score(
                    avg_lab, histogram, edge_density, tile, config
                )
                candidates_with_score.append((score, idx))

            candidates_with_score.sort(key=lambda x: x[0])

            # Phase 3: top_k候補からSSIMで最終選択 
            best_score = float('inf')
            best_idx = candidates_with_score[0][1]

            target_region_rgb = (target_rgb[
                row * ts:(row + 1) * ts,
                col * ts:(col + 1) * ts
            ] * 255).astype(np.uint8)

            for score, idx in candidates_with_score[:config.top_k]:
                tile = tiles[idx]
                if tile.thumbnail is not None:
                    # SSIMで構造的類似度を計算
                    try:
                        ssim_val = ssim(
                            target_region_rgb, tile.thumbnail,
                            channel_axis=2,
                            data_range=255
                        )
                        combined = score * (1.0 - 0.3 * ssim_val)  # SSIMを30%加味
                        if combined < best_score:
                            best_score = combined
                            best_idx = idx
                    except Exception:
                        pass

            chosen_tile = tiles[best_idx]
            used_tiles[(row, col)] = chosen_tile.index

            # タイル画像を配置
            if chosen_tile.thumbnail is not None:
                tile_img = Image.fromarray(chosen_tile.thumbnail)
            else:
                tile_img = Image.open(chosen_tile.path).convert('RGB')
                w, h = tile_img.size
                side = min(w, h)
                left = (w - side) // 2
                top = (h - side) // 2
                tile_img = tile_img.crop((left, top, left + side, top + side))
                tile_img = tile_img.resize((ts, ts), Image.LANCZOS)

            # 出力スケールに合わせてリサイズ
            if config.output_scale != 1:
                tile_img = tile_img.resize((out_tile_size, out_tile_size), Image.LANCZOS)

            # 色調ブレンド (オプション)
            if config.blend_ratio > 0:
                target_region = target_img.crop((
                    col * ts, row * ts,
                    (col + 1) * ts, (row + 1) * ts
                ))
                if config.output_scale != 1:
                    target_region = target_region.resize(
                        (out_tile_size, out_tile_size), Image.LANCZOS
                    )
                tile_img = Image.blend(tile_img, target_region, config.blend_ratio)

            output_img.paste(tile_img, (col * out_tile_size, row * out_tile_size))

    elapsed = time.time() - t0
    print(f"\n   ✅ モザイク生成完了! ({elapsed:.1f}秒)")

    # 保存
    output_img.save(output_path, quality=95)
    print(f"\n💾 出力: {output_path}")
    print(f"   サイズ: {output_w}x{output_h} px")

    return output_path


# ─────────────────────────────────────────────
#  メイン
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="🖼️ Photo Mosaic Generator - フォトモザイク生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本的な使い方
  python photo_mosaic.py --target photo.jpg --tiles ./images/

  # タイルサイズを小さくして精細に (処理時間は増加)
  python photo_mosaic.py --target photo.jpg --tiles ./images/ --tile-size 24

  # 色調ブレンドで元画像の色を少し混ぜる
  python photo_mosaic.py --target photo.jpg --tiles ./images/ --blend 0.2

  # 出力を2倍サイズに (タイルのディテールが見やすい)
  python photo_mosaic.py --target photo.jpg --tiles ./images/ --output-scale 2

  # 全パラメータ指定
  python photo_mosaic.py --target photo.jpg --tiles ./images/ \\
      --tile-size 32 --output-scale 2 --blend 0.15 \\
      --no-repeat 3 --top-k 8 --output mosaic_output.png
        """
    )

    parser.add_argument('--target', '-t', required=True,
                        help='ターゲット画像のパス')
    parser.add_argument('--tiles', '-d', required=True,
                        help='タイル画像が入ったフォルダのパス')
    parser.add_argument('--output', '-o', default='mosaic_output.png',
                        help='出力ファイルパス (default: mosaic_output.png)')
    parser.add_argument('--tile-size', type=int, default=48,
                        help='タイルサイズ (px, default: 48)')
    parser.add_argument('--output-scale', type=int, default=1,
                        help='出力倍率 (default: 1)')
    parser.add_argument('--blend', type=float, default=0.0,
                        help='色調ブレンド比率 0.0-0.5 (default: 0.0 = ブレンドなし)')
    parser.add_argument('--no-repeat', type=int, default=2,
                        help='同一タイル再利用禁止半径 (default: 2)')
    parser.add_argument('--top-k', type=int, default=5,
                        help='最終SSIM比較する候補数 (default: 5, 増やすと精度↑速度↓)')
    parser.add_argument('--color-weight', type=float, default=0.6,
                        help='平均色の重み (default: 0.6)')
    parser.add_argument('--hist-weight', type=float, default=0.25,
                        help='ヒストグラムの重み (default: 0.25)')
    parser.add_argument('--texture-weight', type=float, default=0.15,
                        help='テクスチャの重み (default: 0.15)')
    parser.add_argument('--workers', type=int, default=0,
                        help='並列ワーカー数 (default: 0 = 自動)')

    args = parser.parse_args()

    # 入力チェック
    if not os.path.isfile(args.target):
        print(f"❌ ターゲット画像が見つかりません: {args.target}")
        sys.exit(1)
    if not os.path.isdir(args.tiles):
        print(f"❌ タイルフォルダが見つかりません: {args.tiles}")
        sys.exit(1)

    print("=" * 60)
    print("  🖼️  Photo Mosaic Generator")
    print("  フォトモザイク生成ツール")
    print("=" * 60)

    config = MosaicConfig(
        tile_size=args.tile_size,
        output_scale=args.output_scale,
        blend_ratio=max(0.0, min(0.5, args.blend)),
        no_repeat_radius=args.no_repeat,
        top_k=args.top_k,
        color_weight=args.color_weight,
        histogram_weight=args.hist_weight,
        texture_weight=args.texture_weight,
        max_workers=args.workers,
    )

    print(f"\n⚙️  設定:")
    print(f"   タイルサイズ: {config.tile_size}px")
    print(f"   出力倍率: {config.output_scale}x")
    print(f"   色調ブレンド: {config.blend_ratio}")
    print(f"   重複回避半径: {config.no_repeat_radius}")
    print(f"   SSIM候補数: {config.top_k}")
    print(f"   重み [色:{config.color_weight} / ヒスト:{config.histogram_weight} / テクスチャ:{config.texture_weight}]")

    # タイルデータベースの構築
    tiles = load_tile_database(args.tiles, config)

    # モザイク生成
    result = generate_mosaic(args.target, tiles, config, args.output)

    print(f"\n{'=' * 60}")
    print(f"  ✨ 完了! → {result}")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()
