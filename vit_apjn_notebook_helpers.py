
# Auto-generated from vit_apjn_theory_plots.py for the notebook workflow.

from __future__ import annotations

import gc
import importlib.util
import math
import pickle
import subprocess
import sys
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn as nn
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable

build_dataset = None
DynamicErf = None
convert_ln_to_derf = None
create_model = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_BOOTSTRAP_STATE = {
    "repo_dir": None,
    "path_ready": False,
}

_CIFAR_DATASET_CACHE = {}
_CIFAR_STREAM_CACHE = {}


def bootstrap_vit_dyt_repo(
    repo_dir: str | Path = "vit-dyt",
    *,
    clone_if_missing: bool = False,
    install_requirements: bool = False,
    install_timm: bool = False,
):
    # Align imports with the original Colab workflow: dynamic_tanh and dataset helpers
    # should come from the vit-dyt repo once it is on sys.path.
    repo_path = Path(repo_dir).expanduser().resolve()

    if install_timm:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm"], check=True)

    if not repo_path.exists():
        if not clone_if_missing:
            raise FileNotFoundError(
                f"vit-dyt repo not found at {repo_path}. "
                "Pass clone_if_missing=True or clone it manually first."
            )
        subprocess.run(
            ["git", "clone", "https://github.com/labofdoubt/vit-dyt", str(repo_path)],
            check=True,
        )

    req = repo_path / "requirements.txt"
    if install_requirements and req.exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)], check=True)

    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    _BOOTSTRAP_STATE["repo_dir"] = repo_path
    _BOOTSTRAP_STATE["path_ready"] = True
    return repo_path


def _ensure_repo_path():
    if _BOOTSTRAP_STATE["path_ready"]:
        return _BOOTSTRAP_STATE["repo_dir"]

    local_candidate = Path("vit-dyt")
    if local_candidate.exists():
        return bootstrap_vit_dyt_repo(local_candidate, clone_if_missing=False)

    raise RuntimeError(
        "vit-dyt repo path is not configured. "
        "Run bootstrap_vit_dyt_repo(...) first in the notebook setup cell."
    )


def _load_repo_module(module_name: str, relative_path: str):
    repo_dir = _ensure_repo_path()
    module_path = repo_dir / relative_path
    if not module_path.exists():
        raise FileNotFoundError(f"Required repo module not found: {module_path}")

    cache_key = f"_vit_dyt_repo_{module_name}"
    if cache_key in sys.modules:
        return sys.modules[cache_key]

    spec = importlib.util.spec_from_file_location(cache_key, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[cache_key] = module
    spec.loader.exec_module(module)
    return module


def _ensure_build_dataset():
    global build_dataset
    if build_dataset is None:
        build_dataset = _load_repo_module("datasets", "datasets.py").build_dataset
    return build_dataset


def _ensure_dynamic_tanh():
    global DynamicErf, convert_ln_to_derf
    if DynamicErf is None or convert_ln_to_derf is None:
        mod = _load_repo_module("dynamic_tanh", "dynamic_tanh.py")
        DynamicErf = mod.DynamicErf
        convert_ln_to_derf = mod.convert_ln_to_derf
    return DynamicErf, convert_ln_to_derf


def _ensure_create_model():
    global create_model
    if create_model is None:
        _ensure_repo_path()
        from timm.models import create_model as _create_model
        create_model = _create_model
    return create_model


def clear_cifar_experiment_cache():
    _CIFAR_DATASET_CACHE.clear()
    _CIFAR_STREAM_CACHE.clear()


def _make_cifar_dataset_key(*, img_size: int, num_classes: int):
    repo_dir = _BOOTSTRAP_STATE.get("repo_dir")
    return (
        str(repo_dir) if repo_dir is not None else None,
        int(img_size),
        int(num_classes),
    )


def _get_cached_cifar_dataset(*, img_size: int, num_classes: int):
    dataset_key = _make_cifar_dataset_key(img_size=img_size, num_classes=num_classes)
    if dataset_key in _CIFAR_DATASET_CACHE:
        return _CIFAR_DATASET_CACHE[dataset_key]

    _build_dataset = _ensure_build_dataset()
    args = SimpleNamespace(
        data_set="CIFAR",
        data_path="tmp/cifar",
        eval_data_path=None,
        nb_classes=num_classes,
        input_size=img_size,
        imagenet_default_mean_and_std=True,
        color_jitter=0.4,
        aa="rand-m9-mstd0.5-inc1",
        train_interpolation="bicubic",
        reprob=0.25,
        remode="pixel",
        recount=1,
        crop_pct=None,
    )
    train_dataset, _ = _build_dataset(is_train=True, args=args)
    _CIFAR_DATASET_CACHE[dataset_key] = train_dataset
    return train_dataset


def _make_cifar_stream_key(
    *,
    batch_size: int,
    img_size: int,
    num_classes: int,
    loader_seed: int,
    std_threshold: float,
):
    return (
        _make_cifar_dataset_key(img_size=img_size, num_classes=num_classes),
        int(batch_size),
        int(loader_seed),
        float(std_threshold),
    )


def _make_cifar_loader(*, dataset, batch_size: int, loader_seed: int):
    gen = torch.Generator()
    gen.manual_seed(int(loader_seed))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        generator=gen,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )


def _get_or_create_cifar_stream(
    *,
    batch_size: int,
    img_size: int,
    num_classes: int,
    loader_seed: int,
    std_threshold: float,
):
    stream_key = _make_cifar_stream_key(
        batch_size=batch_size,
        img_size=img_size,
        num_classes=num_classes,
        loader_seed=loader_seed,
        std_threshold=std_threshold,
    )
    if stream_key in _CIFAR_STREAM_CACHE:
        return _CIFAR_STREAM_CACHE[stream_key]

    dataset = _get_cached_cifar_dataset(img_size=img_size, num_classes=num_classes)
    loader = _make_cifar_loader(dataset=dataset, batch_size=batch_size, loader_seed=loader_seed)
    stream_state = {
        "loader": loader,
        "iterator": iter(loader),
        "epoch": 0,
        "accepted_batches": [],
    }
    _CIFAR_STREAM_CACHE[stream_key] = stream_state
    return stream_state

BASE_FONTSIZE = 13
LABEL_FONTSIZE = 14
TITLE_FONTSIZE = 14
TICK_FONTSIZE = 11
COLORBAR_FONTSIZE = 12
PLOT_WSPACE = 0.34
BASE_CMAP_NAME = "viridis"
CVAL_MIN = 0.08
CVAL_MAX = 0.92
CVAL_GAMMA = 0.95

WHITE_RED_CMAP = LinearSegmentedColormap.from_list("white_to_red", ["#ffffff", "#b11226"])

def set_pub_style():
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": BASE_FONTSIZE,
        "axes.labelsize": LABEL_FONTSIZE,
        "axes.titlesize": TITLE_FONTSIZE,
        "xtick.labelsize": TICK_FONTSIZE,
        "ytick.labelsize": TICK_FONTSIZE,
        "legend.fontsize": TICK_FONTSIZE,
        "axes.linewidth": 0.9,
        "lines.linewidth": 1.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "xtick.minor.width": 0.7,
        "ytick.minor.width": 0.7,
        "axes.grid": False,
    })

set_pub_style()


def _clear_matplotlib_tex_cache():
    tex_cache = Path(mpl.get_cachedir()) / "tex.cache"
    if tex_cache.exists():
        shutil.rmtree(tex_cache)


def configure_neurips_like_tex_fonts():
    from matplotlib_inline.backend_inline import set_matplotlib_formats

    set_matplotlib_formats("svg")
    _clear_matplotlib_tex_cache()

    neurips_like_preamble = r"""
\usepackage[T1]{fontenc}
\usepackage{newtxtext}
\usepackage{newtxmath}
\usepackage[cal=cm]{mathalfa}

\usepackage{microtype}
\usepackage{xcolor}

\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
\usepackage{bm}
\usepackage{accents}

\usepackage{nicefrac}
\usepackage{xfrac}

\newcommand{\dbtilde}[1]{\accentset{\approx}{#1}}
\newcommand{\vardbtilde}[1]{\tilde{\raisebox{0pt}[0.87\height]{$\tilde{#1}$}}}

\makeatletter
\AtBeginDocument{%
  \DeclareSymbolFont{CMcal}{OMS}{cmsy}{m}{n}%
  \SetSymbolFont{CMcal}{bold}{OMS}{cmsy}{b}{n}%
  \DeclareSymbolFontAlphabet{\mathcal}{CMcal}%
}
\makeatother
"""

    mpl.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": neurips_like_preamble,
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def configure_times_like_tex_fonts():
    from matplotlib_inline.backend_inline import set_matplotlib_formats

    set_matplotlib_formats("svg")
    _clear_matplotlib_tex_cache()

    mpl.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"""
\usepackage[T1]{fontenc}
\usepackage{mathptmx}
\usepackage{amsmath,amssymb,amsfonts,bm}
\usepackage{microtype}
""",
    })

def seed_all(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def cuda_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def shade_from_index(i, n, light=CVAL_MIN, dark=CVAL_MAX, gamma=CVAL_GAMMA):
    if n <= 1:
        return 0.5 * (light + dark)
    t = (i / (n - 1)) ** gamma
    return light + (dark - light) * t

def prettify_log_axis(ax, axis="y"):
    locator_major = LogLocator(base=10.0, numticks=10)
    locator_minor = LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
    formatter = LogFormatterMathtext(base=10.0)
    if axis == "y":
        ax.yaxis.set_major_locator(locator_major)
        ax.yaxis.set_minor_locator(locator_minor)
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.xaxis.set_major_locator(locator_major)
        ax.xaxis.set_minor_locator(locator_minor)
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_minor_formatter(NullFormatter())

def prettify_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(which="both", top=False, right=False)
    ax.grid(True, which="major", linewidth=0.6, alpha=0.22)
    ax.grid(False, which="minor")

def add_alpha_colorbar_horizontal_single(
    cax,
    alphas,
    colors,
    label=r"$\alpha$",
    cb_fs=COLORBAR_FONTSIZE,
    max_tick_labels: int | None = None,
    ):

  alphas = np.asarray(alphas, dtype=float)
  n = len(alphas)
  img = np.zeros((1, n, 4))
  for i in range(n):
      img[0, i, :] = colors[i]

  da = float(np.mean(np.diff(alphas))) if n > 1 else 1.0
  x0, x1 = float(alphas[0] - da / 2), float(alphas[-1] + da / 2)

  cax.imshow(img, origin="lower", aspect="auto", extent=(x0, x1, 0, 1))
  cax.set_ylim(0, 1)
  cax.set_yticks([])

  if max_tick_labels is None or max_tick_labels >= n:
      idx = np.arange(n)
  else:
      idx = np.unique(np.linspace(0, n - 1, num=max_tick_labels, dtype=int))

  tick_vals = alphas[idx]
  cax.set_xticks(tick_vals)
  cax.set_xticklabels([f"{v:g}" for v in tick_vals], fontsize=cb_fs)
  cax.tick_params(axis="x", which="both", bottom=True, labelbottom=True, top=False, labeltop=False, length=0, width=0, pad=3)

  cax.set_frame_on(False)
  for sp in cax.spines.values():
      sp.set_visible(False)


  cax.set_xlabel(label, fontsize=cb_fs)
  cax.xaxis.set_label_position("bottom")
  return cax


def add_alpha_colorbar_vertical_single(
    ax,
    alphas,
    colors,
    label=r"$\alpha$",
    cb_fs=COLORBAR_FONTSIZE,
    pad=0.018,
    width=0.02,
    max_tick_labels: int | None = None,
    tick_values=None,
):
    fig = ax.figure
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + pad, pos.y0, width, pos.height])

    alphas = np.asarray(alphas, dtype=float)
    n = len(alphas)
    img = np.zeros((n, 1, 4))
    for i in range(n):
        img[i, 0, :] = colors[i]

    da = float(np.mean(np.diff(alphas))) if n > 1 else 1.0
    y0, y1 = float(alphas[0] - da / 2), float(alphas[-1] + da / 2)
    cax.imshow(img, origin="lower", aspect="auto", extent=(0, 1, y0, y1))
    cax.set_xlim(0, 1)
    cax.set_xticks([])

    if tick_values is not None:
        tick_vals = np.asarray(tick_values, dtype=float)
    else:
        if max_tick_labels is None or max_tick_labels >= n:
            idx_vals = np.arange(n)
        else:
            idx_vals = np.unique(np.linspace(0, n - 1, num=max_tick_labels, dtype=int))
        tick_vals = alphas[idx_vals]
    cax.set_yticks(tick_vals)
    cax.set_yticklabels([f"{v:g}" for v in tick_vals], fontsize=cb_fs)

    cax.set_title(label, fontsize=cb_fs, pad=4.0)
    for sp in cax.spines.values():
        sp.set_visible(False)
    return cax

    alphas = np.asarray(alphas, dtype=float)
    n = len(alphas)
    img = np.zeros((n, 1, 4))
    for i in range(n):
        img[i, 0, :] = colors[i]

    da = float(np.mean(np.diff(alphas))) if n > 1 else 1.0
    y0, y1 = float(alphas[0] - da / 2), float(alphas[-1] + da / 2)
    cax.imshow(img, origin="lower", aspect="auto", extent=(0, 1, y0, y1))
    cax.set_xlim(0, 1)
    cax.set_xticks([])

    if max_tick_labels is None or max_tick_labels >= n:
        idx_vals = np.arange(n)
    else:
        idx_vals = np.unique(np.linspace(0, n - 1, num=max_tick_labels, dtype=int))
    tick_vals = alphas[idx_vals]
    cax.set_yticks(tick_vals)
    cax.set_yticklabels([f"{v:g}" for v in tick_vals], fontsize=cb_fs)

    cax.set_title(label, fontsize=cb_fs, pad=4.0)
    for sp in cax.spines.values():
        sp.set_visible(False)
    return cax


def cmap_from_base_clip_whiten(
    base_name="RdYlGn",
    *,
    clip_lo=0.08,
    clip_hi=0.92,
    whiten=0.0,
    N=256,
    name=None,
):
    base = plt.get_cmap(base_name)
    xs = np.linspace(clip_lo, clip_hi, N)
    cols = base(xs)
    if whiten > 0:
        cols[:, :3] = cols[:, :3] * (1 - whiten) + 1.0 * whiten
    if name is None:
        name = f"{base_name}_clip{clip_lo:.2f}-{clip_hi:.2f}_w{whiten:.2f}"
    return ListedColormap(cols, name=name)

def make_rdylgn_clean_yellow(name="RdYlGn_clean_yellow", N=256):
    stops = [
        (0.00, "#ef3b2c"),
        (0.22, "#fb6a4a"),
        (0.50, "#ffeb3b"),
        (0.78, "#a1d99b"),
        (1.00, "#31a354"),
    ]
    return LinearSegmentedColormap.from_list(name, stops, N=N)

try:
    CMAPS
except NameError:
    CMAPS = {}

CMAPS["RdYlGn_soft"] = cmap_from_base_clip_whiten("RdYlGn", clip_lo=0.08, clip_hi=0.92, whiten=0.00, name="RdYlGn_soft")
CMAPS["RdYlGn_soft_pastel"] = cmap_from_base_clip_whiten("RdYlGn", clip_lo=0.08, clip_hi=0.92, whiten=0.10, name="RdYlGn_soft_pastel")
CMAPS["RdYlGn_clean_yellow"] = make_rdylgn_clean_yellow()
for k in ("RdYlGn_soft", "RdYlGn_soft_pastel", "RdYlGn_clean_yellow"):
    CMAPS[k] = CMAPS[k].copy()
    CMAPS[k].set_bad(color="lightgray")

def center_shrink_axis(ax, width_scale=0.62, height_scale=0.75):
    pos = ax.get_position()
    new_w = pos.width * width_scale
    new_h = pos.height * height_scale
    new_x = pos.x0 + (pos.width - new_w) / 2
    new_y = pos.y0 + (pos.height - new_h) / 2
    ax.set_position([new_x, new_y, new_w, new_h])

def get_asym_linestyle(mode="dashed", dash_len=6.0, gap_len=3.0, dot_len=0.8):
    if mode == "dash-dot-dash":
        return (0, (dash_len, gap_len, dot_len, gap_len, dash_len, gap_len))
    return (0, (dash_len, gap_len))

def _safe_divide(a, b, eps=1e-12):
    return np.asarray(a, dtype=float) / np.maximum(np.asarray(b, dtype=float), eps)

def _float_list(xs):
    return [float(x) for x in xs]

WHITE_SOFT_RED_CMAP = LinearSegmentedColormap.from_list(
    "white_soft_red", ["#ffffff", CMAPS["RdYlGn_soft_pastel"](0.02)], N=256
)

GOLD_CMAP_OPTIONS = {
    "classic_gold": LinearSegmentedColormap.from_list(
        "white_to_classic_gold", ["#ffffff", "#d4af37"], N=256
    ),
    "amber_gold": LinearSegmentedColormap.from_list(
        "white_to_amber_gold", ["#ffffff", "#ffbf00"], N=256
    ),
    "bronze_gold": LinearSegmentedColormap.from_list(
        "white_to_bronze_gold", ["#ffffff", "#b8860b"], N=256
    ),
}
for _name, _cmap in GOLD_CMAP_OPTIONS.items():
    GOLD_CMAP_OPTIONS[_name] = _cmap.copy()
    GOLD_CMAP_OPTIONS[_name].set_bad(color="lightgray")

ZETA_GOLD_CMAP = GOLD_CMAP_OPTIONS["classic_gold"]

# -------------------- configuration dataclasses --------------------

@dataclass
class ModelConfig:
    model_name: str = "vit_large_patch16_224"
    depth: int = 24
    num_classes: int = 100
    img_size: int = 224
    replace_gelu_with_relu: bool = True
    inplace_relu: bool = False
    seed: int = 0

@dataclass
class MeanFieldConfig:
    sigma_w1: float = 0.64
    sigma_w2: float = 1.28
    sigma_o: float = 0.64
    sigma_v: float = 0.64
    sigma_a: float = 0.64 * 0.64

@dataclass
class PermTokenConfig:
    batch_size: int = 32
    q0_init: float = 1.0
    p0_init: float = 0.2
    alphas: tuple = tuple(np.arange(0.1, 2.0 + 1e-9, 0.2).astype(float))
    perm_start_index: int = 0
    perm_n_replace: int | None = None
    perm_seed: int = 1234
    perm_random_rotate: bool = True
    theory_n_tokens_override: int | None = None
    num_model_inits: int = 1
    direct_layers: tuple | None = None
    direct_source_block: int = 1

@dataclass
class APJNCifarConfig:
    input_source: str = "cifar"  # "cifar" | "equiangular"
    batch_size: int = 1
    cifar_batch_seed: int = 0
    cifar_batch_draw_index: int = 0
    cifar_std_threshold: float = 0.8
    cifar_max_epochs_to_search: int = 20
    equiangular_q0: float = 1.0
    equiangular_p0: float = 0.0
    equiangular_seed: int = 0
    equiangular_random_rotate: bool = True
    apjn_layers: tuple = (0, 6, 12, 18, 24)
    direct_layers: tuple = (2, 4, 6)
    direct_source_block: int = 1
    alphas: tuple = tuple(np.arange(0.1, 2.0 + 1e-9, 0.2).astype(float))
    j_num_draws: int = 10
    j_normalize_by: str = "Y"  # "Y" | "X" | "none"
    num_model_inits: int = 1

@dataclass
class APJNFitConfig:
    metric: str = "mse"   # "mse" | "mape"
    q0_values: tuple | None = None
    p0_values: tuple | None = None
    q0_num: int = 41
    p0_num: int = 41
    separate_panel_d_fits: bool = False
    preln_scale_values: tuple | None = None
    preln_scale_num: int = 41
    refine_radius: float = 0.2
    rescale_vit_preln_apjn: bool = False

@dataclass
class PanelDConfig:
    num_layers: int = 512
    q0: float = 1.0
    p0: float = 0.5
    n_tokens: int = 196
    sigma_w1: float = 0.64
    sigma_w2: float = 1.28
    sigma_o: float = 0.64
    sigma_v: float = 0.64
    sigma_a: float = 0.64 * 0.64


@dataclass
class AsymptoticHeatmapConfig:
    alpha: float = 0.5
    sigma21_min: float = 0.0
    sigma21_max: float = 4.0
    sigmaOV_min: float = 0.0
    sigmaOV_max: float = 4.0
    grid_size: int = 10

@dataclass
class FinalThreePanelStyleConfig:
    figsize: tuple = (13.8, 6.0)
    panel_wspace: float = 0.32
    theory_vit_wspace: float = 0.18
    panel_row_hspace: float = 0.32
    row_label_offset: float = 0.01
    panel_label_xoffset: float = 0.02
    colorbar_pad: float = 0.0
    annotation_fs: float = 10.0
    perm_marker_size: float = 3.0
    vit_marker_size: float = 4.6
    line_width: float = 1.4
    asym_line_width: float = 1.2
    asym_style_mode: str = "dashed"  # or "dash-dot-dash"
    asym_dash_len: float = 3.0
    asym_gap_len: float = 6.0
    asym_dot_len: float = 0.8
    title_fs: float = 11
    label_fs: float = 11
    tick_fs: float = 10
    alpha_legend_fs: float = 10
    colorbar_max_ticks: int | None = None
    save_path: str | None = None
    three_panel_save_path: str | None = None

@dataclass
class HeatmapFigureStyleConfig:
    figsize: tuple = (14.0, 5.6)
    panel_ab_wspace: float = 0.4
    panel_b_heatmap_wspace: float = 0.8
    panel_cd_hspace: float = 0.44
    panel_title_offset: float = 0.05
    annotation_fs: float = 10.0
    direct_marker_area: float = 90.0
    average_direct_markers_across_inits: bool = False
    line_width: float = 1.4
    asym_line_width: float = 1.2
    asym_style_mode: str = "dashed"  # or "dash-dot-dash"
    asym_dash_len: float = 3.0
    asym_gap_len: float = 6.0
    asym_dot_len: float = 0.8
    title_fs: float = 11
    label_fs: float = 11
    tick_fs: float = 10
    alpha_legend_fs: float = 10
    alpha_colorbar_tick_values: tuple | None = None
    alpha_cbar_pad: float = 0.02
    alpha_cbar_width: float = 0.018
    lambda_title_shift: float = 0.5
    zeta_title_shift: float = 0.5
    save_path: str | None = None

def cfg_to_dict(cfg):
    out = asdict(cfg)
    for k, v in list(out.items()):
        if isinstance(v, tuple):
            out[k] = list(v)
    return out

# -------------------- theory: q, p, p/q, chi, J --------------------

def kappa_relu_np(rho):
    rho = np.clip(rho, -1.0, 1.0)
    return (1.0 / (2.0 * np.pi)) * (
        np.sqrt(np.maximum(0.0, 1.0 - rho**2)) + rho * (np.pi - np.arccos(rho))
    )

def tilde_q_erf_np(q, alpha):
    x = (2.0 * alpha**2 * q) / (1.0 + 2.0 * alpha**2 * q)
    x = np.clip(x, -1.0, 1.0)
    return (2.0 / np.pi) * np.arcsin(x)

def tilde_p_erf_np(q, p, alpha):
    x = (2.0 * alpha**2 * p) / (1.0 + 2.0 * alpha**2 * q)
    x = np.clip(x, -1.0, 1.0)
    return (2.0 / np.pi) * np.arcsin(x)

def erf_prime_sq_expect_np(q, alpha):
    return (4.0 * alpha**2 / np.pi) * (1.0 / np.sqrt(1.0 + 4.0 * alpha**2 * q))

def simulate_recursions_full(
    num_layers: int,
    p0: float,
    n_tokens: int,
    mode: str,
    alpha: float = 1.0,
    sigma_w1: float = 0.64,
    sigma_w2: float = 1.28,
    sigma_o: float = 0.64,
    sigma_v: float = 0.64,
    sigma_a: float = 0.64 * 0.64,
    q0: float = 1.0,
):
    L = int(num_layers)
    n = float(n_tokens)
    eps = 1e-12

    q = np.zeros(L + 1, dtype=float)
    p = np.zeros(L + 1, dtype=float)
    chi_att = np.zeros(L, dtype=float)
    chi_mlp = np.zeros(L, dtype=float)

    q[0], p[0] = float(q0), float(p0)

    att_scale = (sigma_o ** 2) * (sigma_v ** 2)
    mlp_scale = (sigma_w1 ** 2) * (sigma_w2 ** 2)

    for l in range(L):
        ql, pl = q[l], p[l]

        if mode.lower() == "erf":
            uq = tilde_q_erf_np(ql, alpha)
            up = tilde_p_erf_np(ql, pl, alpha)

            chi_att[l] = 1.0

            beta = np.exp((sigma_a ** 2) * uq * (up - uq))
            gamma = np.exp((sigma_a ** 2) * up * (up - uq))
            q_mix = (uq + (n - 1.0) * up * beta) / (1.0 + (n - 1.0) * beta)
            p_mix = (uq + (n - 1.0) * up * gamma) / (1.0 + (n - 1.0) * gamma)

            qh = ql + att_scale * q_mix
            ph = pl + att_scale * p_mix

            chi_mlp[l] = 1.0 + mlp_scale * (2.0 * alpha**2 / np.pi) * (1.0 / np.sqrt(1.0 + 4.0 * alpha**2 * qh))

            u_half = tilde_q_erf_np(qh, alpha)
            v_half = tilde_p_erf_np(qh, ph, alpha)
            rho_half = np.clip(v_half / (u_half + eps), -1.0, 1.0)

            dq_mlp = 0.5 * mlp_scale * u_half
            dp_mlp = mlp_scale * u_half * kappa_relu_np(rho_half)

            q[l + 1] = qh + dq_mlp
            p[l + 1] = ph + dp_mlp

        elif mode.lower() == "layernorm":
            rho = np.clip(pl / (ql + eps), -1.0, 1.0)
            beta = np.exp((sigma_a ** 2) * (rho - 1.0))
            gamma = np.exp((sigma_a ** 2) * rho * (rho - 1.0))

            qprime = 1.0 / (ql + eps)
            chi_att[l] = 1.0

            q_mix = (1.0 + (n - 1.0) * rho * beta) / (1.0 + (n - 1.0) * beta)
            p_mix = (1.0 + (n - 1.0) * rho * gamma) / (1.0 + (n - 1.0) * gamma)

            qh = ql + att_scale * q_mix
            ph = pl + att_scale * p_mix

            chi_mlp[l] = 1.0 + mlp_scale / (2.0 * (qh + eps))

            rho_half = np.clip(ph / (qh + eps), -1.0, 1.0)
            dq_mlp = 0.5 * mlp_scale
            dp_mlp = mlp_scale * kappa_relu_np(rho_half)

            q[l + 1] = qh + dq_mlp
            p[l + 1] = ph + dp_mlp
        else:
            raise ValueError("mode must be 'erf' or 'layernorm'")

    chi = chi_att * chi_mlp
    J = np.ones(L + 1, dtype=float)
    for l in range(L - 1, -1, -1):
        J[l] = J[l + 1] * chi[l]

    J_direct = np.ones(L + 1, dtype=float)
    for l in range(L):
        J_direct[l + 1] = J_direct[l] * chi[l]

    return {
        "q": q,
        "p": p,
        "p_over_q": _safe_divide(p, q),
        "chi_att": chi_att,
        "chi_mlp": chi_mlp,
        "chi": chi,
        "J": J,
        "J_direct": J_direct,
    }

def compute_theory_qp_bundle(
    *,
    num_layers: int,
    alphas,
    n_tokens: int,
    q0: float,
    p0: float,
    mean_field_cfg: MeanFieldConfig,
):
    alphas = np.asarray(alphas, dtype=float)
    preln = simulate_recursions_full(
        num_layers=num_layers,
        q0=q0,
        p0=p0,
        n_tokens=n_tokens,
        mode="layernorm",
        sigma_w1=mean_field_cfg.sigma_w1,
        sigma_w2=mean_field_cfg.sigma_w2,
        sigma_o=mean_field_cfg.sigma_o,
        sigma_v=mean_field_cfg.sigma_v,
        sigma_a=mean_field_cfg.sigma_a,
    )
    derf = {}
    for a in alphas:
        derf[float(a)] = simulate_recursions_full(
            num_layers=num_layers,
            q0=q0,
            p0=p0,
            n_tokens=n_tokens,
            mode="erf",
            alpha=float(a),
            sigma_w1=mean_field_cfg.sigma_w1,
            sigma_w2=mean_field_cfg.sigma_w2,
            sigma_o=mean_field_cfg.sigma_o,
            sigma_v=mean_field_cfg.sigma_v,
            sigma_a=mean_field_cfg.sigma_a,
        )
    return {
        "l": np.arange(num_layers + 1, dtype=int),
        "preln": preln,
        "derf": derf,
    }

# -------------------- panel (d) asymptotics --------------------

def _p_of_c(c):
    c = np.clip(c, 0.0, 1.0)
    return (2.0 / np.pi) * np.arcsin(c)

def _kappa_fixed_point(x):
    x = np.clip(x, -1.0, 1.0)
    return (1.0 / (2.0 * np.pi)) * (
        np.sqrt(np.maximum(0.0, 1.0 - x**2)) + x * (np.pi - np.arccos(x))
    )

def _bisect_root(fun, a, b, tol=1e-12, max_iter=200):
    fa, fb = fun(a), fun(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0:
        raise ValueError("Bisection requires a sign change on [a, b].")

    lo, hi = a, b
    flo = fa
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = fun(mid)
        if abs(fmid) <= tol or (hi - lo) <= tol:
            return mid
        if flo * fmid <= 0:
            hi = mid
        else:
            lo = mid
            flo = fmid
    return 0.5 * (lo + hi)

def _find_p_star(sigma_OV, sigma_21, grid_n=4001):
    def g(c):
        pc = _p_of_c(c)
        num = (sigma_OV**2) * pc + (sigma_21**2) * _kappa_fixed_point(pc)
        den = (sigma_OV**2) * pc + (sigma_21**2) / 2.0
        return num / den

    def f(c):
        return g(c) - c

    xs = np.linspace(0.0, 1.0, grid_n)
    fs = np.array([f(x) for x in xs])

    roots = []
    near = np.where(np.abs(fs) < 1e-8)[0]
    for idx in near:
        roots.append(xs[idx])

    for i in range(grid_n - 1):
        a, b = xs[i], xs[i + 1]
        fa, fb = fs[i], fs[i + 1]
        if fa == 0.0:
            roots.append(a)
        if fa * fb < 0:
            roots.append(_bisect_root(f, a, b))

    roots = np.array(sorted(roots))
    dedup = []
    for r in roots:
        if not dedup or abs(r - dedup[-1]) > 1e-7:
            dedup.append(r)
    roots = np.array(dedup)

    if roots.size == 0:
        roots = np.array([xs[np.argmin(np.abs(fs))]])

    cand = roots[roots < (1.0 - 1e-10)]
    c_star = float(np.max(cand)) if cand.size > 0 else float(xs[xs < 1.0][-1])
    p_star = float(_p_of_c(c_star))
    return p_star, c_star, roots

def _logJ_from_chi(chi):
    eps = 1e-300
    chi = np.asarray(chi, dtype=float)
    return np.concatenate([[0.0], np.cumsum(np.log(np.maximum(chi, eps)))])

def _fit_log_amplitude(logJ_data, log_shape, fit_frac=0.4):
    idx = np.arange(logJ_data.size)
    start = max(1, int(np.floor((1.0 - fit_frac) * (logJ_data.size - 1))))
    mask = (idx >= start) & np.isfinite(logJ_data) & np.isfinite(log_shape)
    if not np.any(mask):
        return 0.0
    return float(np.mean(logJ_data[mask] - log_shape[mask]))

def _safe_exp(x):
    return np.exp(np.clip(x, -700.0, 700.0))

def _prepare_panel_d_empirical_points(cifar_bundle):
    if cifar_bundle is None:
        return None
    preln_pack = (cifar_bundle.get("preln_with_J") or {})
    derf_pack = (cifar_bundle.get("derf_pack_with_J") or {})
    eps = 1e-30

    def _prep_dict(raw_dict):
        if not raw_dict:
            return None
        layers = []
        vals = []
        mean_layers = []
        mean_vals = []
        for layer in sorted(int(k) for k in raw_dict.keys()):
            raw_val = raw_dict[int(layer)]
            arr = np.atleast_1d(np.asarray(raw_val, dtype=float)).reshape(-1)
            if arr.size == 0:
                continue
            layers.extend([layer] * arr.size)
            vals.extend(arr.tolist())
            mean_layers.append(layer)
            mean_vals.append(float(arr.mean()))
        if not vals:
            return None
        layers = np.asarray(layers, dtype=float)
        vals = np.asarray(vals, dtype=float)
        mean_layers = np.asarray(mean_layers, dtype=float)
        mean_vals = np.asarray(mean_vals, dtype=float)
        return {
            "layers": layers,
            "J": vals,
            "logJ_sq": np.log(np.maximum(vals, eps)) ** 2,
            "layers_mean": mean_layers,
            "J_mean": mean_vals,
            "logJ_sq_mean": np.log(np.maximum(mean_vals, eps)) ** 2,
        }

    pre_direct = _prep_dict(preln_pack.get("direct_J_raw"))
    derf_direct = {}
    for entry in derf_pack.get("results", []):
        a = entry.get("alpha")
        if a is None:
            continue
        dire = _prep_dict(entry.get("direct_J_raw"))
        if dire:
            derf_direct[float(a)] = dire

    if not (pre_direct or derf_direct):
        return None

    return {
        "preln": pre_direct,
        "derf": derf_direct,
        "direct_source_block": int(cifar_bundle.get("direct_source_block", 1)),
        "num_model_inits": int(cifar_bundle.get("num_model_inits", 1)),
    }

def build_panel_d_curves(panel_d_cfg: PanelDConfig, alphas, cifar_panel_bundle=None, panel_d_fit_bundle=None):
    alphas = np.asarray(alphas, dtype=float)
    mf = MeanFieldConfig(
        sigma_w1=panel_d_cfg.sigma_w1,
        sigma_w2=panel_d_cfg.sigma_w2,
        sigma_o=panel_d_cfg.sigma_o,
        sigma_v=panel_d_cfg.sigma_v,
        sigma_a=panel_d_cfg.sigma_a,
    )

    theory_q0 = float(panel_d_cfg.q0)
    theory_p0 = float(panel_d_cfg.p0)
    preln_scale_C = 1.0
    if panel_d_fit_bundle:
        theory_q0 = float(panel_d_fit_bundle.get("q0", theory_q0))
        theory_p0 = float(panel_d_fit_bundle.get("p0", theory_p0))
        preln_scale_C = float(panel_d_fit_bundle.get("preln_scale_C", 1.0))

    L = int(panel_d_cfg.num_layers)
    sigma_OV = mf.sigma_o * mf.sigma_v
    sigma_21 = mf.sigma_w1 * mf.sigma_w2
    zeta = ((sigma_21**2) / 2.0) / (((sigma_21**2) / 2.0) + sigma_OV**2)
    p_star, c_star, roots = _find_p_star(sigma_OV=sigma_OV, sigma_21=sigma_21)

    preln = simulate_recursions_full(
        num_layers=L,
        q0=theory_q0,
        p0=theory_p0,
        n_tokens=panel_d_cfg.n_tokens,
        mode="layernorm",
        sigma_w1=mf.sigma_w1,
        sigma_w2=mf.sigma_w2,
        sigma_o=mf.sigma_o,
        sigma_v=mf.sigma_v,
        sigma_a=mf.sigma_a,
    )
    logJ_ln = _logJ_from_chi(preln["chi"])
    l_arr = np.arange(L + 1, dtype=float)
    l_pos = np.arange(1, L + 1, dtype=float)

    if preln_scale_C != 1.0:
        logJ_ln[1:] = logJ_ln[1:] + np.log(preln_scale_C)

    log_shape_ln = np.zeros(L + 1, dtype=float)
    log_shape_ln[1:] = zeta * np.log(l_pos)
    b_ln = _fit_log_amplitude(logJ_ln, log_shape_ln)
    logJ_ln_asym = np.zeros(L + 1, dtype=float)
    logJ_ln_asym[1:] = b_ln + zeta * np.log(l_pos)

    derf_curves = []
    for a in alphas:
        derf = simulate_recursions_full(
            num_layers=L,
            q0=theory_q0,
            p0=theory_p0,
            n_tokens=panel_d_cfg.n_tokens,
            mode="erf",
            alpha=float(a),
            sigma_w1=mf.sigma_w1,
            sigma_w2=mf.sigma_w2,
            sigma_o=mf.sigma_o,
            sigma_v=mf.sigma_v,
            sigma_a=mf.sigma_a,
        )
        logJ_d = _logJ_from_chi(derf["chi"])
        C = float(a) * 2.0 / np.pi
        lam = (((sigma_21**2) / 2.0) + sigma_OV**2 * p_star) / (C**2 * sigma_21**4)

        log_shape_d = np.zeros(L + 1, dtype=float)
        # log_shape_d[1:] = np.sqrt(l_pos / lam) - (1.0 / (8.0 * lam)) * np.log(l_pos)
        log_shape_d[1:] = np.sqrt(l_pos / lam)
        b_d = _fit_log_amplitude(logJ_d, log_shape_d)
        logJ_d_asym = np.zeros(L + 1, dtype=float)
        logJ_d_asym[1:] = b_d + log_shape_d[1:]

        derf_curves.append({
            "alpha": float(a),
            "lambda": float(lam),
            "logJ": logJ_d,
            "J": _safe_exp(logJ_d),
            "logJ_asym": logJ_d_asym,
            "J_asym": _safe_exp(logJ_d_asym),
        })

    return {
        "L": L,
        "l_arr": l_arr,
        "zeta": float(zeta),
        "p_star": float(p_star),
        "c_star": float(c_star),
        "roots": roots,
        "fit_metadata": {
            "q0": theory_q0,
            "p0": theory_p0,
            "preln_scale_C": preln_scale_C,
        },
        "preln": {
            "logJ": logJ_ln,
            "J": _safe_exp(logJ_ln),
            "logJ_asym": logJ_ln_asym,
            "J_asym": _safe_exp(logJ_ln_asym),
        },
        "derf": derf_curves,
        "empirical": _prepare_panel_d_empirical_points(cifar_panel_bundle),
    }

def compute_asymptotic_heatmaps(cfg: AsymptoticHeatmapConfig):
    grid_size = max(2, int(cfg.grid_size))
    sigma21_vals = np.linspace(float(cfg.sigma21_min), float(cfg.sigma21_max), grid_size, dtype=float)
    sigmaOV_vals = np.linspace(float(cfg.sigmaOV_min), float(cfg.sigmaOV_max), grid_size, dtype=float)

    lam_inv = np.zeros((sigmaOV_vals.size, sigma21_vals.size), dtype=float)
    zeta = np.zeros_like(lam_inv)

    eps = 1e-8
    alpha = float(cfg.alpha)
    C = 2.0 * alpha / np.pi

    for j, sigma_OV in enumerate(sigmaOV_vals):
        for i, sigma_21 in enumerate(sigma21_vals):
            denom = (sigma_21**2) / 2.0 + sigma_OV**2
            zeta[j, i] = ((sigma_21**2) / 2.0) / denom if denom > eps else 0.0

            if sigma_21 <= eps or C <= eps:
                lam_inv[j, i] = 0.0
                continue

            p_star, _, _ = _find_p_star(sigma_OV=sigma_OV, sigma_21=sigma_21)

            lam = (((sigma_21**2) / 2.0) + sigma_OV**2 * p_star) / (C**2 * sigma_21**4)
            lam_inv[j, i] = 0.0 if lam <= eps or not np.isfinite(lam) else 1.0 / lam

    return {
        "sigma21_vals": sigma21_vals,
        "sigmaOV_vals": sigmaOV_vals,
        "lam_inv": lam_inv,
        "zeta": zeta,
        "alpha": alpha,
    }

# -------------------- model / token / data helpers --------------------

criterion = nn.CrossEntropyLoss()

def replace_gelu_with_relu(module: nn.Module, *, inplace_relu: bool = False) -> nn.Module:
    for name, child in module.named_children():
        if isinstance(child, nn.GELU):
            setattr(module, name, nn.ReLU(inplace=inplace_relu))
        else:
            replace_gelu_with_relu(child, inplace_relu=inplace_relu)
    return module

def build_vit(model_cfg: ModelConfig, *, use_derf: bool):
    _create_model = _ensure_create_model()
    create_kwargs = dict(
        pretrained=False,
        num_classes=model_cfg.num_classes,
        global_pool="avg",
        drop_path_rate=0.0,
    )
    if model_cfg.depth is not None:
        create_kwargs["depth"] = int(model_cfg.depth)

    try:
        model = _create_model(model_cfg.model_name, **create_kwargs)
    except TypeError as e:
        raise TypeError(
            f"create_model({model_cfg.model_name!r}, depth={model_cfg.depth}) failed. "
            f"Choose a model family whose constructor accepts `depth`."
        ) from e

    if model_cfg.replace_gelu_with_relu:
        replace_gelu_with_relu(model, inplace_relu=model_cfg.inplace_relu)

    if use_derf:
        _, _convert_ln_to_derf = _ensure_dynamic_tanh()
        model = _convert_ln_to_derf(model)

    model.eval().to(DEVICE)
    return model

@torch.no_grad()
def set_all_derf_alpha_(model: nn.Module, alpha_value: float) -> int:
    _DynamicErf, _ = _ensure_dynamic_tanh()
    n = 0
    a = float(alpha_value)
    for m in model.modules():
        if isinstance(m, _DynamicErf):
            m.alpha_init_value = a
            m.alpha.data.fill_(a)
            n += 1
    return n

def get_vit_seq_len_and_dim(model: nn.Module) -> tuple[int, int]:
    d = int(getattr(model, "embed_dim"))
    n_patches = int(getattr(model.patch_embed, "num_patches"))
    has_cls = hasattr(model, "cls_token") and (model.cls_token is not None)
    N = n_patches + (1 if has_cls else 0)
    return N, d

def _random_orthogonal(d: int, device, dtype, generator: torch.Generator):
    M = torch.randn(d, d, device=device, dtype=dtype, generator=generator)
    Q, R = torch.linalg.qr(M, mode="reduced")
    diag = torch.diagonal(R, 0)
    signs = torch.sign(diag)
    signs[signs == 0] = 1.0
    Q = Q * signs
    return Q

def make_equiangular_tokens(
    N: int,
    d: int,
    q0: float,
    p0: float,
    *,
    device,
    dtype,
    seed: int,
    random_rotate: bool = True,
):
    if N < 2:
        raise ValueError("N must be >= 2")
    if q0 <= 0:
        raise ValueError("q0 must be > 0")
    if p0 > q0 + 1e-12:
        raise ValueError("Need p0 <= q0 so that the initial correlation p0/q0 is <= 1")

    rho = float(p0 / q0)
    rho_min = -1.0 / (N - 1)
    if rho < rho_min - 1e-8 or rho >= 1.0:
        raise ValueError(
            f"Need p0/q0 in [{rho_min}, 1). Got p0/q0 = {rho} for N={N}."
        )

    lam1 = d * (1.0 - rho)
    lam2 = d * (1.0 + (N - 1) * rho)

    g = torch.Generator(device=device)
    g.manual_seed(int(seed))

    u = torch.ones(N, 1, device=device, dtype=dtype) / math.sqrt(N)
    M = torch.randn(N, N - 1, device=device, dtype=dtype, generator=g)
    M = M - u @ (u.T @ M)
    V, _ = torch.linalg.qr(M, mode="reduced")

    parts = []
    eps = 1e-12
    if lam1 > eps:
        parts.append(math.sqrt(lam1) * V)
    if lam2 > eps:
        parts.append(math.sqrt(lam2) * u)

    A = torch.cat(parts, dim=1)
    r = A.shape[1]
    if d < r:
        raise ValueError(f"Need d >= rank={r} (got d={d}).")

    X = torch.cat([A, torch.zeros(N, d - r, device=device, dtype=dtype)], dim=1)
    if random_rotate:
        R = _random_orthogonal(d, device=device, dtype=dtype, generator=g)
        X = X @ R

    # Base construction has self-dot = d and pairwise dot = rho * d.
    # Scale by sqrt(q0) so empirical self-dot/d ≈ q0 and pairwise dot/d ≈ p0.
    X = math.sqrt(q0) * X
    return X

def _validate_replacement(N_seq: int, start_index: int, n_replace_tokens: int | None):
    start = int(start_index)
    if start < 0 or start >= N_seq:
        raise ValueError(f"start_index must be in [0, {N_seq - 1}], got {start}")
    n_rep = N_seq - start if (n_replace_tokens is None) else int(n_replace_tokens)
    if n_rep <= 0:
        raise ValueError("n_replace_tokens must be > 0 (or None).")
    if start + n_rep > N_seq:
        raise ValueError(f"Replacement slice [{start}:{start + n_rep}] exceeds seq_len={N_seq}")
    return start, n_rep

def capture_X_list_and_logits(
    model: nn.Module,
    images: torch.Tensor,
    *,
    inject_perm_tokens: bool,
    perm_q0: float,
    perm_p0: float,
    perm_start_index: int,
    perm_n_replace: int | None,
    perm_seed: int,
    perm_random_rotate: bool,
    block0_input_override: torch.Tensor | None = None,
):
    L = len(model.blocks)
    X0_ref = {"x0": None}
    outs = [None] * L
    handles = []

    N_seq, d = get_vit_seq_len_and_dim(model)

    x_base_2d = None
    if inject_perm_tokens and block0_input_override is not None:
        raise ValueError("Cannot use inject_perm_tokens together with block0_input_override.")
    if inject_perm_tokens:
        start, n_rep = _validate_replacement(N_seq, perm_start_index, perm_n_replace)
        x_base_2d = make_equiangular_tokens(
            N=n_rep,
            d=d,
            q0=float(perm_q0),
            p0=float(perm_p0),
            device=DEVICE,
            dtype=torch.float32,
            seed=int(perm_seed),
            random_rotate=bool(perm_random_rotate),
        ).detach()

    def pre_hook_block0(mod, inputs):
        x_in = inputs[0]
        B, N, D = x_in.shape

        if block0_input_override is not None:
            x_override = block0_input_override.to(device=x_in.device, dtype=x_in.dtype)
            if tuple(x_override.shape) != (B, N, D):
                raise ValueError(
                    f"block0_input_override must have shape {(B, N, D)}, got {tuple(x_override.shape)}"
                )
            x0 = x_override.detach().clone().requires_grad_(True)
        else:
            x0 = x_in.detach().clone().requires_grad_(True)

        if inject_perm_tokens:
            start, n_rep = _validate_replacement(N, perm_start_index, perm_n_replace)
            x_rep = x_base_2d.unsqueeze(0).expand(B, -1, -1).clone().detach()
            with torch.no_grad():
                x0[:, start:start + n_rep, :] = x_rep

        X0_ref["x0"] = x0
        return (x0,)

    def hook_block_i(i: int):
        def _hook(mod, inputs, output):
            outs[i] = output[0] if isinstance(output, (tuple, list)) else output
        return _hook

    handles.append(model.blocks[0].register_forward_pre_hook(pre_hook_block0))
    for i in range(L):
        handles.append(model.blocks[i].register_forward_hook(hook_block_i(i)))

    try:
        logits = model(images.to(DEVICE, dtype=torch.float32, non_blocking=False))
        X0 = X0_ref["x0"]
        if X0 is None or any(o is None for o in outs):
            raise RuntimeError("Failed to capture block tensors.")
        return [X0] + outs, logits
    finally:
        for h in handles:
            h.remove()

def get_cifar_batch(
    batch_size: int,
    img_size: int,
    num_classes: int,
    *,
    loader_seed: int,
    draw_index: int,
    std_threshold: float,
    max_epochs_to_search: int,
):
    stream_state = _get_or_create_cifar_stream(
        batch_size=int(batch_size),
        img_size=int(img_size),
        num_classes=int(num_classes),
        loader_seed=int(loader_seed),
        std_threshold=float(std_threshold),
    )
    accepted_batches = stream_state["accepted_batches"]
    target_index = int(draw_index)

    if target_index < len(accepted_batches):
        entry = accepted_batches[target_index]
        return entry["samples"].clone(), entry["targets"].clone(), dict(entry["meta"])

    loader = stream_state["loader"]
    iterator = stream_state["iterator"]
    epoch = int(stream_state["epoch"])

    while len(accepted_batches) <= target_index:
        if epoch >= int(max_epochs_to_search):
            break
        try:
            samples, targets = next(iterator)
        except StopIteration:
            epoch += 1
            if epoch >= int(max_epochs_to_search):
                break
            iterator = iter(loader)
            stream_state["iterator"] = iterator
            stream_state["epoch"] = epoch
            continue

        if float(samples.std()) <= float(std_threshold):
            continue

        accepted_idx = len(accepted_batches)
        meta = {
            "loader_seed": int(loader_seed),
            "draw_index": int(accepted_idx),
            "std_threshold": float(std_threshold),
            "accepted_before_return": int(accepted_idx),
        }
        accepted_batches.append({
            "samples": samples.detach().cpu().clone(),
            "targets": targets.detach().cpu().clone(),
            "meta": meta,
        })

    stream_state["iterator"] = iterator
    stream_state["epoch"] = epoch

    if target_index < len(accepted_batches):
        entry = accepted_batches[target_index]
        return entry["samples"].clone(), entry["targets"].clone(), dict(entry["meta"])

    raise RuntimeError(
        f"Could not find CIFAR batch with draw_index={draw_index} after {max_epochs_to_search} epochs. "
        "Try increasing cifar_max_epochs_to_search or lowering cifar_std_threshold."
    )

def get_synth_images_batch(batch_size: int, img_size: int, num_classes: int):
    samples = torch.randn(batch_size, 3, img_size, img_size, dtype=torch.float32)
    targets = torch.randint(low=0, high=num_classes, size=(batch_size,), dtype=torch.long)
    return samples, targets, {"kind": "synthetic_images"}


def make_apjn_equiangular_block0_batch(
    *,
    batch_size: int,
    seq_len: int,
    embed_dim: int,
    q0: float,
    p0: float,
    seed: int,
    random_rotate: bool,
):
    x_base_2d = make_equiangular_tokens(
        N=int(seq_len),
        d=int(embed_dim),
        q0=float(q0),
        p0=float(p0),
        device=DEVICE,
        dtype=torch.float32,
        seed=int(seed),
        random_rotate=bool(random_rotate),
    ).detach()
    x_batch = x_base_2d.unsqueeze(0).expand(int(batch_size), -1, -1).clone().detach()
    return x_batch

# -------------------- empirical measurements --------------------

def _maybe_drop_cls(x: torch.Tensor, exclude_cls: bool):
    if exclude_cls and x.ndim == 3 and x.shape[1] > 1:
        return x[:, 1:, :]
    return x

def mean_token_sqnorm_over_d(x: torch.Tensor, exclude_cls: bool = True) -> torch.Tensor:
    x = _maybe_drop_cls(x, exclude_cls)
    return (x.pow(2).sum(dim=-1).mean()) / x.shape[-1]

def mean_all_pairs_token_dot_over_d(x: torch.Tensor, exclude_cls: bool = True) -> torch.Tensor:
    x = _maybe_drop_cls(x, exclude_cls)
    B, N, d = x.shape
    if N < 2:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    gram = (x @ x.transpose(-1, -2)) / d
    eye = torch.eye(N, device=x.device, dtype=torch.bool).unsqueeze(0)
    return gram.masked_select(~eye).mean()

def compute_empirical_qp_trajectories_for_model(model, images, perm_cfg: PermTokenConfig):
    model.eval()
    with torch.no_grad():
        X_list, _ = capture_X_list_and_logits(
            model,
            images,
            inject_perm_tokens=True,
            perm_q0=perm_cfg.q0_init,
            perm_p0=perm_cfg.p0_init,
            perm_start_index=perm_cfg.perm_start_index,
            perm_n_replace=perm_cfg.perm_n_replace,
            perm_seed=perm_cfg.perm_seed,
            perm_random_rotate=perm_cfg.perm_random_rotate,
        )

    q_vals = []
    p_vals = []
    for X in X_list:
        Xd = X.detach().to(torch.float32)
        q_vals.append(float(mean_token_sqnorm_over_d(Xd, exclude_cls=True).cpu().item()))
        p_vals.append(float(mean_all_pairs_token_dot_over_d(Xd, exclude_cls=True).cpu().item()))

    q_vals = np.asarray(q_vals, dtype=float)
    p_vals = np.asarray(p_vals, dtype=float)
    return {
        "l": np.arange(len(X_list), dtype=int),
        "q": q_vals,
        "p": p_vals,
        "p_over_q": _safe_divide(p_vals, q_vals),
        "x0_emp_q": float(q_vals[0]),
        "x0_emp_p": float(p_vals[0]),
    }

def resolve_apjn_layers(apjn_layers, depth: int):
    out = sorted(set(int(x) for x in apjn_layers))
    for l in out:
        if l < 0 or l > depth:
            raise ValueError(f"APJN layer {l} is outside [0, {depth}]")
    return out


def resolve_direct_layers(direct_layers, depth: int, source_block: int):
    if direct_layers is None:
        return []
    out = sorted(set(int(x) for x in direct_layers))
    if source_block < 0 or source_block > depth:
        raise ValueError(f"direct source block must be in [0, {depth}], got {source_block}")
    for l in out:
        if l < source_block:
            raise ValueError(f"Direct APJN layer {l} must be >= source_block={source_block}")
        if l > depth:
            raise ValueError(f"Direct APJN layer {l} is outside [0, {depth}]")
    return out

def _capture_block_outputs_for_apjn(model: nn.Module, images: torch.Tensor, block0_input_override: torch.Tensor | None = None):
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        X_list, _ = capture_X_list_and_logits(
            model,
            images,
            inject_perm_tokens=False,
            perm_q0=1.0,
            perm_p0=0.0,
            perm_start_index=0,
            perm_n_replace=None,
            perm_seed=0,
            perm_random_rotate=False,
            block0_input_override=block0_input_override,
        )
    return X_list


def _estimate_inverse_J_from_Xlist(
    X_list,
    *,
    l0_list,
    j_num_draws: int,
    j_normalize_by: str,
):
    if j_normalize_by not in ("Y", "X", "none"):
        raise ValueError("j_normalize_by must be 'Y', 'X', or 'none'")

    L = len(X_list) - 1
    Y = X_list[-1]

    def denom_for(X_l0: torch.Tensor):
        if j_normalize_by == "none":
            return 1.0
        if j_normalize_by == "Y":
            return float(Y.numel())
        return float(X_l0.numel())

    out = {}
    for l0 in l0_list:
        if l0 < 0 or l0 > L:
            raise ValueError(f"l0 must be in [0..{L}], got {l0}")
        if l0 == L:
            out[l0] = 1.0 if j_normalize_by != "none" else float(Y.numel())
            continue
        X_l0 = X_list[l0]
        den = denom_for(X_l0)
        acc = 0.0
        for _ in range(int(j_num_draws)):
            R = torch.randn_like(Y)
            s = (Y * R).sum()
            g = torch.autograd.grad(s, X_l0, retain_graph=True, create_graph=False, allow_unused=False)[0]
            acc += float(g.pow(2).sum().detach().cpu().item()) / den
        out[l0] = acc / float(j_num_draws)
    return out


def _estimate_direct_J_from_Xlist(
    X_list,
    *,
    target_layers,
    source_block_index: int,
    j_num_draws: int,
    j_normalize_by: str,
):
    if not target_layers:
        return {}
    if j_normalize_by not in ("Y", "X", "none"):
        raise ValueError("j_normalize_by must be 'Y', 'X', or 'none'")

    L = len(X_list) - 1
    if source_block_index < 0 or source_block_index > L:
        raise ValueError(f"source_block_index must be in [0..{L}], got {source_block_index}")

    source = X_list[source_block_index]

    def denom_for(X_target: torch.Tensor):
        if j_normalize_by == "none":
            return 1.0
        if j_normalize_by == "Y":
            return float(X_target.numel())
        return float(source.numel())

    out = {}
    target_layers = sorted(int(l) for l in target_layers)
    last_idx = len(target_layers) - 1

    for idx, l_abs in enumerate(target_layers):
        if l_abs < source_block_index or l_abs > L:
            raise ValueError(
                f"Direct APJN layer {l_abs} must satisfy source <= l <= {L}"
            )

        rel_l = l_abs - source_block_index
        X_target = X_list[l_abs]
        den = denom_for(X_target)
        acc = 0.0

        for draw in range(int(j_num_draws)):
            R = torch.randn_like(X_target)
            s = (X_target * R).sum()
            retain_graph = not (idx == last_idx and draw == j_num_draws - 1)
            g = torch.autograd.grad(
                s,
                source,
                retain_graph=retain_graph,
                create_graph=False,
                allow_unused=False,
            )[0]
            acc += float(g.pow(2).sum().detach().cpu().item()) / den

        out[rel_l] = acc / float(j_num_draws)

    return out


def estimate_inverse_and_direct_J_points(
    model: nn.Module,
    images: torch.Tensor,
    *,
    inverse_layers,
    direct_layers,
    direct_source_block: int,
    j_num_draws: int,
    j_normalize_by: str,
    block0_input_override: torch.Tensor | None = None,
):
    X_list = _capture_block_outputs_for_apjn(model, images, block0_input_override=block0_input_override)
    inverse = _estimate_inverse_J_from_Xlist(
        X_list,
        l0_list=inverse_layers if inverse_layers else [],
        j_num_draws=j_num_draws,
        j_normalize_by=j_normalize_by,
    ) if inverse_layers else {}
    direct = _estimate_direct_J_from_Xlist(
        X_list,
        target_layers=direct_layers if direct_layers else [],
        source_block_index=direct_source_block,
        j_num_draws=j_num_draws,
        j_normalize_by=j_normalize_by,
    ) if direct_layers else {}
    return inverse, direct


def estimate_J_points_hutchinson(
    model: nn.Module,
    images: torch.Tensor,
    *,
    l0_list,
    j_num_draws: int,
    j_normalize_by: str,
    block0_input_override: torch.Tensor | None = None,
):
    inverse, _ = estimate_inverse_and_direct_J_points(
        model,
        images,
        inverse_layers=list(int(x) for x in l0_list),
        direct_layers=None,
        direct_source_block=0,
        j_num_draws=j_num_draws,
        j_normalize_by=j_normalize_by,
        block0_input_override=block0_input_override,
    )
    return inverse


def estimate_direct_J_points_hutchinson(
    model: nn.Module,
    images: torch.Tensor,
    *,
    direct_layers,
    direct_source_block: int,
    j_num_draws: int,
    j_normalize_by: str,
    block0_input_override: torch.Tensor | None = None,
):
    _, direct = estimate_inverse_and_direct_J_points(
        model,
        images,
        inverse_layers=None,
        direct_layers=list(int(x) for x in direct_layers),
        direct_source_block=int(direct_source_block),
        j_num_draws=j_num_draws,
        j_normalize_by=j_normalize_by,
        block0_input_override=block0_input_override,
    )
    return direct

def compute_G_raw_across_layers(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, block0_input_override: torch.Tensor | None = None):
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        X_list, logits = capture_X_list_and_logits(
            model,
            images,
            inject_perm_tokens=False,
            perm_q0=1.0,
            perm_p0=0.0,
            perm_start_index=0,
            perm_n_replace=None,
            perm_seed=0,
            perm_random_rotate=False,
            block0_input_override=block0_input_override,
        )
        loss = criterion(logits, labels.to(DEVICE, non_blocking=False))
        grads = torch.autograd.grad(loss, X_list, retain_graph=False, create_graph=False, allow_unused=False)

    G = torch.empty(len(grads), dtype=torch.float32, device="cpu")
    for i, g in enumerate(grads):
        G[i] = mean_token_sqnorm_over_d(g.detach(), exclude_cls=True).cpu()
    return G.numpy()

def _average_empirical_runs(run_list):
    if not run_list:
        raise ValueError("Need at least one empirical run to average.")
    l_ref = np.asarray(run_list[0]["l"], dtype=int)
    for run in run_list[1:]:
        if not np.array_equal(np.asarray(run["l"], dtype=int), l_ref):
            raise ValueError("Empirical runs must share identical layer indices.")
    q_stack = np.stack([np.asarray(run["q"], dtype=float) for run in run_list], axis=0)
    p_stack = np.stack([np.asarray(run["p"], dtype=float) for run in run_list], axis=0)
    q_mean = q_stack.mean(axis=0)
    p_mean = p_stack.mean(axis=0)
    return {
        "l": l_ref.copy(),
        "q": q_mean,
        "p": p_mean,
        "p_over_q": _safe_divide(p_mean, q_mean),
        "x0_emp_q": float(q_mean[0]),
        "x0_emp_p": float(p_mean[0]),
    }


def _average_scalar_dicts(dict_list):
    if not dict_list:
        raise ValueError("Need at least one dict to average.")
    ref_keys = sorted(int(k) for k in dict_list[0].keys())
    for d in dict_list[1:]:
        if sorted(int(k) for k in d.keys()) != ref_keys:
            raise ValueError("All dicts must share the same layer keys.")
    out = {}
    for k in ref_keys:
        vals = [float(d[k]) for d in dict_list]
        out[int(k)] = float(np.mean(vals))
    return out

def collect_block0_input_stats(model: nn.Module, images: torch.Tensor, block0_input_override: torch.Tensor | None = None):
    model.eval()
    with torch.no_grad():
        X_list, _ = capture_X_list_and_logits(
            model,
            images,
            inject_perm_tokens=False,
            perm_q0=1.0,
            perm_p0=0.0,
            perm_start_index=0,
            perm_n_replace=None,
            perm_seed=0,
            perm_random_rotate=False,
            block0_input_override=block0_input_override,
        )
    X0 = X_list[0].detach().to(torch.float32)
    X0 = _maybe_drop_cls(X0, exclude_cls=True)
    token_vals = (X0.pow(2).sum(dim=-1) / X0.shape[-1]).reshape(-1).cpu().numpy()
    if X0.shape[1] >= 2:
        gram = (X0 @ X0.transpose(-1, -2)) / X0.shape[-1]
        mask = ~torch.eye(X0.shape[1], dtype=torch.bool, device=X0.device).unsqueeze(0)
        pairwise_vals = gram.masked_select(mask).cpu().numpy()
    else:
        pairwise_vals = np.empty(0, dtype=float)
    return {
        "token_sqnorm_over_d": token_vals,
        "pairwise_dot_over_d": pairwise_vals,
    }


def plot_block0_stats_histograms(block0_stats, *, figure_title="Distributions of inputs to transformer block 1"):
    token_vals = np.asarray(block0_stats["token_sqnorm_over_d"], dtype=float)
    pairwise_vals = np.asarray(block0_stats["pairwise_dot_over_d"], dtype=float)
    tok_mean = float(token_vals.mean()) if token_vals.size else float("nan")
    tok_std = float(token_vals.std(ddof=0)) if token_vals.size else float("nan")
    pair_mean = float(pairwise_vals.mean()) if pairwise_vals.size else float("nan")
    pair_std = float(pairwise_vals.std(ddof=0)) if pairwise_vals.size else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    axes[0].hist(token_vals, bins=40, color="#1f77b4", alpha=0.85)
    axes[0].set_title(r"Block-0 token $\Vert x\Vert^2 / d$")
    axes[0].set_xlabel("normalized token sq. norm")
    axes[0].set_ylabel("count")
    axes[0].text(0.02, 0.95, f"mean={tok_mean:.3f}, std={tok_std:.3f}", transform=axes[0].transAxes, va="top", ha="left", fontsize=10, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    if pairwise_vals.size:
        axes[1].hist(pairwise_vals, bins=40, color="#ff7f0e", alpha=0.85)
        axes[1].set_ylabel("count")
        axes[1].text(0.02, 0.95, f"mean={pair_mean:.3f}, std={pair_std:.3f}", transform=axes[1].transAxes, va="top", ha="left", fontsize=10, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    else:
        axes[1].text(0.5, 0.5, "Not enough tokens for pairwise dots", ha="center", va="center")
    axes[1].set_title("Block-0 pairwise $(x_i^T x_j)/d$")
    axes[1].set_xlabel("normalized pairwise dot")
    fig.suptitle(figure_title, y=1.02)
    fig.tight_layout()
    plt.show()


def preview_apjn_sample_stats(model_cfg: ModelConfig, apjn_cfg: APJNCifarConfig):
    input_source = str(getattr(apjn_cfg, "input_source", "cifar")).lower()
    if input_source not in ("cifar", "equiangular"):
        raise ValueError("APJNCifarConfig.input_source must be 'cifar' or 'equiangular'.")

    batch_size = max(1, int(getattr(apjn_cfg, "batch_size", 1)))
    seed_all(model_cfg.seed)
    cuda_cleanup()

    if input_source == "cifar":
        samples, _, batch_meta = get_cifar_batch(
            batch_size=batch_size,
            img_size=model_cfg.img_size,
            num_classes=model_cfg.num_classes,
            loader_seed=apjn_cfg.cifar_batch_seed,
            draw_index=apjn_cfg.cifar_batch_draw_index,
            std_threshold=apjn_cfg.cifar_std_threshold,
            max_epochs_to_search=apjn_cfg.cifar_max_epochs_to_search,
        )
        preview_model = build_vit(model_cfg, use_derf=False)
        try:
            block0_stats = collect_block0_input_stats(preview_model, samples)
        finally:
            del preview_model
            cuda_cleanup()
        return {"batch_meta": batch_meta, "input_block0_stats": block0_stats}

    samples, _, synth_meta = get_synth_images_batch(
        batch_size=batch_size,
        img_size=model_cfg.img_size,
        num_classes=model_cfg.num_classes,
    )
    preview_model = build_vit(model_cfg, use_derf=False)
    try:
        seq_len, embed_dim = get_vit_seq_len_and_dim(preview_model)
        block0_input_override = make_apjn_equiangular_block0_batch(
            batch_size=batch_size,
            seq_len=seq_len,
            embed_dim=embed_dim,
            q0=float(apjn_cfg.equiangular_q0),
            p0=float(apjn_cfg.equiangular_p0),
            seed=int(apjn_cfg.equiangular_seed),
            random_rotate=bool(apjn_cfg.equiangular_random_rotate),
        )
        block0_stats = collect_block0_input_stats(preview_model, samples, block0_input_override=block0_input_override)
    finally:
        del preview_model
        cuda_cleanup()

    return {
        "batch_meta": {
            "kind": "equiangular_block0_tokens",
            "batch_size": int(batch_size),
            "seq_len": int(seq_len),
            "embed_dim": int(embed_dim),
            "q0": float(apjn_cfg.equiangular_q0),
            "p0": float(apjn_cfg.equiangular_p0),
            "seed": int(apjn_cfg.equiangular_seed),
            "random_rotate": bool(apjn_cfg.equiangular_random_rotate),
            "backing_inputs": synth_meta,
        },
        "input_block0_stats": block0_stats,
    }


def run_perm_token_experiment(
    model_cfg: ModelConfig,
    perm_cfg: PermTokenConfig,
):
    num_inits = perm_cfg.num_model_inits if perm_cfg.num_model_inits is not None else 1
    num_inits = max(1, int(num_inits))
    seed_all(model_cfg.seed)
    cuda_cleanup()

    samples, _, batch_meta = get_synth_images_batch(
        perm_cfg.batch_size,
        model_cfg.img_size,
        model_cfg.num_classes,
    )

    alphas = [float(a) for a in np.asarray(perm_cfg.alphas, dtype=float)]
    preln_runs = []
    derf_runs = {float(a): [] for a in alphas}
    depth = None
    seq_len = None
    embed_dim = None

    base_seed = int(model_cfg.seed)
    for init_idx in range(num_inits):
        init_seed = base_seed + init_idx
        seed_all(init_seed)
        cuda_cleanup()

        preln_model = build_vit(model_cfg, use_derf=False)
        derf_model = build_vit(model_cfg, use_derf=True)

        if depth is None:
            depth = len(preln_model.blocks)
        if seq_len is None or embed_dim is None:
            seq_len, embed_dim = get_vit_seq_len_and_dim(preln_model)

        preln_runs.append(
            compute_empirical_qp_trajectories_for_model(preln_model, samples, perm_cfg)
        )
        for a in alphas:
            alpha_val = float(a)
            set_all_derf_alpha_(derf_model, alpha_val)
            derf_runs[alpha_val].append(
                compute_empirical_qp_trajectories_for_model(derf_model, samples, perm_cfg)
            )

        del preln_model
        del derf_model
        cuda_cleanup()

    preln_emp = _average_empirical_runs(preln_runs)
    derf_emp = {a: _average_empirical_runs(runs) for a, runs in derf_runs.items()}

    return {
        "model_cfg": cfg_to_dict(model_cfg),
        "perm_cfg": cfg_to_dict(perm_cfg),
        "batch_meta": batch_meta,
        "depth": int(depth),
        "seq_len": int(seq_len),
        "embed_dim": int(embed_dim),
        "n_tokens_ex_cls": int(seq_len - 1),
        "num_model_inits": int(num_inits),
        "preln": preln_emp,
        "derf": derf_emp,
    }

def run_cifar_apjn_experiment(
    model_cfg: ModelConfig,
    apjn_cfg: APJNCifarConfig,
):
    num_inits = apjn_cfg.num_model_inits if apjn_cfg.num_model_inits is not None else 1
    num_inits = max(1, int(num_inits))
    seed_all(model_cfg.seed)
    cuda_cleanup()

    input_source = str(getattr(apjn_cfg, "input_source", "cifar")).lower()
    if input_source not in ("cifar", "equiangular"):
        raise ValueError("APJNCifarConfig.input_source must be 'cifar' or 'equiangular'.")
    batch_size = max(1, int(getattr(apjn_cfg, "batch_size", 1)))

    samples = None
    targets = None
    batch_meta = None
    block0_input_override = None
    if input_source == "cifar":
        samples, targets, batch_meta = get_cifar_batch(
            batch_size=batch_size,
            img_size=model_cfg.img_size,
            num_classes=model_cfg.num_classes,
            loader_seed=apjn_cfg.cifar_batch_seed,
            draw_index=apjn_cfg.cifar_batch_draw_index,
            std_threshold=apjn_cfg.cifar_std_threshold,
            max_epochs_to_search=apjn_cfg.cifar_max_epochs_to_search,
        )

    alphas = [float(a) for a in np.asarray(apjn_cfg.alphas, dtype=float)]
    preln_runs = []
    derf_runs = {float(a): [] for a in alphas}
    pre_G_list = []
    derf_G_accum = {float(a): [] for a in alphas}
    token_sqnorm_batches = []
    pairwise_batches = []
    depth = None
    seq_len = None
    embed_dim = None
    apjn_layers = None
    direct_layers_abs = None
    direct_layers_rel = None
    direct_layers_cfg = tuple(apjn_cfg.direct_layers) if apjn_cfg.direct_layers is not None else None
    pre_direct_runs = {}
    derf_direct_runs = {float(a): {} for a in alphas}

    base_seed = int(model_cfg.seed)
    for init_idx in range(num_inits):
        init_seed = base_seed + init_idx
        seed_all(init_seed)
        cuda_cleanup()

        preln_model = build_vit(model_cfg, use_derf=False)
        derf_model = build_vit(model_cfg, use_derf=True)

        if depth is None:
            depth = len(preln_model.blocks)
            apjn_layers_raw = tuple(apjn_cfg.apjn_layers) if apjn_cfg.apjn_layers is not None else tuple()
            apjn_layers = resolve_apjn_layers(apjn_layers_raw, depth) if apjn_layers_raw else []
            seq_len, embed_dim = get_vit_seq_len_and_dim(preln_model)

            if input_source == "equiangular":
                samples, targets, synth_meta = get_synth_images_batch(
                    batch_size=batch_size,
                    img_size=model_cfg.img_size,
                    num_classes=model_cfg.num_classes,
                )
                block0_input_override = make_apjn_equiangular_block0_batch(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    embed_dim=embed_dim,
                    q0=float(apjn_cfg.equiangular_q0),
                    p0=float(apjn_cfg.equiangular_p0),
                    seed=int(apjn_cfg.equiangular_seed),
                    random_rotate=bool(apjn_cfg.equiangular_random_rotate),
                )
                batch_meta = {
                    "kind": "equiangular_block0_tokens",
                    "batch_size": int(batch_size),
                    "seq_len": int(seq_len),
                    "embed_dim": int(embed_dim),
                    "q0": float(apjn_cfg.equiangular_q0),
                    "p0": float(apjn_cfg.equiangular_p0),
                    "seed": int(apjn_cfg.equiangular_seed),
                    "random_rotate": bool(apjn_cfg.equiangular_random_rotate),
                    "backing_inputs": synth_meta,
                }

            direct_targets = direct_layers_cfg if direct_layers_cfg is not None else tuple(apjn_layers)
            if direct_targets:
                direct_layers_abs = resolve_direct_layers(
                    direct_targets,
                    depth,
                    apjn_cfg.direct_source_block,
                )
                direct_layers_rel = [int(l - apjn_cfg.direct_source_block) for l in direct_layers_abs]
            else:
                direct_layers_abs = []
                direct_layers_rel = []

            pre_direct_runs = {int(l): [] for l in direct_layers_rel}
            derf_direct_runs = {
                float(a): {int(l): [] for l in direct_layers_rel}
                for a in alphas
            }
        else:
            direct_layers_abs = direct_layers_abs or []
            direct_layers_rel = direct_layers_rel or []

        stats_block0 = collect_block0_input_stats(preln_model, samples, block0_input_override=block0_input_override)
        token_sqnorm_batches.append(stats_block0["token_sqnorm_over_d"])
        pairwise_batches.append(stats_block0["pairwise_dot_over_d"])

        Gb = compute_G_raw_across_layers(preln_model, samples, targets, block0_input_override=block0_input_override)
        pre_G_list.append(Gb)

        if apjn_layers:
            Jb = estimate_J_points_hutchinson(
                preln_model,
                samples,
                l0_list=apjn_layers,
                j_num_draws=apjn_cfg.j_num_draws,
                j_normalize_by=apjn_cfg.j_normalize_by,
                block0_input_override=block0_input_override,
            )
            preln_runs.append(Jb)

        if direct_layers_abs:
            direct_pre = estimate_direct_J_points_hutchinson(
                preln_model,
                samples,
                direct_layers=direct_layers_abs,
                direct_source_block=apjn_cfg.direct_source_block,
                j_num_draws=apjn_cfg.j_num_draws,
                j_normalize_by=apjn_cfg.j_normalize_by,
                block0_input_override=block0_input_override,
            )
            for l, val in direct_pre.items():
                pre_direct_runs[int(l)].append(float(val))

        for a in alphas:
            a_float = float(a)
            set_all_derf_alpha_(derf_model, a_float)
            G = compute_G_raw_across_layers(derf_model, samples, targets, block0_input_override=block0_input_override)
            derf_G_accum[a_float].append(G)

            if apjn_layers:
                inv = estimate_J_points_hutchinson(
                    derf_model,
                    samples,
                    l0_list=apjn_layers,
                    j_num_draws=apjn_cfg.j_num_draws,
                    j_normalize_by=apjn_cfg.j_normalize_by,
                    block0_input_override=block0_input_override,
                )
                derf_runs[a_float].append(inv)

            if direct_layers_abs:
                direct_vals = estimate_direct_J_points_hutchinson(
                    derf_model,
                    samples,
                    direct_layers=direct_layers_abs,
                    direct_source_block=apjn_cfg.direct_source_block,
                    j_num_draws=apjn_cfg.j_num_draws,
                    j_normalize_by=apjn_cfg.j_normalize_by,
                    block0_input_override=block0_input_override,
                )
                for l, val in direct_vals.items():
                    derf_direct_runs[a_float][int(l)].append(float(val))

            cuda_cleanup()

        del preln_model
        del derf_model
        cuda_cleanup()

    def _prune_direct_lists(layer_dict):
        out = {}
        for l, values in layer_dict.items():
            if not values:
                continue
            out[int(l)] = [float(v) for v in values]
        return out

    preln_pack = {
        "G_raw": pre_G_list,
        "J_raw": _average_scalar_dicts(preln_runs) if preln_runs else {},
        "direct_J_raw": _prune_direct_lists(pre_direct_runs),
    }

    input_stats = None
    if token_sqnorm_batches:
        input_stats = {
            "token_sqnorm_over_d": np.concatenate(token_sqnorm_batches),
            "pairwise_dot_over_d": np.concatenate(pairwise_batches) if pairwise_batches else np.empty(0, dtype=float),
        }

    derf_results = []
    for a in alphas:
        a_float = float(a)
        derf_results.append({
            "alpha": a_float,
            "G_raw": derf_G_accum[a_float],
            "J_raw": _average_scalar_dicts(derf_runs[a_float]) if derf_runs[a_float] else {},
            "direct_J_raw": _prune_direct_lists(derf_direct_runs[a_float]),
        })

    return {
        "model_cfg": cfg_to_dict(model_cfg),
        "apjn_cfg": cfg_to_dict(apjn_cfg),
        "batch_meta": batch_meta,
        "depth": int(depth) if depth is not None else 0,
        "seq_len": int(seq_len) if seq_len is not None else 0,
        "embed_dim": int(embed_dim) if embed_dim is not None else 0,
        "preln_with_J": preln_pack,
        "input_block0_stats": input_stats,
        "derf_pack_with_J": {
            "alphas": np.asarray(apjn_cfg.alphas, dtype=float).copy(),
            "results": derf_results,
        },
        "num_model_inits": int(num_inits),
        "direct_layers": [int(l) for l in direct_layers_rel] if direct_layers_rel else [],
        "direct_layers_absolute": [int(l) for l in direct_layers_abs] if direct_layers_abs else [],
        "direct_source_block": int(apjn_cfg.direct_source_block),
    }



def _extract_direct_panel_measurements(cifar_bundle):
    if not cifar_bundle:
        return {"preln": {}, "derf": {}, "layers": []}
    pre_direct_raw = ((cifar_bundle.get("preln_with_J") or {}).get("direct_J_raw") or {})
    derf_entries = (cifar_bundle.get("derf_pack_with_J") or {}).get("results", [])
    pre_means = {}
    layers = set()
    for l, values in pre_direct_raw.items():
        arr = np.atleast_1d(np.asarray(values, dtype=float)).reshape(-1)
        if arr.size == 0:
            continue
        pre_means[int(l)] = float(arr.mean())
        layers.add(int(l))
    derf_means = {}
    for entry in derf_entries:
        alpha = entry.get("alpha")
        if alpha is None:
            continue
        direct_raw = entry.get("direct_J_raw") or {}
        alpha_means = {}
        for l, values in direct_raw.items():
            arr = np.atleast_1d(np.asarray(values, dtype=float)).reshape(-1)
            if arr.size == 0:
                continue
            alpha_means[int(l)] = float(arr.mean())
            layers.add(int(l))
        if alpha_means:
            derf_means[float(alpha)] = alpha_means
    return {
        "preln": pre_means,
        "derf": derf_means,
        "layers": sorted(layers),
    }


def fit_panel_d_direct_initial_conditions(
    cifar_panel_bundle,
    fit_cfg: APJNFitConfig,
    mean_field_cfg: MeanFieldConfig,
    panel_d_cfg: PanelDConfig,
    alphas,
):
    metric = str(fit_cfg.metric).lower()
    direct_data = _extract_direct_panel_measurements(cifar_panel_bundle)
    if not direct_data["layers"]:
        raise ValueError("No direct APJN measurements found; set APJN_CFG.direct_layers and rerun the CIFAR APJN cell.")

    q0_values = _resolve_grid_values(fit_cfg.q0_values, fit_cfg.q0_num)
    p0_values = _resolve_grid_values(fit_cfg.p0_values, fit_cfg.p0_num)
    c_values = _resolve_grid_values(getattr(fit_cfg, "preln_scale_values", None), getattr(fit_cfg, "preln_scale_num", 41))
    refine_radius = float(getattr(fit_cfg, "refine_radius", 0.2))
    rescale_vit_preln_apjn = bool(getattr(fit_cfg, "rescale_vit_preln_apjn", False))

    max_layer = max(int(l) for l in direct_data["layers"])
    fit_layers = max(1, max_layer)

    if cifar_panel_bundle and int(cifar_panel_bundle.get("seq_len", 0)) > 0:
        n_tokens = int(cifar_panel_bundle["seq_len"]) - 1
    else:
        n_tokens = int(panel_d_cfg.n_tokens)

    alpha_arr = np.asarray(alphas, dtype=float)

    def _metric_ignore_nan(vit_vals, th_vals):
        vit_arr = np.asarray(vit_vals, dtype=float)
        th_arr = np.asarray(th_vals, dtype=float)
        mask = np.isfinite(vit_arr) & np.isfinite(th_arr)
        if not np.any(mask):
            return None
        return _fit_metric_value(vit_arr[mask], th_arr[mask], metric)

    def _scaled_preln_curve(theory_arr, c_scale, layer_ids):
        theory_arr = np.asarray(theory_arr, dtype=float)
        out = []
        for l in layer_ids:
            val = float(theory_arr[int(l)])
            if int(l) >= 1:
                val *= float(c_scale)
            out.append(val)
        return out

    def _fit_qp_for_subset(*, preln_subset, derf_subset, c_scale=1.0, q_grid=None, p_grid=None):
        q_grid = q0_values if q_grid is None else np.asarray(q_grid, dtype=float)
        p_grid = p0_values if p_grid is None else np.asarray(p_grid, dtype=float)
        best = None
        for q0 in q_grid:
            for p0 in p_grid:
                if p0 > q0 + 1e-12:
                    continue

                theory_bundle = compute_theory_qp_bundle(
                    num_layers=fit_layers,
                    alphas=alpha_arr,
                    n_tokens=n_tokens,
                    q0=float(q0),
                    p0=float(p0),
                    mean_field_cfg=mean_field_cfg,
                )

                metrics = []

                if preln_subset:
                    layer_ids = sorted(int(l) for l in preln_subset.keys())
                    vit_vals = [preln_subset[l] for l in layer_ids]
                    th_vals = _scaled_preln_curve(theory_bundle["preln"]["J_direct"], c_scale, layer_ids)
                    val = _metric_ignore_nan(vit_vals, th_vals)
                    if val is not None:
                        metrics.append(val)

                for a in alpha_arr:
                    a_float = float(a)
                    if a_float not in derf_subset:
                        continue
                    layer_ids = sorted(int(l) for l in derf_subset[a_float].keys())
                    vit_vals = [derf_subset[a_float][l] for l in layer_ids]
                    th_vals = [theory_bundle["derf"][a_float]["J_direct"][l] for l in layer_ids]
                    val = _metric_ignore_nan(vit_vals, th_vals)
                    if val is not None:
                        metrics.append(val)

                if not metrics:
                    continue

                value = float(np.mean(metrics))
                if best is None or value < best["value"]:
                    best = {
                        "q0": float(q0),
                        "p0": float(p0),
                        "value": value,
                        "metric": metric,
                    }

        if best is None:
            raise RuntimeError("Unable to fit panel (d) direct measurements; check the grid or data availability.")
        return best

    derf_fit = _fit_qp_for_subset(preln_subset={}, derf_subset=direct_data["derf"], c_scale=1.0)

    derf_theory_bundle = compute_theory_qp_bundle(
        num_layers=fit_layers,
        alphas=alpha_arr,
        n_tokens=n_tokens,
        q0=float(derf_fit["q0"]),
        p0=float(derf_fit["p0"]),
        mean_field_cfg=mean_field_cfg,
    )

    if rescale_vit_preln_apjn:
        if 1 not in direct_data["preln"]:
            raise RuntimeError("rescale_vit_preln_apjn=True requires a pre-LN direct APJN measurement at layer 1.")
        vit_anchor = float(direct_data["preln"][1])
        theory_anchor = float(derf_theory_bundle["preln"]["J_direct"][1])
        if not np.isfinite(vit_anchor) or abs(vit_anchor) <= 1e-12:
            raise RuntimeError("Cannot rescale pre-LN direct APJN: ViT pre-LN value at layer 1 is invalid or zero.")
        if not np.isfinite(theory_anchor):
            raise RuntimeError("Cannot rescale pre-LN direct APJN: theory pre-LN value at layer 1 is invalid.")

        preln_rescaled = {
            int(l): float(v) / vit_anchor * theory_anchor
            for l, v in direct_data["preln"].items()
        }
        preln_rescaled_for_fit = {
            int(l): float(v)
            for l, v in preln_rescaled.items()
            if int(l) != 1
        }

        q_ref = float(derf_fit["q0"])
        p_ref = float(derf_fit["p0"])
        q_local = q0_values[(q0_values >= q_ref - refine_radius - 1e-12) & (q0_values <= q_ref + refine_radius + 1e-12)]
        p_local = p0_values[(p0_values >= p_ref - refine_radius - 1e-12) & (p0_values <= p_ref + refine_radius + 1e-12)]
        if q_local.size == 0:
            q_local = np.asarray([q_ref], dtype=float)
        if p_local.size == 0:
            p_local = np.asarray([p_ref], dtype=float)

        final_fit = _fit_qp_for_subset(
            preln_subset=preln_rescaled_for_fit,
            derf_subset=direct_data["derf"],
            c_scale=1.0,
            q_grid=q_local,
            p_grid=p_local,
        )

        return {
            "q0": final_fit["q0"],
            "p0": final_fit["p0"],
            "value": final_fit["value"],
            "metric": metric,
            "rescale_vit_preln_apjn": True,
            "preln_scale_C": None,
            "preln_layer1_vit": vit_anchor,
            "preln_layer1_theory": theory_anchor,
            "derf_only_fit": derf_fit,
            "preln_rescaled_data": preln_rescaled,
            "preln_rescaled_fit_data": preln_rescaled_for_fit,
            "final_fit": final_fit,
        }

    best_c = None
    if direct_data["preln"]:
        layer_ids = sorted(int(l) for l in direct_data["preln"].keys())
        vit_vals = [direct_data["preln"][l] for l in layer_ids]
        for c_scale in c_values:
            th_vals = _scaled_preln_curve(derf_theory_bundle["preln"]["J_direct"], c_scale, layer_ids)
            val = _metric_ignore_nan(vit_vals, th_vals)
            if val is None:
                continue
            if best_c is None or val < best_c["value"]:
                best_c = {
                    "C": float(c_scale),
                    "value": float(val),
                    "metric": metric,
                }
    if best_c is None:
        raise RuntimeError("Unable to fit the pre-LN multiplicative factor C; check the C-grid or data availability.")

    q_ref = float(derf_fit["q0"])
    p_ref = float(derf_fit["p0"])
    q_local = q0_values[(q0_values >= q_ref - refine_radius - 1e-12) & (q0_values <= q_ref + refine_radius + 1e-12)]
    p_local = p0_values[(p0_values >= p_ref - refine_radius - 1e-12) & (p0_values <= p_ref + refine_radius + 1e-12)]
    if q_local.size == 0:
        q_local = np.asarray([q_ref], dtype=float)
    if p_local.size == 0:
        p_local = np.asarray([p_ref], dtype=float)

    final_fit = _fit_qp_for_subset(
        preln_subset=direct_data["preln"],
        derf_subset=direct_data["derf"],
        c_scale=best_c["C"],
        q_grid=q_local,
        p_grid=p_local,
    )

    return {
        "q0": final_fit["q0"],
        "p0": final_fit["p0"],
        "value": final_fit["value"],
        "metric": metric,
        "rescale_vit_preln_apjn": False,
        "preln_scale_C": float(best_c["C"]),
        "derf_only_fit": derf_fit,
        "preln_scale_fit": best_c,
        "final_fit": final_fit,
    }

def _shift_apjn_curve_for_plot(y: np.ndarray):
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or y.size < 2:
        raise ValueError("APJN curve must have length at least 2.")
    x = np.arange(y.size - 1, dtype=int)
    return x, y[1:]

def _shift_apjn_points_for_plot(j_raw: dict):
    l0s = np.array(sorted(int(l) for l in j_raw.keys()), dtype=int)
    keep = l0s > 0
    x = l0s[keep] - 1
    y = np.array([float(j_raw[int(l)]) for l in l0s[keep]], dtype=float)
    return x, y

def _panel_c_apjn_arrays(vit_bundle, theory_bundle):
    theory_J = {
        "preln": theory_bundle["preln"],
        "derf": {float(a): theory_bundle["derf"][float(a)] for a in theory_bundle["derf"].keys()},
    }

    apjn_layers = sorted(int(l) for l in vit_bundle["preln_with_J"]["J_raw"].keys())
    keep_layers = [l for l in apjn_layers if l > 0]
    x_shift = np.asarray([l - 1 for l in keep_layers], dtype=int)

    preln_vit = np.asarray([vit_bundle["preln_with_J"]["J_raw"][l] for l in keep_layers], dtype=float)
    preln_th = np.asarray([theory_J["preln"]["J"][l] for l in keep_layers], dtype=float)

    derf_vit = {}
    derf_th = {}
    for r in vit_bundle["derf_pack_with_J"]["results"]:
        a = float(r["alpha"])
        derf_vit[a] = np.asarray([r["J_raw"][l] for l in keep_layers], dtype=float)
        derf_th[a] = np.asarray([theory_J["derf"][a]["J"][l] for l in keep_layers], dtype=float)

    return {
        "x_shift": x_shift,
        "preln_vit": preln_vit,
        "preln_th": preln_th,
        "derf_vit": derf_vit,
        "derf_th": derf_th,
    }

def _fit_metric_value(y_true, y_pred, metric):
    if metric == "mse":
        return float(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2))
    if metric == "mape":
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return float(np.mean(np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), 1e-12)))
    raise ValueError("metric must be 'mse' or 'mape'")

def _resolve_grid_values(values, num):
    if values is None:
        return np.linspace(0.0, 1.0, int(num), dtype=float)
    arr = np.unique(np.asarray(values, dtype=float))
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("Grid values must be a non-empty 1D array-like.")
    return arr

def fit_theory_for_apjn(
    cifar_apjn_bundle,
    fit_cfg: APJNFitConfig,
    mean_field_cfg: MeanFieldConfig,
):
    metric = str(fit_cfg.metric).lower()
    q0_values = _resolve_grid_values(fit_cfg.q0_values, fit_cfg.q0_num)
    p0_values = _resolve_grid_values(fit_cfg.p0_values, fit_cfg.p0_num)

    alphas = np.asarray(cifar_apjn_bundle["derf_pack_with_J"]["alphas"], dtype=float)
    depth = int(cifar_apjn_bundle["depth"])
    n_tokens = int(cifar_apjn_bundle["seq_len"] - 1)

    best = None
    metric_matrix = np.full((len(p0_values), len(q0_values)), np.nan, dtype=float)

    for iq, q0 in enumerate(q0_values):
        for ip, p0 in enumerate(p0_values):
            if p0 > q0 + 1e-12:
                continue

            theory_bundle = compute_theory_qp_bundle(
                num_layers=depth,
                alphas=alphas,
                n_tokens=n_tokens,
                q0=float(q0),
                p0=float(p0),
                mean_field_cfg=mean_field_cfg,
            )
            arrs = _panel_c_apjn_arrays(cifar_apjn_bundle, theory_bundle)

            vals = [_fit_metric_value(arrs["preln_vit"], arrs["preln_th"], metric)]
            for a in alphas:
                vals.append(_fit_metric_value(arrs["derf_vit"][float(a)], arrs["derf_th"][float(a)], metric))
            value = float(np.mean(vals))
            metric_matrix[ip, iq] = value

            if best is None or value < best["value"]:
                best = {
                    "value": value,
                    "q0": float(q0),
                    "p0": float(p0),
                    "theory_bundle": theory_bundle,
                    "panel_c_arrays": arrs,
                }

    if best is None:
        raise RuntimeError("No valid (q0, p0) points in fit grid.")

    return {
        "metric": metric,
        "q0_values": q0_values,
        "p0_values": p0_values,
        "metric_matrix": metric_matrix,
        "q0": best["q0"],
        "p0": best["p0"],
        "value": best["value"],
        "theory_bundle": best["theory_bundle"],
        "panel_c_arrays": best["panel_c_arrays"],
    }

# -------------------- plotting helpers --------------------

def _make_alpha_colors(alphas):
    alphas = np.asarray(alphas, dtype=float)
    base_cmap = plt.get_cmap(BASE_CMAP_NAME)
    colors = [base_cmap(shade_from_index(i, len(alphas))) for i in range(len(alphas))]
    return colors

def plot_perm_qpq_grid(perm_empirical_bundle, perm_theory_bundle, style_cfg: FinalThreePanelStyleConfig):
    alphas = np.asarray(perm_empirical_bundle["perm_cfg"]["alphas"], dtype=float)
    colors = _make_alpha_colors(alphas)

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.0), sharex=True)
    ax_q_th, ax_q_emp = axes[0, 0], axes[0, 1]
    ax_r_th, ax_r_emp = axes[1, 0], axes[1, 1]

    l = perm_theory_bundle["l"]

    ax_q_th.plot(l, perm_theory_bundle["preln"]["q"], color="black", lw=style_cfg.line_width)
    for i, a in enumerate(alphas):
        ax_q_th.plot(l, perm_theory_bundle["derf"][float(a)]["q"], color=colors[i], lw=style_cfg.line_width)
    ax_q_th.set_title("q vs. l (theory)")
    ax_q_th.set_ylabel(r"$q$")
    prettify_axes(ax_q_th)

    ax_q_emp.plot(
        perm_empirical_bundle["preln"]["l"], perm_empirical_bundle["preln"]["q"],
        color="black", lw=style_cfg.line_width, marker="o", ms=style_cfg.perm_marker_size
    )
    for i, a in enumerate(alphas):
        ax_q_emp.plot(
            perm_empirical_bundle["derf"][float(a)]["l"], perm_empirical_bundle["derf"][float(a)]["q"],
            color=colors[i], lw=style_cfg.line_width, marker="o", ms=style_cfg.perm_marker_size
        )
    ax_q_emp.set_title("q vs. l (ViT)")
    prettify_axes(ax_q_emp)

    ax_r_th.plot(l, perm_theory_bundle["preln"]["p_over_q"], color="black", lw=style_cfg.line_width)
    for i, a in enumerate(alphas):
        ax_r_th.plot(l, perm_theory_bundle["derf"][float(a)]["p_over_q"], color=colors[i], lw=style_cfg.line_width)
    ax_r_th.set_title("p/q vs. l (theory)")
    ax_r_th.set_xlabel(r"$b$")
    ax_r_th.set_ylabel(r"$p/q$")
    prettify_axes(ax_r_th)

    ax_r_emp.plot(
        perm_empirical_bundle["preln"]["l"], perm_empirical_bundle["preln"]["p_over_q"],
        color="black", lw=style_cfg.line_width, marker="o", ms=style_cfg.perm_marker_size
    )
    for i, a in enumerate(alphas):
        ax_r_emp.plot(
            perm_empirical_bundle["derf"][float(a)]["l"], perm_empirical_bundle["derf"][float(a)]["p_over_q"],
            color=colors[i], lw=style_cfg.line_width, marker="o", ms=style_cfg.perm_marker_size
        )
    ax_r_emp.set_title("p/q vs. l (ViT)")
    ax_r_emp.set_xlabel(r"$b$")
    prettify_axes(ax_r_emp)

    fig.tight_layout()
    save_path = style_cfg.save_path
    if save_path is not None:
        fig.savefig(save_path)
        print("Saved:", save_path)

    plt.show()

def plot_apjn_fit(cifar_apjn_bundle, apjn_fit_bundle, style_cfg: FinalThreePanelStyleConfig):
    alphas = np.asarray(cifar_apjn_bundle["derf_pack_with_J"]["alphas"], dtype=float)
    colors = _make_alpha_colors(alphas)
    panel_c = apjn_fit_bundle["panel_c_arrays"]

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8))
    ax.plot(panel_c["x_shift"], panel_c["preln_th"], color="black", lw=style_cfg.line_width, ls="--")
    ax.plot(
        panel_c["x_shift"], panel_c["preln_vit"],
        color="black", ls="none", marker="o", ms=style_cfg.vit_marker_size
    )

    for i, a in enumerate(alphas):
        ax.plot(panel_c["x_shift"], panel_c["derf_th"][float(a)], color=colors[i], lw=style_cfg.line_width, ls="--")
        ax.plot(
            panel_c["x_shift"], panel_c["derf_vit"][float(a)],
            color=colors[i], ls="none", marker="o", ms=style_cfg.vit_marker_size
        )

    ax.set_title("APJN: theory fit vs. ViT")
    ax.set_xlabel(r"$b$")
    ax.set_ylabel(r"$\mathcal{J}^{\, b, 0}$")
    ax.set_yscale("log")
    prettify_log_axis(ax, "y")
    prettify_axes(ax)

    ax.text(
        0.02, 0.03,
        f"{apjn_fit_bundle['metric'].upper()} fit: q0={apjn_fit_bundle['q0']:.4g}, p0={apjn_fit_bundle['p0']:.4g}",
        transform=ax.transAxes,
        fontsize=style_cfg.tick_fs,
        va="bottom",
    )

    handles = [
        Line2D([0], [0], color="black", lw=style_cfg.line_width, ls="--", label="Theory"),
        Line2D([0], [0], color="black", marker="o", ls="none", ms=style_cfg.vit_marker_size, label="ViT"),
    ]
    ax.legend(handles=handles, frameon=False, loc="best")

    plt.show()

def plot_panel_d(panel_d_bundle, alphas, style_cfg: FinalThreePanelStyleConfig):
    alphas = np.asarray(alphas, dtype=float)
    colors = _make_alpha_colors(alphas)
    empirical = panel_d_bundle.get("empirical") or {}
    empirical_derf = empirical.get("derf") or {}
    empirical_preln = empirical.get("preln")
    direct_source_block = empirical.get("direct_source_block") if empirical else None
    asym_ls = get_asym_linestyle(
        mode=style_cfg.asym_style_mode,
        dash_len=style_cfg.asym_dash_len,
        gap_len=style_cfg.asym_gap_len,
        dot_len=style_cfg.asym_dot_len,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.3))

    for i, d in enumerate(panel_d_bundle["derf"]):
        ax1.plot(panel_d_bundle["l_arr"], d["logJ"]**2, color=colors[i], lw=style_cfg.line_width)
        ax1.plot(
            panel_d_bundle["l_arr"], d["logJ_asym"]**2,
            color="0.35", lw=style_cfg.asym_line_width, ls=asym_ls, dash_capstyle="round", zorder=10
        )
    ax1.set_title(r"panel (d), DERF: $(\log \mathcal{J}^{\, b, 0})^2$")
    ax1.set_xlabel(r"$b$")
    ax1.set_ylabel(r"$(\log \mathcal{J}^{\, b, 0})^2$")
    prettify_axes(ax1)

    ax2.plot(
        panel_d_bundle["l_arr"][1:], panel_d_bundle["preln"]["J"][1:],
        color="black", lw=style_cfg.line_width
    )
    ax2.plot(
        panel_d_bundle["l_arr"][1:], panel_d_bundle["preln"]["J_asym"][1:],
        color="0.35", lw=style_cfg.asym_line_width, ls=asym_ls, dash_capstyle="round", zorder=10
    )
    ax2.set_title(r"panel (d), pre-LN: $\mathcal{J}^{\, b, 0}$")
    ax2.set_xlabel(r"$b$")
    ax2.set_ylabel(r"$\mathcal{J}^{\, b, 0}$")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    prettify_log_axis(ax2, "x")
    prettify_log_axis(ax2, "y")
    prettify_axes(ax2)

    def _scatter(ax, data, use_log_sq, color):
        ax.scatter(
            data["layers"],
            data["logJ_sq"] if use_log_sq else data["J"],
            marker="o",
            facecolors="none",
            edgecolors=color,
            linewidths=style_cfg.line_width * 0.9,
            s=(style_cfg.vit_marker_size * 4.5) ** 2,
            zorder=16,
        )

    legend_label = "ViT direct" if direct_source_block is None else f"ViT direct (block {direct_source_block})"
    legend_handle = Line2D(
        [0], [0], marker="o", ls="none", markerfacecolor="none", markeredgecolor="0.2", mew=style_cfg.line_width * 0.9, label=legend_label
    )

    if empirical_derf:
        plotted = False
        for i, a in enumerate(alphas):
            dire = empirical_derf.get(float(a))
            if dire is None:
                continue
            _scatter(ax1, dire, True, colors[i])
            plotted = True
        if plotted:
            ax1.legend([legend_handle], [legend_label], frameon=False, loc="best", fontsize=style_cfg.annotation_fs)

    if empirical_preln:
        _scatter(ax2, empirical_preln, False, "black")
        ax2.legend([legend_handle], [legend_label], frameon=False, loc="best", fontsize=style_cfg.annotation_fs)

    plt.tight_layout()
    plt.show()

def plot_asymptotic_heatmap_figure(panel_d_bundle, alphas, heatmap_bundle, style_cfg: HeatmapFigureStyleConfig):
    alphas = np.asarray(alphas, dtype=float)
    colors = _make_alpha_colors(alphas)
    empirical = panel_d_bundle.get("empirical") or {}
    empirical_derf = empirical.get("derf") or {}
    empirical_preln = empirical.get("preln")
    num_model_inits = int(empirical.get("num_model_inits", 1)) if empirical else 1
    show_means = bool(style_cfg.average_direct_markers_across_inits) and num_model_inits > 1
    asym_ls = get_asym_linestyle(
        mode=style_cfg.asym_style_mode,
        dash_len=style_cfg.asym_dash_len,
        gap_len=style_cfg.asym_gap_len,
        dot_len=style_cfg.asym_dot_len,
    )

    fig = plt.figure(figsize=style_cfg.figsize)
    gs = fig.add_gridspec(
        nrows=3,
        ncols=5,
        height_ratios=[1.0, style_cfg.panel_cd_hspace, 1.0],
        width_ratios=[2.8, style_cfg.panel_ab_wspace, 2.8, style_cfg.panel_b_heatmap_wspace, 1.0],
        hspace=0.0,
        wspace=0.0,
    )
    ax_derf = fig.add_subplot(gs[:, 0])
    ax_pre = fig.add_subplot(gs[:, 2])
    ax_lam = fig.add_subplot(gs[0, 4])
    ax_zeta = fig.add_subplot(gs[2, 4])

    for i, d in enumerate(panel_d_bundle["derf"]):
        ax_derf.plot(panel_d_bundle["l_arr"], d["logJ"]**2, color=colors[i], lw=style_cfg.line_width)
        ax_derf.plot(
            panel_d_bundle["l_arr"], d["logJ_asym"]**2,
            color="0.35", lw=style_cfg.asym_line_width, ls=asym_ls, dash_capstyle="round", zorder=30
        )
    ax_derf.set_xlabel(r"$b$", fontsize=style_cfg.label_fs)
    ax_derf.set_ylabel(r"$(\log \mathcal{J}^{\,b,0})^2$", fontsize=style_cfg.label_fs)
    ax_derf.set_xlim(0, panel_d_bundle["L"])
    ax_derf.tick_params(labelsize=style_cfg.tick_fs)
    ax_derf.set_box_aspect(1)
    prettify_axes(ax_derf)

    ax_pre.plot(
        panel_d_bundle["l_arr"][1:], panel_d_bundle["preln"]["J"][1:],
        color="black", lw=style_cfg.line_width
    )
    ax_pre.plot(
        panel_d_bundle["l_arr"][1:], panel_d_bundle["preln"]["J_asym"][1:],
        color="0.35", lw=style_cfg.asym_line_width, ls=asym_ls, dash_capstyle="round", zorder=30
    )
    ax_pre.set_xlabel(r"$b$", fontsize=style_cfg.label_fs)
    ax_pre.set_ylabel(r"$\mathcal{J}^{\,b,0}$", fontsize=style_cfg.label_fs)
    ax_pre.set_xscale("log")
    ax_pre.set_yscale("log")
    prettify_log_axis(ax_pre, "x")
    prettify_log_axis(ax_pre, "y")
    ax_pre.set_xlim(1, panel_d_bundle["L"])
    ax_pre.tick_params(labelsize=style_cfg.tick_fs)
    ax_pre.set_box_aspect(1)
    prettify_axes(ax_pre)

    def _series_xy(data, kind):
        if show_means:
            x = data["layers_mean"]
            y = data["logJ_sq_mean"] if kind == "logJ_sq" else data["J_mean"]
        else:
            x = data["layers"]
            y = data["logJ_sq"] if kind == "logJ_sq" else data["J"]
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    def _scatter_derf(ax):
        if not empirical_derf:
            return
        for i, a in enumerate(alphas):
            dire = empirical_derf.get(float(a))
            if dire is None:
                continue
            x_vals, y_vals = _series_xy(dire, "logJ_sq")
            ax.scatter(
                x_vals,
                y_vals,
                marker="o",
                facecolors=colors[i],
                edgecolors=colors[i],
                linewidths=style_cfg.line_width * 0.9,
                s=style_cfg.direct_marker_area,
                zorder=16,
            )

    def _scatter_pre(ax):
        if not empirical_preln:
            return
        x_vals, y_vals = _series_xy(empirical_preln, "J")
        ax.scatter(
            x_vals,
            y_vals,
            marker="o",
            facecolors="black",
            edgecolors="black",
            linewidths=style_cfg.line_width * 0.9,
            s=style_cfg.direct_marker_area,
            zorder=16,
        )

    _scatter_derf(ax_derf)
    _scatter_pre(ax_pre)

    legend_handles = [
        Line2D([0], [0], marker="o", ls="none", markerfacecolor="black", markeredgecolor="black", markersize=np.sqrt(style_cfg.direct_marker_area), label="ViT"),
        Line2D([0], [0], color="black", lw=style_cfg.line_width, label="theory"),
        Line2D([0], [0], color="0.35", lw=style_cfg.asym_line_width, ls=asym_ls, label="asymptote"),
    ]
    ax_derf.legend(handles=legend_handles, loc="upper left", frameon=False, fontsize=style_cfg.annotation_fs)

    sigma21_vals = heatmap_bundle["sigma21_vals"]
    sigmaOV_vals = heatmap_bundle["sigmaOV_vals"]
    extent = [
        float(sigma21_vals[0]),
        float(sigma21_vals[-1]),
        float(sigmaOV_vals[0]),
        float(sigmaOV_vals[-1]),
    ]

    lam_data = np.asarray(heatmap_bundle["lam_inv"], dtype=float)
    im_lam = ax_lam.imshow(
        lam_data,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap=WHITE_SOFT_RED_CMAP,
        vmin=0.0,
        vmax=max(float(np.nanmax(lam_data)), 1e-6),
    )
    ax_lam.set_xlabel(r"$\sigma_{21}$", fontsize=style_cfg.label_fs)
    ax_lam.set_ylabel(r"$\sigma_{OV}$", fontsize=style_cfg.label_fs)
    ax_lam.tick_params(labelsize=style_cfg.tick_fs)
    ax_lam.set_box_aspect(1)
    cbar_lam = fig.colorbar(im_lam, ax=ax_lam, fraction=0.046, pad=0.04)
    cbar_lam.ax.tick_params(labelsize=style_cfg.tick_fs)
    lam_title = cbar_lam.ax.set_title(r"$\lambda^{-1}$", fontsize=style_cfg.label_fs, pad=6.0)
    lam_title.set_position((style_cfg.lambda_title_shift, lam_title.get_position()[1]))

    zeta_data = np.asarray(heatmap_bundle["zeta"], dtype=float)
    im_zeta = ax_zeta.imshow(
        zeta_data,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap=ZETA_GOLD_CMAP,
        vmin=0.0,
        vmax=1.0,
    )
    ax_zeta.set_xlabel(r"$\sigma_{21}$", fontsize=style_cfg.label_fs)
    ax_zeta.set_ylabel(r"$\sigma_{OV}$", fontsize=style_cfg.label_fs)
    ax_zeta.tick_params(labelsize=style_cfg.tick_fs)
    ax_zeta.set_box_aspect(1)
    cbar_zeta = fig.colorbar(im_zeta, ax=ax_zeta, fraction=0.046, pad=0.04)
    cbar_zeta.ax.tick_params(labelsize=style_cfg.tick_fs)
    zeta_title = cbar_zeta.ax.set_title(r"$\zeta$", fontsize=style_cfg.label_fs, pad=6.0)
    zeta_title.set_position((style_cfg.zeta_title_shift, zeta_title.get_position()[1]))

    fig.canvas.draw()

    alpha_ticks = style_cfg.alpha_colorbar_tick_values
    if alpha_ticks is not None:
        min_alpha = float(np.min(alphas))
        max_alpha = float(np.max(alphas))
        alpha_ticks = [float(v) for v in alpha_ticks if min_alpha - 1e-9 <= v <= max_alpha + 1e-9]
        if not alpha_ticks:
            alpha_ticks = None

    cax_alpha = add_alpha_colorbar_vertical_single(
        ax_derf,
        alphas,
        colors,
        label=r"$\alpha$",
        cb_fs=style_cfg.alpha_legend_fs,
        pad=style_cfg.alpha_cbar_pad,
        width=style_cfg.alpha_cbar_width,
        tick_values=alpha_ticks,
    )
    cax_alpha.yaxis.tick_right()
    cax_alpha.tick_params(labelright=True, labelleft=False, pad=2)

    fig.canvas.draw()

    def _panel_title(ax, text):
        pos = ax.get_position()
        x = pos.x0 + pos.width / 2
        y = pos.y1 + style_cfg.panel_title_offset
        fig.text(x, y, text, ha="center", va="bottom", fontsize=style_cfg.title_fs)

    _panel_title(ax_derf, r"(a) $\mathcal{J}^{\,b,0}$ (Derf)")
    _panel_title(ax_pre, r"(b) $\mathcal{J}^{\,b,0}$ (pre-LN)")
    _panel_title(ax_lam, r"(c) $\lambda^{-1}$ (Derf)")
    _panel_title(ax_zeta, r"(d) $\zeta$ (pre-LN)")

    if style_cfg.save_path is not None:
        fig.savefig(style_cfg.save_path)
        print("Saved:", style_cfg.save_path)

    plt.show()

def plot_final_three_panel(
    perm_empirical_bundle,
    perm_theory_bundle,
    cifar_apjn_bundle,
    apjn_fit_bundle,
    style_cfg: FinalThreePanelStyleConfig,
):
    alphas = np.asarray(perm_empirical_bundle["perm_cfg"]["alphas"], dtype=float)
    colors = _make_alpha_colors(alphas)

    fig = plt.figure(figsize=style_cfg.figsize)
    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        height_ratios=[1.0, 1.0, 0.2],
        width_ratios=[0.9, 1.0],
        hspace=0.0,
        wspace=style_cfg.panel_wspace,
    )

    ax_a = fig.add_subplot(gs[0:2, 0])
    gs_bc = gs[0:2, 1].subgridspec(
        nrows=2,
        ncols=2,
        wspace=style_cfg.theory_vit_wspace,
        hspace=style_cfg.panel_row_hspace,
    )
    ax_b_th = fig.add_subplot(gs_bc[0, 0])
    ax_b_emp = fig.add_subplot(gs_bc[0, 1])
    ax_c_th = fig.add_subplot(gs_bc[1, 0])
    ax_c_emp = fig.add_subplot(gs_bc[1, 1])
    cax = fig.add_subplot(gs[2, :])

    panel_c = apjn_fit_bundle["panel_c_arrays"]
    theory_bundle = apjn_fit_bundle.get("theory_bundle")
    ax_a.set_title(r"(a) $\mathcal{J}^{\, B, b}$", fontsize=style_cfg.title_fs)

    if theory_bundle is not None:
        full_x = np.asarray(theory_bundle["l"], dtype=float)[1:] - 1.0
        ax_a.plot(full_x, np.asarray(theory_bundle["preln"]["J"], dtype=float)[1:], color="black", lw=style_cfg.line_width, ls="--")
    else:
        full_x = np.asarray(panel_c["x_shift"], dtype=float)
        ax_a.plot(full_x, panel_c["preln_th"], color="black", lw=style_cfg.line_width, ls="--")

    ax_a.plot(
        panel_c["x_shift"], panel_c["preln_vit"],
        color="black", ls="none", marker="o", ms=style_cfg.vit_marker_size
    )
    for i, a in enumerate(alphas):
        if theory_bundle is not None:
            ax_a.plot(full_x, np.asarray(theory_bundle["derf"][float(a)]["J"], dtype=float)[1:], color=colors[i], lw=style_cfg.line_width, ls="--")
        else:
            ax_a.plot(panel_c["x_shift"], panel_c["derf_th"][float(a)], color=colors[i], lw=style_cfg.line_width, ls="--")
        ax_a.plot(
            panel_c["x_shift"], panel_c["derf_vit"][float(a)],
            color=colors[i], ls="none", marker="o", ms=style_cfg.vit_marker_size
        )

    ax_a.set_xlabel(r"$b$", fontsize=style_cfg.label_fs)
    ax_a.set_ylabel(r"$\mathcal{J}^{\, B, b}$", fontsize=style_cfg.label_fs)
    ax_a.set_yscale("log")
    prettify_log_axis(ax_a, "y")
    ax_a.tick_params(labelsize=style_cfg.tick_fs)
    ax_a.set_xlim(float(full_x[0]), float(full_x[-1]))
    prettify_axes(ax_a)
    ax_a.legend(
        handles=[
            Line2D([0], [0], color="black", lw=style_cfg.line_width, ls="--", label="theory"),
            Line2D([0], [0], color="black", marker="o", ls="none", ms=style_cfg.vit_marker_size, label="ViT"),
        ],
        frameon=False,
        loc="best",
        fontsize=style_cfg.annotation_fs,
    )

    max_layer = float(np.max(perm_theory_bundle["l"]))

    def _annotate_panel(ax, text):
        ax.text(style_cfg.panel_label_xoffset, 0.95, text, transform=ax.transAxes, ha="left", va="top", fontsize=style_cfg.annotation_fs)

    # Panel (b): q^l
    ax_b_th.plot(perm_theory_bundle["l"], perm_theory_bundle["preln"]["q"], color="black", lw=style_cfg.line_width)
    for i, a in enumerate(alphas):
        ax_b_th.plot(perm_theory_bundle["l"], perm_theory_bundle["derf"][float(a)]["q"], color=colors[i], lw=style_cfg.line_width)
    ax_b_th.set_ylabel(r"$q$", fontsize=style_cfg.label_fs)
    ax_b_th.set_xlim(0, max_layer)
    ax_b_th.tick_params(labelsize=style_cfg.tick_fs, labelbottom=False)
    prettify_axes(ax_b_th)
    _annotate_panel(ax_b_th, "theory")

    ax_b_emp.plot(
        perm_empirical_bundle["preln"]["l"], perm_empirical_bundle["preln"]["q"],
        color="black", lw=style_cfg.line_width, marker="o", ms=style_cfg.perm_marker_size
    )
    for i, a in enumerate(alphas):
        ax_b_emp.plot(
            perm_empirical_bundle["derf"][float(a)]["l"], perm_empirical_bundle["derf"][float(a)]["q"],
            color=colors[i], lw=style_cfg.line_width, marker="o", ms=style_cfg.perm_marker_size
        )
    ax_b_emp.set_xlim(0, max_layer)
    ax_b_emp.tick_params(labelsize=style_cfg.tick_fs, labelbottom=False, labelleft=False)
    ax_b_emp.set_ylabel("")
    prettify_axes(ax_b_emp)
    _annotate_panel(ax_b_emp, "ViT")

    # Panel (c): p^l / q^l
    ax_c_th.plot(perm_theory_bundle["l"], perm_theory_bundle["preln"]["p_over_q"], color="black", lw=style_cfg.line_width)
    for i, a in enumerate(alphas):
        ax_c_th.plot(perm_theory_bundle["l"], perm_theory_bundle["derf"][float(a)]["p_over_q"], color=colors[i], lw=style_cfg.line_width)
    ax_c_th.set_xlabel(r"$b$", fontsize=style_cfg.label_fs)
    ax_c_th.set_ylabel(r"$p/q$", fontsize=style_cfg.label_fs)
    ax_c_th.set_xlim(0, max_layer)
    ax_c_th.tick_params(labelsize=style_cfg.tick_fs)
    prettify_axes(ax_c_th)
    _annotate_panel(ax_c_th, "theory")

    ax_c_emp.plot(
        perm_empirical_bundle["preln"]["l"], perm_empirical_bundle["preln"]["p_over_q"],
        color="black", lw=style_cfg.line_width, marker="o", ms=style_cfg.perm_marker_size
    )
    for i, a in enumerate(alphas):
        ax_c_emp.plot(
            perm_empirical_bundle["derf"][float(a)]["l"], perm_empirical_bundle["derf"][float(a)]["p_over_q"],
            color=colors[i], lw=style_cfg.line_width, marker="o", ms=style_cfg.perm_marker_size
        )
    ax_c_emp.set_xlabel(r"$b$", fontsize=style_cfg.label_fs)
    ax_c_emp.set_xlim(0, max_layer)
    ax_c_emp.tick_params(labelsize=style_cfg.tick_fs, labelleft=False)
    ax_c_emp.set_ylabel("")
    prettify_axes(ax_c_emp)
    _annotate_panel(ax_c_emp, "ViT")

    def _match_ylim(ax_left, ax_right):
        lo = min(ax_left.get_ylim()[0], ax_right.get_ylim()[0])
        hi = max(ax_left.get_ylim()[1], ax_right.get_ylim()[1])
        ax_left.set_ylim(lo, hi)
        ax_right.set_ylim(lo, hi)

    _match_ylim(ax_b_th, ax_b_emp)
    _match_ylim(ax_c_th, ax_c_emp)

    center_shrink_axis(cax, width_scale=0.62, height_scale=0.75)
    if style_cfg.colorbar_pad:
        pos = cax.get_position()
        cax.set_position([pos.x0, pos.y0 - style_cfg.colorbar_pad, pos.width, pos.height])
    add_alpha_colorbar_horizontal_single(
        cax,
        alphas,
        colors,
        label=r"$\alpha$ (Derf)",
        cb_fs=style_cfg.alpha_legend_fs,
        max_tick_labels=style_cfg.colorbar_max_ticks,
    )
    rect = mpatches.Rectangle((1.02, 0.05), 0.08, 0.9, transform=cax.transAxes, facecolor="black", edgecolor="black", clip_on=False)
    cax.add_patch(rect)
    cax.text(1.06, -0.2, "pre-LN", transform=cax.transAxes, ha="center", va="top", fontsize=style_cfg.alpha_legend_fs)

    fig.canvas.draw()

    def _row_label(text, ax_left, ax_right):
        bb_left = ax_left.get_position()
        bb_right = ax_right.get_position()
        x = 0.5 * (bb_left.x0 + bb_right.x1)
        y = bb_left.y1 + style_cfg.row_label_offset
        fig.text(x, y, text, ha="center", va="bottom", fontsize=style_cfg.title_fs)

    _row_label(r"(b) $q^l$", ax_b_th, ax_b_emp)
    _row_label(r"(c) $p^l / q^l$", ax_c_th, ax_c_emp)

    three_panel_path = style_cfg.three_panel_save_path or style_cfg.save_path
    if three_panel_path is not None:
        fig.savefig(three_panel_path)
        print("Saved:", three_panel_path)

    plt.show()


# -------------------- notebook-specific helpers --------------------

def build_mean_field_cfg_for_vit_base(depth: int | None = None) -> MeanFieldConfig:
    # Matches the scaling used in vit_apjn_theory_plots.py for ViT-Base (d = 768).
    del depth
    scale = math.sqrt(768.0 / 1024.0)
    return MeanFieldConfig(
        sigma_w1=0.64 * scale,
        sigma_w2=1.28 * scale,
        sigma_o=0.64 * scale,
        sigma_v=0.64 * scale,
        sigma_a=0.64 * 0.64 * 768.0 / 1024.0,
    )


def simulate_recursions_full_with_mix(
    num_layers: int,
    p0: float,
    n_tokens: int,
    mode: str,
    alpha: float = 1.0,
    sigma_w1: float = 0.64,
    sigma_w2: float = 1.28,
    sigma_o: float = 0.64,
    sigma_v: float = 0.64,
    sigma_a: float = 0.64 * 0.64,
    q0: float = 1.0,
):
    # Full recurrence, plus recorded attention mixing diagnostics.
    L = int(num_layers)
    n = float(n_tokens)
    eps = 1e-12

    q = np.zeros(L + 1, dtype=float)
    p = np.zeros(L + 1, dtype=float)
    chi_att = np.zeros(L, dtype=float)
    chi_mlp = np.zeros(L, dtype=float)
    uq_arr = np.zeros(L, dtype=float)
    up_arr = np.zeros(L, dtype=float)
    q_mix_arr = np.zeros(L, dtype=float)
    p_mix_arr = np.zeros(L, dtype=float)

    q[0], p[0] = float(q0), float(p0)
    att_scale = (sigma_o ** 2) * (sigma_v ** 2)
    mlp_scale = (sigma_w1 ** 2) * (sigma_w2 ** 2)

    for l in range(L):
        ql, pl = q[l], p[l]
        if mode.lower() == "erf":
            uq = tilde_q_erf_np(ql, alpha)
            up = tilde_p_erf_np(ql, pl, alpha)
            beta = np.exp((sigma_a ** 2) * uq * (up - uq))
            gamma = np.exp((sigma_a ** 2) * up * (up - uq))
            q_mix = (uq + (n - 1.0) * up * beta) / (1.0 + (n - 1.0) * beta)
            p_mix = (uq + (n - 1.0) * up * gamma) / (1.0 + (n - 1.0) * gamma)
            qh = ql + att_scale * q_mix
            ph = pl + att_scale * p_mix
            chi_att[l] = 1.0
            chi_mlp[l] = 1.0 + mlp_scale * (2.0 * alpha**2 / np.pi) * (1.0 / np.sqrt(1.0 + 4.0 * alpha**2 * qh))
            u_half = tilde_q_erf_np(qh, alpha)
            v_half = tilde_p_erf_np(qh, ph, alpha)
            rho_half = np.clip(v_half / (u_half + eps), -1.0, 1.0)
            dq_mlp = 0.5 * mlp_scale * u_half
            dp_mlp = mlp_scale * u_half * kappa_relu_np(rho_half)
        elif mode.lower() == "layernorm":
            rho = np.clip(pl / (ql + eps), -1.0, 1.0)
            uq = 1.0
            up = rho
            beta = np.exp((sigma_a ** 2) * (rho - 1.0))
            gamma = np.exp((sigma_a ** 2) * rho * (rho - 1.0))
            q_mix = (1.0 + (n - 1.0) * rho * beta) / (1.0 + (n - 1.0) * beta)
            p_mix = (1.0 + (n - 1.0) * rho * gamma) / (1.0 + (n - 1.0) * gamma)
            qh = ql + att_scale * q_mix
            ph = pl + att_scale * p_mix
            chi_att[l] = 1.0
            chi_mlp[l] = 1.0 + mlp_scale / (2.0 * (qh + eps))
            rho_half = np.clip(ph / (qh + eps), -1.0, 1.0)
            dq_mlp = 0.5 * mlp_scale
            dp_mlp = mlp_scale * kappa_relu_np(rho_half)
        else:
            raise ValueError("mode must be 'erf' or 'layernorm'")

        uq_arr[l] = uq
        up_arr[l] = up
        q_mix_arr[l] = q_mix
        p_mix_arr[l] = p_mix
        q[l + 1] = qh + dq_mlp
        p[l + 1] = ph + dp_mlp

    chi = chi_att * chi_mlp
    J = np.ones(L + 1, dtype=float)
    for l in range(L - 1, -1, -1):
        J[l] = J[l + 1] * chi[l]

    J_direct = np.ones(L + 1, dtype=float)
    for l in range(L):
        J_direct[l + 1] = J_direct[l] * chi[l]

    return {
        "q": q,
        "p": p,
        "p_over_q": _safe_divide(p, q),
        "chi_att": chi_att,
        "chi_mlp": chi_mlp,
        "chi": chi,
        "J": J,
        "J_direct": J_direct,
        "uq": uq_arr,
        "up": up_arr,
        "q_mix": q_mix_arr,
        "p_mix": p_mix_arr,
    }


def simulate_recursions_simplified(
    num_layers: int,
    p0: float,
    n_tokens: int,
    mode: str,
    alpha: float = 1.0,
    sigma_w1: float = 0.64,
    sigma_w2: float = 1.28,
    sigma_o: float = 0.64,
    sigma_v: float = 0.64,
    sigma_a: float = 0.64 * 0.64,
    q0: float = 1.0,
):
    # Simplified recurrence requested in the notebook specification.
    del n_tokens, sigma_a
    L = int(num_layers)
    eps = 1e-12

    q = np.zeros(L + 1, dtype=float)
    p = np.zeros(L + 1, dtype=float)
    chi_att = np.zeros(L, dtype=float)
    chi_mlp = np.zeros(L, dtype=float)
    uq_arr = np.zeros(L, dtype=float)
    up_arr = np.zeros(L, dtype=float)
    q_mix_arr = np.zeros(L, dtype=float)
    p_mix_arr = np.zeros(L, dtype=float)

    q[0], p[0] = float(q0), float(p0)
    att_scale = (sigma_o ** 2) * (sigma_v ** 2)
    mlp_scale = (sigma_w1 ** 2) * (sigma_w2 ** 2)

    for l in range(L):
        ql, pl = q[l], p[l]
        if mode.lower() == "erf":
            uq = tilde_q_erf_np(ql, alpha)
            up = tilde_p_erf_np(ql, pl, alpha)
            q_mix = up
            p_mix = up
            qh = ql + att_scale * up
            ph = pl + att_scale * up
            chi_att[l] = 1.0
            chi_mlp[l] = 1.0 + mlp_scale * (2.0 * alpha**2 / np.pi) * (1.0 / np.sqrt(1.0 + 4.0 * alpha**2 * qh))
            u_half = tilde_q_erf_np(qh, alpha)
            v_half = tilde_p_erf_np(qh, ph, alpha)
            rho_half = np.clip(v_half / (u_half + eps), -1.0, 1.0)
            dq_mlp = 0.5 * mlp_scale * u_half
            dp_mlp = mlp_scale * u_half * kappa_relu_np(rho_half)
        elif mode.lower() == "layernorm":
            uq = 1.0
            up = np.clip(pl / (ql + eps), -1.0, 1.0)
            q_mix = up
            p_mix = up
            qh = ql + att_scale * up
            ph = pl + att_scale * up
            chi_att[l] = 1.0
            chi_mlp[l] = 1.0 + mlp_scale / (2.0 * (qh + eps))
            rho_half = np.clip(ph / (qh + eps), -1.0, 1.0)
            dq_mlp = 0.5 * mlp_scale
            dp_mlp = mlp_scale * kappa_relu_np(rho_half)
        else:
            raise ValueError("mode must be 'erf' or 'layernorm'")

        uq_arr[l] = uq
        up_arr[l] = up
        q_mix_arr[l] = q_mix
        p_mix_arr[l] = p_mix
        q[l + 1] = qh + dq_mlp
        p[l + 1] = ph + dp_mlp

    chi = chi_att * chi_mlp
    J = np.ones(L + 1, dtype=float)
    for l in range(L - 1, -1, -1):
        J[l] = J[l + 1] * chi[l]

    J_direct = np.ones(L + 1, dtype=float)
    for l in range(L):
        J_direct[l + 1] = J_direct[l] * chi[l]

    return {
        "q": q,
        "p": p,
        "p_over_q": _safe_divide(p, q),
        "chi_att": chi_att,
        "chi_mlp": chi_mlp,
        "chi": chi,
        "J": J,
        "J_direct": J_direct,
        "uq": uq_arr,
        "up": up_arr,
        "q_mix": q_mix_arr,
        "p_mix": p_mix_arr,
    }


def direct_curve_from_source(J_direct: np.ndarray, source_layer: int) -> np.ndarray:
    J_direct = np.asarray(J_direct, dtype=float)
    source = int(source_layer)
    return J_direct / np.maximum(J_direct[source], 1e-300)


def _save_notebook_figure(fig, filename: str):
    out_dir = Path("/content")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    fig.savefig(out_path)
    print("Saved:", out_path)
    return out_path


def _merge_saved_fit_hist_bundle(old_bundle, new_bundle):
    merged = dict(new_bundle)
    merged["inverse_mape"] = np.concatenate([
        np.asarray(old_bundle.get("inverse_mape", []), dtype=float),
        np.asarray(new_bundle.get("inverse_mape", []), dtype=float),
    ])
    merged["direct_mape"] = np.concatenate([
        np.asarray(old_bundle.get("direct_mape", []), dtype=float),
        np.asarray(new_bundle.get("direct_mape", []), dtype=float),
    ])
    merged["samples"] = list(old_bundle.get("samples", [])) + list(new_bundle.get("samples", []))
    return merged


def _merge_saved_points_bundle(old_bundle, new_bundle):
    merged = dict(new_bundle)
    merged["points"] = list(old_bundle.get("points", [])) + list(new_bundle.get("points", []))
    return merged


def _folder_name_with_postfix(base_name: str, result_postfix: str | None):
    postfix = str(result_postfix or "").strip()
    if not postfix:
        return base_name
    return f"{base_name}_{postfix}"


def _save_bundle_pickle(bundle, save_root, folder_name, filename, *, rewrite=True, merge_kind=None):
    out_dir = Path(save_root) / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    if out_path.exists() and not rewrite:
        old_bundle = load_saved_bundle(out_path)
        if merge_kind == "fit_hist":
            bundle = _merge_saved_fit_hist_bundle(old_bundle, bundle)
        elif merge_kind == "points":
            bundle = _merge_saved_points_bundle(old_bundle, bundle)
        else:
            raise ValueError("merge_kind must be 'fit_hist' or 'points' when rewrite=False")
    with out_path.open("wb") as f:
        pickle.dump(bundle, f)
    print("Saved bundle:", out_path)
    return out_path, bundle


def load_saved_bundle(path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def _maybe_load_bundle(bundle_or_path):
    if isinstance(bundle_or_path, (str, Path)):
        return load_saved_bundle(bundle_or_path)
    return bundle_or_path


def _resolve_plot_text_sizes(
    style_cfg: FinalThreePanelStyleConfig,
    *,
    tick_fs=None,
    label_fs=None,
    alpha_legend_fs=None,
    title_fs=None,
    annotation_fs=None,
):
    return {
        "tick_fs": style_cfg.tick_fs if tick_fs is None else tick_fs,
        "label_fs": style_cfg.label_fs if label_fs is None else label_fs,
        "alpha_legend_fs": style_cfg.alpha_legend_fs if alpha_legend_fs is None else alpha_legend_fs,
        "title_fs": style_cfg.title_fs if title_fs is None else title_fs,
        "annotation_fs": style_cfg.annotation_fs if annotation_fs is None else annotation_fs,
    }


def _draw_alpha_preln_legend_like_equangular(cax, alphas, colors, alpha_legend_fs):
    add_alpha_colorbar_horizontal_single(
        cax,
        alphas,
        colors,
        label=r"$\alpha$ (Derf)",
        cb_fs=alpha_legend_fs,
    )
    rect = mpatches.Rectangle(
        (1.02, 0.05),
        0.08,
        0.9,
        transform=cax.transAxes,
        facecolor="black",
        edgecolor="black",
        clip_on=False,
    )
    cax.add_patch(rect)
    cax.text(
        1.06,
        -0.2,
        "pre-LN",
        transform=cax.transAxes,
        ha="center",
        va="top",
        fontsize=alpha_legend_fs,
    )


def _resolve_alpha_key(mapping, alpha_value: float, tol: float = 1e-9) -> float:
    keys = np.asarray(sorted(float(k) for k in mapping.keys()), dtype=float)
    if keys.size == 0:
        raise KeyError("alpha mapping is empty")
    idx = int(np.argmin(np.abs(keys - float(alpha_value))))
    key = float(keys[idx])
    if abs(key - float(alpha_value)) > tol:
        raise KeyError(
            f"Could not resolve alpha={alpha_value} within tolerance {tol}. "
            f"Available values: {keys.tolist()}"
        )
    return key


def _extract_inverse_points(apjn_bundle):
    preln = apjn_bundle["preln_with_J"]["J_raw"]
    layers = sorted(int(k) for k in preln.keys())
    derf = {}
    for entry in apjn_bundle["derf_pack_with_J"]["results"]:
        derf[float(entry["alpha"])] = {int(k): float(v) for k, v in entry["J_raw"].items()}
    return layers, {int(k): float(v) for k, v in preln.items()}, derf


def _extract_direct_points(apjn_bundle):
    source = int(apjn_bundle.get("direct_source_block", 0))
    preln_raw = apjn_bundle["preln_with_J"].get("direct_J_raw", {})
    preln = {source + int(k): float(np.mean(np.asarray(v, dtype=float))) for k, v in preln_raw.items()}
    derf = {}
    for entry in apjn_bundle["derf_pack_with_J"]["results"]:
        derf[float(entry["alpha"])] = {
            source + int(k): float(np.mean(np.asarray(v, dtype=float)))
            for k, v in (entry.get("direct_J_raw") or {}).items()
        }
    layers = sorted(preln.keys()) if preln else sorted({k for d in derf.values() for k in d.keys()})
    return source, layers, preln, derf


def plot_equangular_p_inverse_direct_figure(
    perm_empirical_bundle,
    perm_theory_bundle,
    apjn_bundle,
    style_cfg: FinalThreePanelStyleConfig,
    panel_width_ratios=(1.0, 1.3, 1.3),
    panel_gap_ab=0.18,
    panel_gap_bc=0.18,
    tick_fs=None,
    label_fs=None,
    alpha_legend_fs=None,
    title_fs=None,
    annotation_fs=None,
):
    # Figure with (a) p, (b) inverse APJN, (c) direct APJN.
    alphas = np.asarray(perm_empirical_bundle["perm_cfg"]["alphas"], dtype=float)
    colors = _make_alpha_colors(alphas)
    if len(panel_width_ratios) != 3:
        raise ValueError("panel_width_ratios must have length 3 for panels (a), (b), (c).")
    sizes = _resolve_plot_text_sizes(
        style_cfg,
        tick_fs=tick_fs,
        label_fs=label_fs,
        alpha_legend_fs=alpha_legend_fs,
        title_fs=title_fs,
        annotation_fs=annotation_fs,
    )
    fig = plt.figure(figsize=(14.8, 5.8))
    gs = fig.add_gridspec(
        3,
        5,
        height_ratios=[1.0, 1.0, 0.18],
        width_ratios=[panel_width_ratios[0], panel_gap_ab, panel_width_ratios[1], panel_gap_bc, panel_width_ratios[2]],
        hspace=0.18,
        wspace=0.0,
    )
    gs_p = gs[0:2, 0].subgridspec(2, 1, hspace=0.08)
    ax_p_th = fig.add_subplot(gs_p[0, 0])
    ax_p_vit = fig.add_subplot(gs_p[1, 0], sharex=ax_p_th)
    ax_inv = fig.add_subplot(gs[0:2, 2])
    ax_dir = fig.add_subplot(gs[0:2, 4])
    cax = fig.add_subplot(gs[2, :])

    l = np.asarray(perm_theory_bundle["l"], dtype=float)
    ax_p_th.plot(l, perm_theory_bundle["preln"]["p"], color="black", lw=style_cfg.line_width, zorder=10)
    ax_p_vit.plot(
        perm_empirical_bundle["preln"]["l"],
        perm_empirical_bundle["preln"]["p"],
        color="black",
        lw=style_cfg.line_width,
        marker="o",
        ms=0.2,
        zorder=11,
    )
    for i, a in enumerate(alphas):
        ax_p_th.plot(l, perm_theory_bundle["derf"][float(a)]["p"], color=colors[i], lw=style_cfg.line_width, zorder=2)
        ax_p_vit.plot(
            perm_empirical_bundle["derf"][float(a)]["l"],
            perm_empirical_bundle["derf"][float(a)]["p"],
            color=colors[i],
            lw=style_cfg.line_width,
            marker="o",
            ms=0.2,
            zorder=3,
        )
    ax_p_th.set_title(r"(a) $P^b$", fontsize=sizes["title_fs"])
    ax_p_th.set_ylabel("$P$", fontsize=sizes["label_fs"])
    ax_p_vit.set_ylabel("$P$", fontsize=sizes["label_fs"])
    ax_p_vit.set_xlabel(r"$b$", fontsize=sizes["label_fs"])
    prettify_axes(ax_p_th)
    prettify_axes(ax_p_vit)
    ax_p_th.tick_params(labelbottom=False, labelsize=sizes["tick_fs"])
    ax_p_vit.tick_params(labelsize=sizes["tick_fs"])
    ax_p_th.text(
        style_cfg.panel_label_xoffset,
        0.95,
        "theory",
        transform=ax_p_th.transAxes,
        ha="left",
        va="top",
        fontsize=sizes["annotation_fs"],
    )
    ax_p_vit.text(
        style_cfg.panel_label_xoffset,
        0.95,
        "ViT",
        transform=ax_p_vit.transAxes,
        ha="left",
        va="top",
        fontsize=sizes["annotation_fs"],
    )

    inv_layers, inv_preln, inv_derf = _extract_inverse_points(apjn_bundle)
    ax_inv.plot(l, perm_theory_bundle["preln"]["J"], color="black", lw=style_cfg.line_width, zorder=10)
    ax_inv.scatter(inv_layers, [inv_preln[k] for k in inv_layers], color="black", s=18, zorder=11)
    for i, a in enumerate(alphas):
        ax_inv.plot(l, perm_theory_bundle["derf"][float(a)]["J"], color=colors[i], lw=style_cfg.line_width, zorder=2)
        y = [inv_derf[float(a)][k] for k in inv_layers]
        ax_inv.scatter(inv_layers, y, color=colors[i], s=18, zorder=3)
    ax_inv.set_title(r"(b) backward APJN $\mathcal{J}^{\, B, b}$", fontsize=sizes["title_fs"])
    ax_inv.set_xlabel(r"$b$", fontsize=sizes["label_fs"])
    ax_inv.set_ylabel(r"$\mathcal{J}^{\, B, b}$", fontsize=sizes["label_fs"])
    ax_inv.set_yscale("log")
    prettify_log_axis(ax_inv, "y")
    prettify_axes(ax_inv)
    ax_inv.tick_params(labelsize=sizes["tick_fs"])

    direct_source, direct_layers, dir_preln, dir_derf = _extract_direct_points(apjn_bundle)
    pre_curve = direct_curve_from_source(perm_theory_bundle["preln"]["J_direct"], direct_source)
    ax_dir.plot(l, pre_curve, color="black", lw=style_cfg.line_width, zorder=10)
    if direct_layers:
        ax_dir.scatter(direct_layers, [dir_preln[k] for k in direct_layers], color="black", s=18, zorder=11)
    for i, a in enumerate(alphas):
        derf_curve = direct_curve_from_source(perm_theory_bundle["derf"][float(a)]["J_direct"], direct_source)
        ax_dir.plot(l, derf_curve, color=colors[i], lw=style_cfg.line_width, zorder=2)
        if direct_layers:
            y = [dir_derf[float(a)][k] for k in direct_layers]
            ax_dir.scatter(direct_layers, y, color=colors[i], s=18, zorder=3)
    ax_dir.set_title(r"(c) APJN $\mathcal{J}^{\, b, 0}$", fontsize=sizes["title_fs"])
    ax_dir.set_xlabel(r"$b$", fontsize=sizes["label_fs"])
    ax_dir.set_ylabel(r"$\mathcal{J}^{\, b, 0}$", fontsize=sizes["label_fs"])
    ax_dir.set_yscale("log")
    prettify_log_axis(ax_dir, "y")
    prettify_axes(ax_dir)
    ax_dir.tick_params(labelsize=sizes["tick_fs"])

    center_shrink_axis(cax, width_scale=0.62, height_scale=0.75)
    if style_cfg.colorbar_pad:
        pos = cax.get_position()
        cax.set_position([pos.x0, pos.y0 - style_cfg.colorbar_pad, pos.width, pos.height])
    _draw_alpha_preln_legend_like_equangular(cax, alphas, colors, sizes["alpha_legend_fs"])

    _save_notebook_figure(fig, "equiangular_p_inverse_direct_figure.pdf")
    plt.show()
    return fig


def prepare_simplified_inverse_comparison(
    apjn_bundle,
    mean_field_cfg: MeanFieldConfig,
    p0: float,
    alpha_for_mix: float,
):
    # Full/simplified Derf inverse APJN curves and mix diagnostics.
    depth = int(apjn_bundle["depth"])
    n_tokens = int(apjn_bundle["seq_len"] - 1)
    alphas = np.asarray(apjn_bundle["derf_pack_with_J"]["alphas"], dtype=float)
    full = {}
    simplified = {}
    eps = 1e-12
    for a in alphas:
        full[float(a)] = simulate_recursions_full_with_mix(
            num_layers=depth,
            q0=1.0,
            p0=float(p0),
            n_tokens=n_tokens,
            mode="erf",
            alpha=float(a),
            sigma_w1=mean_field_cfg.sigma_w1,
            sigma_w2=mean_field_cfg.sigma_w2,
            sigma_o=mean_field_cfg.sigma_o,
            sigma_v=mean_field_cfg.sigma_v,
            sigma_a=mean_field_cfg.sigma_a,
        )
        simplified[float(a)] = simulate_recursions_simplified(
            num_layers=depth,
            q0=1.0,
            p0=float(p0),
            n_tokens=n_tokens,
            mode="erf",
            alpha=float(a),
            sigma_w1=mean_field_cfg.sigma_w1,
            sigma_w2=mean_field_cfg.sigma_w2,
            sigma_o=mean_field_cfg.sigma_o,
            sigma_v=mean_field_cfg.sigma_v,
            sigma_a=mean_field_cfg.sigma_a,
        )
    preln_full = simulate_recursions_full_with_mix(
        num_layers=depth,
        q0=1.0,
        p0=float(p0),
        n_tokens=n_tokens,
        mode="layernorm",
        sigma_w1=mean_field_cfg.sigma_w1,
        sigma_w2=mean_field_cfg.sigma_w2,
        sigma_o=mean_field_cfg.sigma_o,
        sigma_v=mean_field_cfg.sigma_v,
        sigma_a=mean_field_cfg.sigma_a,
    )
    mix_alpha_key = _resolve_alpha_key(full, float(alpha_for_mix), tol=1e-8)
    mix_layers = np.arange(depth, dtype=int)
    mix_q_err_derf = {
        float(a): np.abs(full[float(a)]["q_mix"] - full[float(a)]["up"]) / np.maximum(np.abs(full[float(a)]["q_mix"]), eps)
        for a in alphas
    }
    mix_p_err_derf = {
        float(a): np.abs(full[float(a)]["p_mix"] - full[float(a)]["up"]) / np.maximum(np.abs(full[float(a)]["p_mix"]), eps)
        for a in alphas
    }
    mix_q_err_preln = np.abs(preln_full["q_mix"] - preln_full["up"]) / np.maximum(np.abs(preln_full["q_mix"]), eps)
    mix_p_err_preln = np.abs(preln_full["p_mix"] - preln_full["up"]) / np.maximum(np.abs(preln_full["p_mix"]), eps)
    vit_derf = {
        float(entry["alpha"]): {int(k): float(v) for k, v in entry["J_raw"].items()}
        for entry in apjn_bundle["derf_pack_with_J"]["results"]
    }
    direct_source_block = int(apjn_bundle.get("direct_source_block", 0))
    vit_derf_direct = {}
    for entry in apjn_bundle["derf_pack_with_J"]["results"]:
        alpha = float(entry["alpha"])
        raw_direct = entry.get("direct_J_raw") or {}
        vit_derf_direct[alpha] = {
            direct_source_block + int(k): float(np.mean(np.asarray(v, dtype=float)))
            for k, v in raw_direct.items()
        }
    return {
        "alphas": alphas,
        "layers": np.arange(depth + 1, dtype=int),
        "vit_derf": vit_derf,
        "vit_derf_direct": vit_derf_direct,
        "full": full,
        "simplified": simplified,
        "mix_alpha": float(mix_alpha_key),
        "mix_layers": mix_layers,
        "mix_q_err_derf": mix_q_err_derf,
        "mix_p_err_derf": mix_p_err_derf,
        "mix_q_err_preln": mix_q_err_preln,
        "mix_p_err_preln": mix_p_err_preln,
        "direct_source_block": direct_source_block,
        "preln_full": preln_full,
    }


def plot_simplified_inverse_and_mix_figure(
    bundle,
    style_cfg: FinalThreePanelStyleConfig,
    panel_gap_ab=0.18,
    panel_gap_bc=0.18,
    mix_panel_hspace=0.12,
    alpha_colorbar_width_scale=0.5,
    alpha_colorbar_height_scale=0.75,
    legend_row_height=0.24,
    tick_fs=None,
    label_fs=None,
    alpha_legend_fs=None,
    title_fs=None,
    annotation_fs=None,
):
    alphas = np.asarray(bundle["alphas"], dtype=float)
    colors = _make_alpha_colors(alphas)
    sizes = _resolve_plot_text_sizes(
        style_cfg,
        tick_fs=tick_fs,
        label_fs=label_fs,
        alpha_legend_fs=alpha_legend_fs,
        title_fs=title_fs,
        annotation_fs=annotation_fs,
    )
    fig = plt.figure(figsize=(14.8, 6.0))
    gs = fig.add_gridspec(
        2,
        5,
        height_ratios=[1.0, legend_row_height],
        width_ratios=[1.0, panel_gap_ab, 1.0, panel_gap_bc, 0.55],
        hspace=0.14,
        wspace=0.0,
    )
    ax_inv = fig.add_subplot(gs[0, 0])
    ax_dir = fig.add_subplot(gs[0, 2])
    gs_mix = gs[0, 4].subgridspec(2, 1, hspace=mix_panel_hspace)
    ax_mix_top = fig.add_subplot(gs_mix[0, 0])
    ax_mix_bot = fig.add_subplot(gs_mix[1, 0], sharex=ax_mix_top)
    cax = fig.add_subplot(gs[1, :])

    for i, a in enumerate(alphas):
        full = bundle["full"][float(a)]
        simp = bundle["simplified"][float(a)]
        vit = bundle["vit_derf"][float(a)]
        inv_layers = sorted(vit.keys())
        ax_inv.plot(bundle["layers"], full["J"], color=colors[i], lw=style_cfg.line_width)
        ax_inv.plot(bundle["layers"], simp["J"], color="0.5", lw=style_cfg.line_width, ls="--")
        ax_inv.scatter(inv_layers, [vit[k] for k in inv_layers], color=colors[i], s=18)

        full_direct = direct_curve_from_source(full["J_direct"], bundle["direct_source_block"])
        simp_direct = direct_curve_from_source(simp["J_direct"], bundle["direct_source_block"])
        vit_direct = bundle["vit_derf_direct"].get(float(a), {})
        dir_layers = sorted(vit_direct.keys())
        ax_dir.plot(bundle["layers"], full_direct, color=colors[i], lw=style_cfg.line_width)
        ax_dir.plot(bundle["layers"], simp_direct, color="0.5", lw=style_cfg.line_width, ls="--")
        if dir_layers:
            ax_dir.scatter(dir_layers, [vit_direct[k] for k in dir_layers], color=colors[i], s=18)

    ax_inv.set_title("(a) backward APJN:\nfull vs. simplified", fontsize=sizes["title_fs"])
    ax_inv.set_xlabel(r"$b$", fontsize=sizes["label_fs"])
    ax_inv.set_ylabel(r"$\mathcal{J}^{\, B, b}$", fontsize=sizes["label_fs"])
    ax_inv.set_yscale("log")
    prettify_log_axis(ax_inv, "y")
    prettify_axes(ax_inv)
    ax_inv.legend(
        handles=[
            Line2D([0], [0], color="black", lw=style_cfg.line_width, label="full theory"),
            Line2D([0], [0], color="0.5", lw=style_cfg.line_width, ls="--", label="simplified theory"),
            Line2D([0], [0], color="black", marker="o", ls="none", markersize=4.5, label="ViT"),
        ],
        frameon=False,
        loc="best",
        fontsize=sizes["annotation_fs"],
    )
    ax_inv.tick_params(labelsize=sizes["tick_fs"])

    ax_dir.set_title(r"(b) APJN: full vs. simplified", fontsize=sizes["title_fs"])
    ax_dir.set_xlabel(r"$b$", fontsize=sizes["label_fs"])
    ax_dir.set_ylabel(r"$\mathcal{J}^{\, b, 0}$", fontsize=sizes["label_fs"])
    ax_dir.set_yscale("log")
    prettify_log_axis(ax_dir, "y")
    prettify_axes(ax_dir)
    ax_dir.tick_params(labelsize=sizes["tick_fs"])

    for i, a in enumerate(alphas):
        ax_mix_top.plot(
            bundle["mix_layers"],
            bundle["mix_q_err_derf"][float(a)],
            color=colors[i],
            lw=style_cfg.line_width,
        )
        ax_mix_bot.plot(
            bundle["mix_layers"],
            bundle["mix_p_err_derf"][float(a)],
            color=colors[i],
            lw=style_cfg.line_width,
        )
    ax_mix_top.plot(bundle["mix_layers"], bundle["mix_q_err_preln"], color="black", lw=style_cfg.line_width, zorder=10)
    ax_mix_bot.plot(bundle["mix_layers"], bundle["mix_p_err_preln"], color="black", lw=style_cfg.line_width, zorder=10)

    ax_mix_top.set_title("(c) " + r"$|f_q-\tilde{p}|/f_q$", fontsize=sizes["title_fs"])
    ax_mix_bot.set_title(r"$|f_p-\tilde{p}|/f_p$", fontsize=sizes["title_fs"])
    ax_mix_bot.set_xlabel(r"$b$", fontsize=sizes["label_fs"])
    ax_mix_top.set_ylabel("rel. error", fontsize=sizes["label_fs"], labelpad=16)
    ax_mix_bot.set_ylabel("rel. error", fontsize=sizes["label_fs"], labelpad=16)
    ax_mix_top.yaxis.set_label_coords(-0.24, 0.5)
    ax_mix_bot.yaxis.set_label_coords(-0.24, 0.5)
    ax_mix_top.set_yscale("log")
    ax_mix_bot.set_yscale("log")
    prettify_log_axis(ax_mix_top, "y")
    prettify_log_axis(ax_mix_bot, "y")
    prettify_axes(ax_mix_top)
    prettify_axes(ax_mix_bot)
    ax_mix_top.tick_params(labelsize=sizes["tick_fs"], labelbottom=False)
    ax_mix_bot.tick_params(labelsize=sizes["tick_fs"])

    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.11)

    # Final legend placement happens after subplot layout so the pad/scale knobs act predictably.
    pos = cax.get_position()
    new_w = pos.width * alpha_colorbar_width_scale
    new_h = pos.height * alpha_colorbar_height_scale
    new_x = pos.x0 + 0.5 * (pos.width - new_w)
    new_y = pos.y0 + 0.5 * (pos.height - new_h) - style_cfg.colorbar_pad
    cax.set_position([new_x, new_y, new_w, new_h])
    _draw_alpha_preln_legend_like_equangular(cax, alphas, colors, sizes["alpha_legend_fs"])
    _save_notebook_figure(fig, "simplified_inverse_direct_mix_figure.pdf")
    plt.show()
    return fig


def plot_simplified_qp_comparison_grid(
    bundle,
    style_cfg: FinalThreePanelStyleConfig,
    grid_wspace=0.22,
    grid_hspace=0.18,
    tick_fs=None,
    label_fs=None,
    title_fs=None,
):
    alphas = np.asarray(bundle["alphas"], dtype=float)
    colors = _make_alpha_colors(alphas)
    sizes = _resolve_plot_text_sizes(
        style_cfg,
        tick_fs=tick_fs,
        label_fs=label_fs,
        title_fs=title_fs,
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.0), sharex=True)
    ax_q_full, ax_q_simp = axes[0, 0], axes[0, 1]
    ax_p_full, ax_p_simp = axes[1, 0], axes[1, 1]
    layers = bundle["layers"]

    for i, a in enumerate(alphas):
        full = bundle["full"][float(a)]
        simp = bundle["simplified"][float(a)]
        ax_q_full.plot(layers, full["q"], color=colors[i], lw=style_cfg.line_width)
        ax_q_simp.plot(layers, simp["q"], color=colors[i], lw=style_cfg.line_width)
        ax_p_full.plot(layers, full["p"], color=colors[i], lw=style_cfg.line_width)
        ax_p_simp.plot(layers, simp["p"], color=colors[i], lw=style_cfg.line_width)

    ax_q_full.set_title(r"(a) $q$ full", fontsize=sizes["title_fs"])
    ax_q_simp.set_title(r"$q$ simplified", fontsize=sizes["title_fs"])
    ax_p_full.set_title(r"(b) $p$ full", fontsize=sizes["title_fs"])
    ax_p_simp.set_title(r"$p$ simplified", fontsize=sizes["title_fs"])

    ax_q_full.set_ylabel(r"$q^l$", fontsize=sizes["label_fs"])
    ax_p_full.set_ylabel(r"$p^l$", fontsize=sizes["label_fs"])
    ax_p_full.set_xlabel(r"$b$", fontsize=sizes["label_fs"])
    ax_p_simp.set_xlabel(r"$b$", fontsize=sizes["label_fs"])

    for ax in [ax_q_full, ax_q_simp, ax_p_full, ax_p_simp]:
        prettify_axes(ax)
        ax.tick_params(labelsize=sizes["tick_fs"])

    fig.subplots_adjust(wspace=grid_wspace, hspace=grid_hspace)
    plt.show()
    return fig


def run_cifar_fit_histograms(
    model_cfg: ModelConfig,
    mean_field_cfg: MeanFieldConfig,
    *,
    alphas,
    n_samples: int = 16,
    layer_stride: int = 4,
    batch_seed: int = 0,
    std_threshold: float = 0.2,
    max_epochs_to_search: int = 20,
    j_num_draws: int = 10,
    q0_values=None,
    p0_values=None,
    preln_scale_values=None,
    rescale_vit_preln_apjn: bool = False,
    save_results: bool = False,
    save_root: str = "/content/drive/MyDrive/ml_projects/mapes_variance",
    rewrite: bool = True,
    result_postfix: str = "",
):
    # Repeats inverse/direct fitting over multiple CIFAR samples.
    rng = np.random.default_rng()
    depth = int(model_cfg.depth)
    loader_seed_random = int(rng.integers(0, 2**31 - 1))
    inverse_layers = tuple(
        l for l in range(0, depth + 1, int(layer_stride))
        if 0 < int(l) < depth
    )
    direct_layers = tuple(
        l for l in range(int(layer_stride), depth + 1, int(layer_stride))
        if 0 < int(l) < depth
    )
    if bool(rescale_vit_preln_apjn) and depth > 1 and 1 not in direct_layers:
        direct_layers = tuple(sorted(set(direct_layers) | {1}))
    fit_cfg = APJNFitConfig(
        metric="mape",
        q0_values=tuple(np.asarray(q0_values, dtype=float)) if q0_values is not None else tuple(np.linspace(0.0, 2.0, 41)),
        p0_values=tuple(np.asarray(p0_values, dtype=float)) if p0_values is not None else tuple(np.linspace(0.0, 2.0, 41)),
        q0_num=41,
        p0_num=41,
        separate_panel_d_fits=False,
        preln_scale_values=tuple(np.asarray(preln_scale_values, dtype=float)) if preln_scale_values is not None else tuple(np.linspace(0.25, 4.0, 61)),
        preln_scale_num=61,
        refine_radius=0.2,
        rescale_vit_preln_apjn=bool(rescale_vit_preln_apjn),
    )

    inverse_mape = []
    direct_mape = []
    sample_summaries = []

    for draw_index in tqdm(range(int(n_samples)), desc="run_cifar_fit_histograms", leave=False):
        apjn_cfg = APJNCifarConfig(
            input_source="cifar",
            batch_size=1,
            cifar_batch_seed=loader_seed_random,
            cifar_batch_draw_index=int(draw_index),
            cifar_std_threshold=float(std_threshold),
            cifar_max_epochs_to_search=int(max_epochs_to_search),
            apjn_layers=inverse_layers,
            direct_layers=direct_layers,
            direct_source_block=0,
            alphas=tuple(np.asarray(alphas, dtype=float)),
            j_num_draws=int(j_num_draws),
            j_normalize_by="Y",
            num_model_inits=1,
        )
        model_cfg_draw = replace(model_cfg, seed=int(rng.integers(0, 2**31 - 1)))
        bundle = run_cifar_apjn_experiment(model_cfg_draw, apjn_cfg)
        inv_fit = fit_theory_for_apjn(bundle, fit_cfg, mean_field_cfg)
        panel_d_cfg = PanelDConfig(
            num_layers=depth,
            q0=1.0,
            p0=0.5,
            n_tokens=int(bundle["seq_len"] - 1),
            sigma_w1=mean_field_cfg.sigma_w1,
            sigma_w2=mean_field_cfg.sigma_w2,
            sigma_o=mean_field_cfg.sigma_o,
            sigma_v=mean_field_cfg.sigma_v,
            sigma_a=mean_field_cfg.sigma_a,
        )
        direct_fit = fit_panel_d_direct_initial_conditions(bundle, fit_cfg, mean_field_cfg, panel_d_cfg, alphas)
        inverse_mape.append(float(inv_fit["value"]))
        direct_mape.append(float(direct_fit["value"]))
        sample_summaries.append({
            "draw_index": int(draw_index),
            "inverse_fit": inv_fit,
            "direct_fit": direct_fit,
            "batch_meta": bundle.get("batch_meta"),
        })

    out = {
        "inverse_mape": np.asarray(inverse_mape, dtype=float),
        "direct_mape": np.asarray(direct_mape, dtype=float),
        "samples": sample_summaries,
        "depth": int(depth),
        "layer_stride": int(layer_stride),
        "alphas": np.asarray(alphas, dtype=float),
        "config": {
            "n_samples": int(n_samples),
            "batch_seed": None,
            "loader_seed_random": int(loader_seed_random),
            "std_threshold": float(std_threshold),
            "max_epochs_to_search": int(max_epochs_to_search),
            "j_num_draws": int(j_num_draws),
            "randomized_sampling": True,
            "rescale_vit_preln_apjn": bool(rescale_vit_preln_apjn),
        },
    }
    if save_results:
        saved_path, out = _save_bundle_pickle(
            out,
            save_root=save_root,
            folder_name=_folder_name_with_postfix(
                f"fit_hist_depth{depth}_stride{int(layer_stride)}",
                result_postfix,
            ),
            filename="results.pkl",
            rewrite=bool(rewrite),
            merge_kind="fit_hist",
        )
        out["saved_path"] = str(saved_path)
    return out


def run_random_direct_scatter(
    model_cfg: ModelConfig,
    *,
    alphas,
    n_samples: int = 16,
    layers_per_sample: int = 8,
    preln_weight: float = 1.0,
    rescale_vit_preln_apjn: bool = False,
    batch_seed: int = 0,
    scatter_seed: int = 0,
    std_threshold: float = 0.2,
    max_epochs_to_search: int = 20,
    j_num_draws: int = 10,
    save_results: bool = False,
    save_root: str = "/content/drive/MyDrive/ml_projects/mapes_variance",
    rewrite: bool = True,
    result_postfix: str = "",
):
    # Empirical direct APJN points on random layer subsets per CIFAR sample.
    # For each sample, choose one model uniformly from Derf(alpha) and pre-LN.
    rng = np.random.default_rng()
    depth = int(model_cfg.depth)
    loader_seed_random = int(rng.integers(0, 2**31 - 1))
    candidate_layers = np.arange(1, depth + 1, dtype=int)
    points = []
    model_choices = ["preln"] + [float(a) for a in np.asarray(alphas, dtype=float)]
    model_probs = np.ones(len(model_choices), dtype=float)
    model_probs[0] = float(preln_weight)
    if np.any(model_probs <= 0):
        raise ValueError("preln_weight must be positive.")
    model_probs = model_probs / model_probs.sum()

    for draw_index in tqdm(range(int(n_samples)), desc="run_random_direct_scatter", leave=False):
        layer_count = min(int(layers_per_sample), candidate_layers.size)
        chosen_layers = tuple(sorted(int(x) for x in rng.choice(candidate_layers, size=layer_count, replace=False)))
        model_choice = model_choices[int(rng.choice(len(model_choices), p=model_probs))]
        seed_all(int(rng.integers(0, 2**31 - 1)))
        cuda_cleanup()
        samples, _, batch_meta = get_cifar_batch(
            batch_size=1,
            img_size=model_cfg.img_size,
            num_classes=model_cfg.num_classes,
            loader_seed=loader_seed_random,
            draw_index=int(draw_index),
            std_threshold=float(std_threshold),
            max_epochs_to_search=int(max_epochs_to_search),
        )
        model_cfg_draw = replace(model_cfg, seed=int(rng.integers(0, 2**31 - 1)))
        model = build_vit(model_cfg_draw, use_derf=model_choice != "preln")
        try:
            if model_choice != "preln":
                set_all_derf_alpha_(model, float(model_choice))
            direct_layers_eval = chosen_layers
            if bool(rescale_vit_preln_apjn) and model_choice == "preln" and 1 not in direct_layers_eval:
                direct_layers_eval = tuple(sorted(set(direct_layers_eval) | {1}))
            direct = estimate_direct_J_points_hutchinson(
                model,
                samples,
                direct_layers=direct_layers_eval,
                direct_source_block=0,
                j_num_draws=int(j_num_draws),
                j_normalize_by="Y",
            )
        finally:
            del model
            cuda_cleanup()
        direct_out = dict(direct)
        if bool(rescale_vit_preln_apjn) and model_choice == "preln":
            anchor = float(direct_out[1])
            if not np.isfinite(anchor) or abs(anchor) <= 1e-12:
                raise RuntimeError("Cannot rescale pre-LN direct APJN in run_random_direct_scatter: layer-1 value is invalid or zero.")
            direct_out = {int(layer): float(value) / anchor for layer, value in direct_out.items()}
            if 1 not in chosen_layers and 1 in direct_out:
                del direct_out[1]
        for layer_rel, value in direct_out.items():
            points.append({
                "sample": int(draw_index),
                "layer": int(layer_rel),
                "value": float(value),
                "model_kind": "preln" if model_choice == "preln" else "derf",
                "alpha": None if model_choice == "preln" else float(model_choice),
                "chosen_layers": chosen_layers,
                "stored_layers": tuple(sorted(int(k) for k in direct_out.keys())),
                "batch_meta": batch_meta,
            })

    out = {
        "points": points,
        "alphas": np.asarray(alphas, dtype=float),
        "depth": int(depth),
        "layers_per_sample": int(layers_per_sample),
        "config": {
            "n_samples": int(n_samples),
            "preln_weight": float(preln_weight),
            "rescale_vit_preln_apjn": bool(rescale_vit_preln_apjn),
            "batch_seed": None,
            "scatter_seed": None,
            "loader_seed_random": int(loader_seed_random),
            "std_threshold": float(std_threshold),
            "max_epochs_to_search": int(max_epochs_to_search),
            "j_num_draws": int(j_num_draws),
            "randomized_sampling": True,
        },
    }
    if save_results:
        saved_path, out = _save_bundle_pickle(
            out,
            save_root=save_root,
            folder_name=_folder_name_with_postfix(
                f"random_direct_depth{depth}_layers{int(layers_per_sample)}",
                result_postfix,
            ),
            filename="results.pkl",
            rewrite=bool(rewrite),
            merge_kind="points",
        )
        out["saved_path"] = str(saved_path)
    return out


def run_random_inverse_scatter(
    model_cfg: ModelConfig,
    *,
    alphas,
    n_samples: int = 16,
    layers_per_sample: int = 8,
    preln_weight: float = 1.0,
    batch_seed: int = 0,
    scatter_seed: int = 0,
    std_threshold: float = 0.2,
    max_epochs_to_search: int = 20,
    j_num_draws: int = 10,
    save_results: bool = False,
    save_root: str = "/content/drive/MyDrive/ml_projects/mapes_variance",
    rewrite: bool = True,
    result_postfix: str = "",
):
    # Empirical backward APJN points on random layer subsets per CIFAR sample.
    # For each sample, choose one model uniformly from Derf(alpha) and pre-LN.
    rng = np.random.default_rng()
    depth = int(model_cfg.depth)
    loader_seed_random = int(rng.integers(0, 2**31 - 1))
    candidate_layers = np.arange(1, depth, dtype=int)
    points = []
    model_choices = ["preln"] + [float(a) for a in np.asarray(alphas, dtype=float)]
    model_probs = np.ones(len(model_choices), dtype=float)
    model_probs[0] = float(preln_weight)
    if np.any(model_probs <= 0):
        raise ValueError("preln_weight must be positive.")
    model_probs = model_probs / model_probs.sum()

    for draw_index in tqdm(range(int(n_samples)), desc="run_random_inverse_scatter", leave=False):
        layer_count = min(int(layers_per_sample), candidate_layers.size)
        chosen_layers = tuple(sorted(int(x) for x in rng.choice(candidate_layers, size=layer_count, replace=False)))
        model_choice = model_choices[int(rng.choice(len(model_choices), p=model_probs))]
        seed_all(int(rng.integers(0, 2**31 - 1)))
        cuda_cleanup()
        samples, _, batch_meta = get_cifar_batch(
            batch_size=1,
            img_size=model_cfg.img_size,
            num_classes=model_cfg.num_classes,
            loader_seed=loader_seed_random,
            draw_index=int(draw_index),
            std_threshold=float(std_threshold),
            max_epochs_to_search=int(max_epochs_to_search),
        )
        model_cfg_draw = replace(model_cfg, seed=int(rng.integers(0, 2**31 - 1)))
        model = build_vit(model_cfg_draw, use_derf=model_choice != "preln")
        try:
            if model_choice != "preln":
                set_all_derf_alpha_(model, float(model_choice))
            inverse = estimate_J_points_hutchinson(
                model,
                samples,
                l0_list=chosen_layers,
                j_num_draws=int(j_num_draws),
                j_normalize_by="Y",
            )
        finally:
            del model
            cuda_cleanup()
        for layer_abs, value in inverse.items():
            points.append({
                "sample": int(draw_index),
                "layer": int(layer_abs),
                "value": float(value),
                "model_kind": "preln" if model_choice == "preln" else "derf",
                "alpha": None if model_choice == "preln" else float(model_choice),
                "chosen_layers": chosen_layers,
                "batch_meta": batch_meta,
            })

    out = {
        "points": points,
        "alphas": np.asarray(alphas, dtype=float),
        "depth": int(depth),
        "layers_per_sample": int(layers_per_sample),
        "config": {
            "n_samples": int(n_samples),
            "preln_weight": float(preln_weight),
            "batch_seed": None,
            "scatter_seed": None,
            "loader_seed_random": int(loader_seed_random),
            "std_threshold": float(std_threshold),
            "max_epochs_to_search": int(max_epochs_to_search),
            "j_num_draws": int(j_num_draws),
            "randomized_sampling": True,
        },
    }
    if save_results:
        saved_path, out = _save_bundle_pickle(
            out,
            save_root=save_root,
            folder_name=_folder_name_with_postfix(
                f"random_inverse_depth{depth}_layers{int(layers_per_sample)}",
                result_postfix,
            ),
            filename="results.pkl",
            rewrite=bool(rewrite),
            merge_kind="points",
        )
        out["saved_path"] = str(saved_path)
    return out


def _swarm_x_positions(n_points: int, center: float, width: float, rng: np.random.Generator):
    if n_points <= 0:
        return np.empty(0, dtype=float)
    if n_points == 1:
        return np.asarray([center], dtype=float)
    offsets = rng.uniform(-0.5 * width, 0.5 * width, size=int(n_points))
    order = np.argsort(offsets)
    return center + offsets[order]


def prepare_fit_and_scatter_plot_data(
    fit_bundle,
    inverse_scatter_bundle,
    direct_scatter_bundle,
):
    return {
        "fit_bundle": _maybe_load_bundle(fit_bundle),
        "inverse_scatter_bundle": _maybe_load_bundle(inverse_scatter_bundle),
        "direct_scatter_bundle": _maybe_load_bundle(direct_scatter_bundle),
    }


def plot_fit_and_scatter_figure(
    fit_scatter_plot_data,
    style_cfg: FinalThreePanelStyleConfig,
    panel_col_gap=0.18,
    panel_row_gap=0.22,
    lower_row_to_colorbar_gap=0.18,
    tick_fs=None,
    label_fs=None,
    alpha_legend_fs=None,
    title_fs=None,
    percentile_annotation_fs=None,
    percentile_annotation_alpha=0.45,
    scatter_point_alpha=0.55,
):
    if not isinstance(fit_scatter_plot_data, dict) or "fit_bundle" not in fit_scatter_plot_data:
        raise TypeError(
            "plot_fit_and_scatter_figure expects the result of "
            "prepare_fit_and_scatter_plot_data(...)."
        )
    fit_bundle = fit_scatter_plot_data["fit_bundle"]
    inverse_scatter_bundle = fit_scatter_plot_data["inverse_scatter_bundle"]
    direct_scatter_bundle = fit_scatter_plot_data["direct_scatter_bundle"]
    alphas = np.asarray(direct_scatter_bundle["alphas"], dtype=float)
    colors = _make_alpha_colors(alphas)
    alpha_to_color = {float(a): colors[i] for i, a in enumerate(alphas)}
    sizes = _resolve_plot_text_sizes(
        style_cfg,
        tick_fs=tick_fs,
        label_fs=label_fs,
        alpha_legend_fs=alpha_legend_fs,
        title_fs=title_fs,
        annotation_fs=percentile_annotation_fs,
    )
    fig = plt.figure(figsize=(13.8, 13.8))
    gs = fig.add_gridspec(
        5,
        3,
        height_ratios=[1.0, panel_row_gap, 1.0, lower_row_to_colorbar_gap, 0.18],
        width_ratios=[1.0, panel_col_gap, 1.0],
        hspace=0.0,
        wspace=0.0,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 2])
    ax_c = fig.add_subplot(gs[2, 0])
    ax_d = fig.add_subplot(gs[2, 2])
    cax = fig.add_subplot(gs[4, :])

    rng_swarm = np.random.default_rng(0)
    inv_mape_pct = 100.0 * np.asarray(fit_bundle["inverse_mape"], dtype=float)
    dir_mape_pct = 100.0 * np.asarray(fit_bundle["direct_mape"], dtype=float)
    x_inv = _swarm_x_positions(inv_mape_pct.size, center=0.0, width=0.42, rng=rng_swarm)
    x_dir = _swarm_x_positions(dir_mape_pct.size, center=0.0, width=0.42, rng=rng_swarm)
    ax_a.scatter(x_inv, inv_mape_pct, color="#4c78a8", s=28, alpha=0.8, edgecolors="none")
    ax_a.set_title(r"(a) Per-sample $\mathcal{J}^{\, B, b}$-fit MAPE", fontsize=sizes["title_fs"])
    ax_a.set_ylabel(r"MAPE, $\%$", fontsize=sizes["label_fs"])
    ax_a.set_xticks([])
    prettify_axes(ax_a)
    ax_a.tick_params(labelsize=sizes["tick_fs"])
    for percentile, label in [(80, r"$80\%$ of data points"), (90, r"$90\%$ of data points")]:
        y_val = float(np.nanpercentile(inv_mape_pct, percentile))
        ax_a.axhline(y_val, color="#4c78a8", ls="--", lw=1.2, alpha=0.75)
        ax_a.text(
            0.98,
            y_val,
            label,
            ha="right",
            va="bottom",
            transform=ax_a.get_yaxis_transform(),
            fontsize=sizes["annotation_fs"],
            alpha=float(percentile_annotation_alpha),
        )

    ax_b.scatter(x_dir, dir_mape_pct, color="#2ca02c", s=28, alpha=0.8, edgecolors="none")
    ax_b.set_title(r"(b) Per-sample $\mathcal{J}^{\, b, 0}$-fit MAPE", fontsize=sizes["title_fs"])
    ax_b.set_ylabel(r"MAPE, $\%$", fontsize=sizes["label_fs"])
    ax_b.set_xticks([])
    prettify_axes(ax_b)
    ax_b.tick_params(labelsize=sizes["tick_fs"])
    for percentile, label in [(80, r"$80\%$ of data points"), (90, r"$90\%$ of data points")]:
        y_val = float(np.nanpercentile(dir_mape_pct, percentile))
        ax_b.axhline(y_val, color="#2ca02c", ls="--", lw=1.2, alpha=0.75)
        ax_b.text(
            0.98,
            y_val,
            label,
            ha="right",
            va="bottom",
            transform=ax_b.get_yaxis_transform(),
            fontsize=sizes["annotation_fs"],
            alpha=float(percentile_annotation_alpha),
        )

    inverse_points = inverse_scatter_bundle.get("points", [])
    for point in inverse_points:
        color = "black" if point["model_kind"] == "preln" else alpha_to_color[float(point["alpha"])]
        ax_c.scatter(point["layer"], point["value"], color=color, s=18, alpha=float(scatter_point_alpha))
    ax_c.set_title(r"(c) $\mathcal{J}^{\, B, b}$ across samples", fontsize=sizes["title_fs"])
    ax_c.set_xlabel(r"$b$", fontsize=sizes["label_fs"])
    ax_c.set_ylabel(r"$\mathcal{J}^{\, B, b}$", fontsize=sizes["label_fs"])
    ax_c.set_yscale("log")
    prettify_log_axis(ax_c, "y")
    prettify_axes(ax_c)
    ax_c.tick_params(labelsize=sizes["tick_fs"])

    direct_points = direct_scatter_bundle.get("points", [])
    for point in direct_points:
        color = "black" if point["model_kind"] == "preln" else alpha_to_color[float(point["alpha"])]
        ax_d.scatter(point["layer"], point["value"], color=color, s=18, alpha=float(scatter_point_alpha))
    ax_d.set_title(r"(d) $\mathcal{J}^{\, b, 0}$ across samples", fontsize=sizes["title_fs"])
    ax_d.set_xlabel(r"$b$", fontsize=sizes["label_fs"])
    ax_d.set_ylabel(r"$\mathcal{J}^{\, b, 0}$", fontsize=sizes["label_fs"])
    ax_d.set_yscale("log")
    prettify_log_axis(ax_d, "y")
    prettify_axes(ax_d)
    ax_d.tick_params(labelsize=sizes["tick_fs"])

    center_shrink_axis(cax, width_scale=0.82, height_scale=0.75)
    if style_cfg.colorbar_pad:
        pos = cax.get_position()
        cax.set_position([pos.x0, pos.y0 - style_cfg.colorbar_pad, pos.width, pos.height])
    _draw_alpha_preln_legend_like_equangular(cax, alphas, colors, sizes["alpha_legend_fs"])
    _save_notebook_figure(fig, "fit_and_scatter_figure.pdf")
    plt.show()
    return fig


def plot_panel_d_reduced_figure(
    panel_d_bundle,
    alphas,
    style_cfg: FinalThreePanelStyleConfig,
    panel_gap=0.2,
    tick_fs=None,
    label_fs=None,
    alpha_legend_fs=None,
    title_fs=None,
):
    # Reduced version of plot_asymptotic_heatmap_figure: only the direct-APJN panels.
    alphas = np.asarray(alphas, dtype=float)
    colors = _make_alpha_colors(alphas)
    sizes = _resolve_plot_text_sizes(
        style_cfg,
        tick_fs=tick_fs,
        label_fs=label_fs,
        alpha_legend_fs=alpha_legend_fs,
        title_fs=title_fs,
    )
    empirical = panel_d_bundle.get("empirical") or {}
    empirical_derf = empirical.get("derf") or {}
    empirical_preln = empirical.get("preln")
    asym_ls = get_asym_linestyle(
        mode=style_cfg.asym_style_mode,
        dash_len=style_cfg.asym_dash_len,
        gap_len=style_cfg.asym_gap_len,
        dot_len=style_cfg.asym_dot_len,
    )

    fig = plt.figure(figsize=(11.6, 4.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.18], width_ratios=[1.0, panel_gap, 1.0], hspace=0.12, wspace=0.0)
    ax_derf = fig.add_subplot(gs[0, 0])
    ax_pre = fig.add_subplot(gs[0, 2])
    cax = fig.add_subplot(gs[1, :])

    for i, d in enumerate(panel_d_bundle["derf"]):
        ax_derf.plot(panel_d_bundle["l_arr"], d["logJ"]**2, color=colors[i], lw=style_cfg.line_width)
        ax_derf.plot(panel_d_bundle["l_arr"], d["logJ_asym"]**2, color="0.35", lw=style_cfg.asym_line_width, ls=asym_ls)
    if empirical_derf:
        for i, a in enumerate(alphas):
            dire = empirical_derf.get(float(a))
            if dire is None:
                continue
            ax_derf.scatter(dire["layers"], dire["logJ_sq"], color=colors[i], s=20, alpha=0.65)
    ax_derf.set_title(r"(a) Derf direct APJN", fontsize=sizes["title_fs"])
    ax_derf.set_xlabel(r"$b$", fontsize=sizes["label_fs"])
    ax_derf.set_ylabel(r"$(\log \mathcal{J}^{\, b, 0})^2$", fontsize=sizes["label_fs"])
    prettify_axes(ax_derf)
    ax_derf.tick_params(labelsize=sizes["tick_fs"])

    ax_pre.plot(panel_d_bundle["l_arr"][1:], panel_d_bundle["preln"]["J"][1:], color="black", lw=style_cfg.line_width)
    ax_pre.plot(panel_d_bundle["l_arr"][1:], panel_d_bundle["preln"]["J_asym"][1:], color="0.35", lw=style_cfg.asym_line_width, ls=asym_ls)
    if empirical_preln is not None:
        ax_pre.scatter(empirical_preln["layers"], empirical_preln["J"], color="black", s=20, alpha=0.65)
    ax_pre.set_title(r"(b) pre-LN direct APJN", fontsize=sizes["title_fs"])
    ax_pre.set_xlabel(r"$b$", fontsize=sizes["label_fs"])
    ax_pre.set_ylabel(r"$\mathcal{J}^{\, b, 0}$", fontsize=sizes["label_fs"])
    ax_pre.set_xscale("log")
    ax_pre.set_yscale("log")
    prettify_log_axis(ax_pre, "x")
    prettify_log_axis(ax_pre, "y")
    prettify_axes(ax_pre)
    ax_pre.tick_params(labelsize=sizes["tick_fs"])

    center_shrink_axis(cax, width_scale=0.62, height_scale=0.75)
    if style_cfg.colorbar_pad:
        pos = cax.get_position()
        cax.set_position([pos.x0, pos.y0 - style_cfg.colorbar_pad, pos.width, pos.height])
    add_alpha_colorbar_horizontal_single(cax, alphas, colors, label=r"$\alpha$ (Derf)", cb_fs=sizes["alpha_legend_fs"])
    rect = mpatches.Rectangle((1.02, 0.05), 0.08, 0.9, transform=cax.transAxes, facecolor="black", edgecolor="black", clip_on=False)
    cax.add_patch(rect)
    cax.text(1.06, -0.2, "pre-LN", transform=cax.transAxes, ha="center", va="top", fontsize=sizes["alpha_legend_fs"])
    _save_notebook_figure(fig, "panel_d_reduced_figure.pdf")
    plt.show()
    return fig
