"""matplotlib 中文字体设置。

服务器（如阿里云 DSW、Kaggle）上通常没有中文字体，图里的中文会变成方框，并刷出大量
"Glyph ... missing from font(s) DejaVu Sans" 警告。解决办法：装一个中文字体，例如
    apt-get install -y fonts-wqy-microhei      # 约 5 MB，装完后重启 notebook 内核
"""
from __future__ import annotations

import warnings

import matplotlib
from matplotlib import font_manager

CANDIDATES = [
    "Noto Sans CJK SC", "Noto Sans SC", "Source Han Sans SC", "WenQuanYi Micro Hei",
    "WenQuanYi Zen Hei", "SimHei", "Microsoft YaHei", "PingFang SC", "Heiti SC",
    "Arial Unicode MS",
]


def setup_cjk_font(verbose: bool = True) -> str | None:
    """找到一个可用的中文字体并设为默认；找不到就屏蔽缺字警告并给出安装提示。"""
    font_manager.fontManager.__init__()  # 重新扫描，刚装的字体也能找到
    available = {f.name for f in font_manager.fontManager.ttflist}
    found = next((c for c in CANDIDATES if c in available), None)
    if found is None:
        found = next((n for n in sorted(available)
                      if any(k in n for k in ("CJK", "Hei", "Song", "Kai", "WenQuanYi", "Han Sans"))), None)
    matplotlib.rcParams["axes.unicode_minus"] = False
    if found:
        matplotlib.rcParams["font.sans-serif"] = [found, "DejaVu Sans"]
        matplotlib.rcParams["font.family"] = "sans-serif"
        if verbose:
            print(f"中文字体：{found}")
    else:
        warnings.filterwarnings("ignore", message=r"Glyph \d+ .* missing from font")
        if verbose:
            print("没有找到中文字体，图中的中文会显示成方框（不影响计算结果）。\n"
                  "安装方法：apt-get install -y fonts-wqy-microhei，然后重启内核。")
    return found
