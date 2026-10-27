"""Tiny PIL-based plotting helpers (no matplotlib dependency)."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/SFNS.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _mix(c0: Tuple[int, int, int], c1: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c0, c1))


def viridis(t: float) -> Tuple[int, int, int]:
    """Approximate viridis colormap."""
    stops = [
        (68, 1, 84),
        (59, 82, 139),
        (33, 145, 140),
        (94, 201, 98),
        (253, 231, 37),
    ]
    t = float(np.clip(t, 0.0, 1.0)) * (len(stops) - 1)
    i = min(int(t), len(stops) - 2)
    return _mix(stops[i], stops[i + 1], t - i)


class Plot:
    def __init__(
        self,
        width: int = 720,
        height: int = 480,
        margin: Tuple[int, int, int, int] = (85, 45, 45, 85),
        bg: str = "white",
    ):
        self.w, self.h = width, height
        self.left, self.right, self.top, self.bottom = margin
        self.img = Image.new("RGB", (width, height), bg)
        self.draw = ImageDraw.Draw(self.img)
        self.x0 = self.left
        self.x1 = width - self.right
        self.y0 = height - self.bottom
        self.y1 = self.top
        self.xlim = (0.0, 1.0)
        self.ylim = (0.0, 1.0)
        self.legend_items: List[Tuple[str, str]] = []
        self.f_title = _font(22)
        self.f_label = _font(17)
        self.f_tick = _font(14)
        self.f_small = _font(13)

    # -- coordinate mapping -------------------------------------------------
    def px(self, x: float, y: float) -> Tuple[float, float]:
        fx = (x - self.xlim[0]) / (self.xlim[1] - self.xlim[0])
        fy = (y - self.ylim[0]) / (self.ylim[1] - self.ylim[0])
        return self.x0 + fx * (self.x1 - self.x0), self.y0 - fy * (self.y0 - self.y1)

    def _rotated_text(self, text: str, font, fill: str):
        tmp = Image.new("RGBA", (400, 40), (255, 255, 255, 0))
        d = ImageDraw.Draw(tmp)
        bbox = d.textbbox((0, 0), text, font=font)
        d.text((2, 2), text, font=font, fill=fill)
        tmp = tmp.crop((0, 0, bbox[2] + 4, bbox[3] + 4))
        return tmp.rotate(90, expand=True)

    # -- axes ---------------------------------------------------------------
    def axes(
        self,
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
        xlabel: str = "",
        ylabel: str = "",
        title: str = "",
        n_xticks: int = 5,
        n_yticks: int = 5,
        xtick_fmt: str = "{:.2g}",
        ytick_fmt: str = "{:.2g}",
        draw_xticklabels: bool = True,
        draw_yticklabels: bool = True,
    ):
        self.xlim = xlim
        self.ylim = ylim
        self.draw.rectangle([self.x0, self.y1, self.x1, self.y0], outline="#333333", width=2)

        for i in range(n_xticks):
            v = xlim[0] + (xlim[1] - xlim[0]) * i / (n_xticks - 1)
            x, _ = self.px(v, ylim[0])
            self.draw.line([x, self.y0, x, self.y0 + 6], fill="#333333", width=1)
            if draw_xticklabels:
                self.draw.text((x, self.y0 + 10), xtick_fmt.format(v), font=self.f_tick,
                               fill="#333333", anchor="mt")
        for i in range(n_yticks):
            v = ylim[0] + (ylim[1] - ylim[0]) * i / (n_yticks - 1)
            _, y = self.px(xlim[0], v)
            self.draw.line([self.x0 - 6, y, self.x0, y], fill="#333333", width=1)
            if draw_yticklabels:
                self.draw.text((self.x0 - 10, y), ytick_fmt.format(v), font=self.f_tick,
                               fill="#333333", anchor="rm")

        if xlabel:
            self.draw.text(((self.x0 + self.x1) / 2, self.h - 26), xlabel,
                           font=self.f_label, fill="#222222", anchor="mm")
        if ylabel:
            tmp = self._rotated_text(ylabel, self.f_label, "#222222")
            self.img.paste(tmp, (10, int((self.y0 + self.y1) / 2 - tmp.height / 2)), tmp)
        if title:
            self.draw.text(((self.x0 + self.x1) / 2, 18), title,
                           font=self.f_title, fill="#111111", anchor="mm")

    # -- primitives ---------------------------------------------------------
    def line(self, xs, ys, color: str = "#1f77b4", width: int = 2, label: Optional[str] = None):
        pts = [self.px(float(x), float(y)) for x, y in zip(xs, ys)]
        if len(pts) >= 2:
            self.draw.line(pts, fill=color, width=width, joint="curve")
        if label:
            self.legend_items.append((label, color))

    def image(self, gray, vmin: Optional[float] = None, vmax: Optional[float] = None):
        arr = np.asarray(gray, dtype=float)
        vmin = float(np.nanmin(arr)) if vmin is None else vmin
        vmax = float(np.nanmax(arr)) if vmax is None else vmax
        g = np.clip((arr - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0)
        rgb = np.stack([g, g, g], axis=-1)
        im = Image.fromarray((rgb * 255).astype(np.uint8)).resize(
            (max(1, self.x1 - self.x0 - 2), max(1, self.y0 - self.y1 - 2))
        )
        self.img.paste(im, (self.x0 + 1, self.y1 + 1))

    def scatter(self, xs, ys, colors, r: float = 3.0):
        for x, y, c in zip(xs, ys, colors):
            px, py = self.px(float(x), float(y))
            self.draw.ellipse([px - r, py - r, px + r, py + r], fill=c, outline=None)

    def heatmap(
        self,
        Z: np.ndarray,
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        colorbar: bool = True,
    ):
        Z = np.asarray(Z, dtype=float)
        vmin = float(np.nanmin(Z)) if vmin is None else vmin
        vmax = float(np.nanmax(Z)) if vmax is None else vmax
        ny, nx = Z.shape
        for i in range(nx):
            for j in range(ny):
                xa = xlim[0] + (xlim[1] - xlim[0]) * i / nx
                xb = xlim[0] + (xlim[1] - xlim[0]) * (i + 1) / nx
                ya = ylim[0] + (ylim[1] - ylim[0]) * j / ny
                yb = ylim[0] + (ylim[1] - ylim[0]) * (j + 1) / ny
                p0 = self.px(xa, ya)
                p1 = self.px(xb, yb)
                val = (Z[j, i] - vmin) / max(vmax - vmin, 1e-9)
                self.draw.rectangle([p0[0], p1[1], p1[0], p0[1]], fill=viridis(val))
        if colorbar:
            cx0, cx1 = self.x1 + 12, self.x1 + 26
            for k in range(100):
                y = self.y0 - (self.y0 - self.y1) * k / 99
                y2 = self.y0 - (self.y0 - self.y1) * (k + 1) / 99
                self.draw.rectangle([cx0, min(y, y2), cx1, max(y, y2)], fill=viridis(k / 99))
            self.draw.text((cx1 + 4, self.y0), f"{vmin:.2g}", font=self.f_small,
                           fill="#333333", anchor="lm")
            self.draw.text((cx1 + 4, self.y1), f"{vmax:.2g}", font=self.f_small,
                           fill="#333333", anchor="lm")

    def gradient_bar(self, vmin: float, vmax: float, label: str = ""):
        x0 = self.x1 + 10
        x1 = x0 + 14
        for k in range(100):
            y = self.y0 - (self.y0 - self.y1) * k / 99
            y2 = self.y0 - (self.y0 - self.y1) * (k + 1) / 99
            self.draw.rectangle([x0, min(y, y2), x1, max(y, y2)], fill=viridis(k / 99))
        self.draw.text((x1 + 4, self.y0), f"{vmin:.2g}", font=self.f_small,
                       fill="#333333", anchor="lm")
        self.draw.text((x1 + 4, self.y1), f"{vmax:.2g}", font=self.f_small,
                       fill="#333333", anchor="lm")
        if label:
            self.draw.text((x0 + 7, self.y1 - 14), label, font=self.f_small,
                           fill="#333333", anchor="mm")

    def marker(self, x: float, y: float, color: str = "white", r: float = 4.0):
        px, py = self.px(x, y)
        self.draw.line([px - r, py - r, px + r, py + r], fill=color, width=2)
        self.draw.line([px - r, py + r, px + r, py - r], fill=color, width=2)

    def bar(self, labels: Sequence[str], values: Sequence[float], colors: Sequence[str]):
        n = len(values)
        vmax = max(values) * 1.15
        self.axes((0, n), (0, vmax), ylabel="loss", n_xticks=n, xtick_fmt="{:.0f}")
        for i, (lab, v, c) in enumerate(zip(labels, values, colors)):
            xa = i + 0.18
            xb = i + 0.82
            p0 = self.px(xa, 0)
            p1 = self.px(xb, v)
            self.draw.rectangle([p0[0], p1[1], p1[0], p0[1]], fill=c)
            self.draw.text(((p0[0] + p1[0]) / 2, p1[1] - 6), f"{v:.2f}",
                           font=self.f_small, fill="#222222", anchor="mb")
            self.draw.text(((p0[0] + p1[0]) / 2, self.y0 + 10), lab,
                           font=self.f_small, fill="#222222", anchor="mt")

    def bar_group(
        self,
        labels: Sequence[str],
        series: Sequence[Sequence[float]],
        colors: Sequence[str],
        series_labels: Sequence[str],
    ):
        n = len(labels)
        m = len(series)
        vmax = max(max(vals) for vals in series) * 1.18
        self.axes((0, n), (0, vmax), ylabel="loss", n_xticks=2,
                  draw_xticklabels=False)
        width = 0.78 / m
        for k, (vals, col, slab) in enumerate(zip(series, colors, series_labels)):
            for i, v in enumerate(vals):
                xa = i + 0.11 + k * width
                xb = xa + width * 0.88
                p0 = self.px(xa, 0)
                p1 = self.px(xb, v)
                self.draw.rectangle([p0[0], p1[1], p1[0], p0[1]], fill=col)
                if k == 0:
                    self.draw.text(((self.px(i + 0.5, 0)[0]), self.y0 + 10), labels[i],
                                   font=self.f_small, fill="#222222", anchor="mt")
            self.legend_items.append((slab, col))

    def legend(self, loc: str = "upper right"):
        if not self.legend_items:
            return
        pad = 6
        line_h = 18
        w = 10 + max(self.draw.textlength(l, font=self.f_small) for l, _ in self.legend_items) + 34
        h = line_h * len(self.legend_items) + 2 * pad
        if loc == "upper right":
            x0, y0 = self.x1 - w - 8, self.y1 + 8
        elif loc == "lower left":
            x0, y0 = self.x0 + 8, self.y0 - h - 8
        else:
            x0, y0 = self.x1 - w - 8, self.y1 + 8
        self.draw.rectangle([x0, y0, x0 + w, y0 + h], fill="white", outline="#999999", width=1)
        for k, (lab, col) in enumerate(self.legend_items):
            yy = y0 + pad + k * line_h + line_h / 2
            self.draw.line([x0 + 8, yy, x0 + 28, yy], fill=col, width=3)
            self.draw.text((x0 + 34, yy), lab, font=self.f_small, fill="#222222", anchor="lm")

    def save(self, path):
        self.img.save(path)
