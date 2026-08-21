"""
Create publication-ready copies of the eval plots by editing the original
PNGs directly: remove the background grid and shorten the title
(e.g. "h - combined" -> "h"). Originals are left untouched.

Outputs (per parameter): eval_combined_<param>_publication.png and .pdf
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager
import os

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "combined")
PARAMS = ["Omega_m", "Omega_b", "h", "n_s", "sigma_8"]

# Original plots: figsize (6,5) @ dpi 150, default matplotlib style,
# grid color #b0b0b0, title fontsize 12pt (= 25 px at 150 dpi)
TITLE_FONT_PX = 25

font_path = font_manager.findfont("DejaVu Sans")
title_font = ImageFont.truetype(font_path, TITLE_FONT_PX)


def find_axes_bbox(arr):
    """Locate the axes spines (long runs of dark pixels)."""
    dark = (arr[..., :3] < 100).all(axis=2)
    h, w = dark.shape
    col_counts = dark.sum(axis=0)
    row_counts = dark.sum(axis=1)
    spine_cols = np.where(col_counts > 0.4 * h)[0]
    spine_rows = np.where(row_counts > 0.4 * w)[0]
    return spine_cols.min(), spine_cols.max(), spine_rows.min(), spine_rows.max()


def grid_gray_mask(arr):
    """Pixels that look like the light-gray grid (incl. anti-aliased edges)."""
    r = arr[..., 0].astype(int)
    g = arr[..., 1].astype(int)
    b = arr[..., 2].astype(int)
    neutral = (np.abs(r - g) <= 12) & (np.abs(g - b) <= 12) & (np.abs(r - b) <= 12)
    return neutral & (r >= 150) & (r <= 252)


def remove_grid(arr, left, right, top, bottom):
    """Whiten grid-gray pixels, but only along full-span grid rows/columns."""
    inner_t, inner_b = top + 3, bottom - 2
    inner_l, inner_r = left + 3, right - 2
    mask = grid_gray_mask(arr)

    inner_h = inner_b - inner_t
    inner_w = inner_r - inner_l

    # Threshold of 25%: grid lines partially hidden behind data still qualify,
    # while shorter gray features (e.g. legend border) do not span that much.
    for x in range(inner_l, inner_r):
        if mask[inner_t:inner_b, x].sum() > 0.25 * inner_h:
            for xx in range(x - 2, x + 3):
                ys = np.where(mask[inner_t:inner_b, xx])[0] + inner_t
                arr[ys, xx] = 255

    for y in range(inner_t, inner_b):
        if mask[y, inner_l:inner_r].sum() > 0.25 * inner_w:
            for yy in range(y - 2, y + 3):
                xs = np.where(mask[yy, inner_l:inner_r])[0] + inner_l
                arr[yy, xs] = 255
    return arr


def replace_title(img, arr, left, right, top, new_title):
    """White out the old title strip and draw the shortened title, centered."""
    strip_bottom = top - 4
    arr[0:strip_bottom, :] = 255
    img_out = Image.fromarray(arr)
    draw = ImageDraw.Draw(img_out)
    tb = draw.textbbox((0, 0), new_title, font=title_font)
    text_w = tb[2] - tb[0]
    text_h = tb[3] - tb[1]
    cx = (left + right) / 2
    x = int(cx - text_w / 2 - tb[0])
    y = int((strip_bottom - text_h) / 2 - tb[1])
    draw.text((x, y), new_title, font=title_font, fill=(0, 0, 0))
    return img_out


for pname in PARAMS:
    src = os.path.join(DIR, f"eval_combined_{pname}.png")
    img = Image.open(src).convert("RGB")
    arr = np.array(img)

    left, right, top, bottom = find_axes_bbox(arr)
    arr = remove_grid(arr, left, right, top, bottom)
    img_out = replace_title(img, arr, left, right, top, pname)

    out_png = os.path.join(DIR, f"eval_combined_{pname}_publication.png")
    out_pdf = os.path.join(DIR, f"eval_combined_{pname}_publication.pdf")
    img_out.save(out_png, dpi=(150, 150))
    img_out.save(out_pdf, resolution=150)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")

print("Done.")
