"""
Combine the five per-parameter scatter plots (publication PDFs) into a single
high-resolution figure for a two-column ICML paper.

Layout: 3 rows x 2 columns, with Omega_b centered alone in the last row.
Each panel gets a subcaption label (a)-(e).

Output: publicationcharts/scatter_all_combined.pdf

This script is fully standalone and does not touch flow.py or any originals.
"""
import os

import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

CHART_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "publicationcharts")
OUTPUT_PATH = os.path.join(CHART_DIR, "scatter_all_combined.pdf")

# (filename, subcaption label) in display order.
# Labels rendered in Computer Modern math italics to match the paper.
PANELS = [
    ("eval_combined_Omega_m_publication.pdf", r"$(a)\ \Omega_m$"),
    ("eval_combined_sigma_8_publication.pdf", r"$(b)\ \sigma_8$"),
    ("eval_combined_n_s_publication.pdf", r"$(c)\ n_s$"),
    ("eval_combined_h_publication.pdf", r"$(d)\ h$"),
    ("eval_combined_Omega_b_publication.pdf", r"$(e)\ \Omega_b$"),
]

RENDER_ZOOM = 4.0  # 72 dpi * 4 = 288 dpi rendering of each PDF page


def load_pdf_as_image(path):
    """Render the first page of a PDF to an RGB numpy array."""
    doc = fitz.open(path)
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(RENDER_ZOOM, RENDER_ZOOM))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    doc.close()
    return img[..., :3]


def main():
    plt.rcParams["mathtext.fontset"] = "cm"  # Computer Modern, matches LaTeX

    images = [load_pdf_as_image(os.path.join(CHART_DIR, fname)) for fname, _ in PANELS]

    fig = plt.figure(figsize=(10, 12))
    # 3x4 grid so the lone plot in the last row can span the middle two cells
    gs = GridSpec(3, 4, figure=fig, hspace=0.16, wspace=0.05)

    slots = [
        gs[0, 0:2],  # Omega_m
        gs[0, 2:4],  # sigma_8
        gs[1, 0:2],  # n_s
        gs[1, 2:4],  # h
        gs[2, 1:3],  # Omega_b, centered
    ]

    for (fname, label), img, slot in zip(PANELS, images, slots):
        ax = fig.add_subplot(slot)
        ax.imshow(img)
        ax.axis("off")
        # Subcaption centered below the panel, LaTeX-style
        ax.text(0.5, -0.03, label, transform=ax.transAxes,
                fontsize=18, va="top", ha="center")

    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
