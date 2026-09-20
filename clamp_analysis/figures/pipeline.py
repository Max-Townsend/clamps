"""Export the six numbered thesis figures and the model parameter table."""

import matplotlib.pyplot as plt

from clamp_analysis import paths

from . import anova, chapter, panels, supplement


def save(figure, stem):
    for extension in ("png", "svg", "pdf"):
        figure.savefig(
            paths.FIGURE_DIR / f"{stem}.{extension}", dpi=220, bbox_inches="tight", pad_inches=0.25
        )
    plt.close(figure)
    print(f"Saved {stem} (PNG, SVG, PDF)", flush=True)


def run():
    paths.ensure_output_dirs()
    for stem, figure in chapter.build_chapter_figures().items():
        save(figure, stem)
    anova.configure_style()
    descriptives = anova.build_plot_data()
    descriptives.to_csv(paths.TABLE_DIR / "thesis_anova_cell_descriptives.csv", index=False)
    save(anova.make_figure(descriptives), "figure_S4_1")
    save(supplement.build_figure(), "figure_S4_2")
    panels.load_mean_timeseries_parameter_table().to_csv(
        paths.TABLE_DIR / "model_parameters.csv", index=False
    )
