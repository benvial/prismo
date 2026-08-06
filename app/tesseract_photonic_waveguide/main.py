"""Main entrypoint for Tesseract Photonic Waveguide pipeline."""

import typer

app = typer.Typer(name="tesseract_photonic_waveguide")


@app.command()
def run() -> None:
    """Run the Tesseract Photonic Waveguide pipeline."""
    # Chain your Tesseracts here. For example, once you have built a component
    # (`make new <mytess>` then `make build <mytess>`):
    #
    #     from tesseract_core import Tesseract
    #
    #     with Tesseract.from_image("tesseract_photonic_waveguide_<mytess>") as tess:
    #         result = tess.apply({"example_input": ...})
    #     typer.echo(result)
    #
    # See app/chain.ipynb for an interactive version.
    typer.echo("Running Tesseract Photonic Waveguide pipeline...")


def entrypoint() -> None:
    """CLI entrypoint for the application."""
    app()


if __name__ == "__main__":
    entrypoint()
