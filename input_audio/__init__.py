import pathlib

__version__ = pathlib.Path(__file__).parent.joinpath("VERSION").read_text().strip()
