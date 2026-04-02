This is a helper library to read COLMAP data into a format usable in rust.

It's a mostly literal translation of the [COLMAP helper script](https://github.com/colmap/colmap/blob/main/scripts/python/read_write_model.py). Only reading colmap data is ported.

The parsed results use glam to be a bit nicer to work with.
