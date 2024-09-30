# Urban Centrality Index (UCI)

Calculate Urban Centrality Index (UCI) as described in [Pereira et al. (2013)](https://www.doi.org/10.1111/gean.12002).

The UCI quantifies the spatial clustering of a city or region based on the distribution of a chosen dimension, such as employment, population, or other points of interest. The index is measured on a continuous scale from 0 to 1, where values closer to 0 indicate a more polycentric pattern, and values near 1 suggest a more monocentric urban structure.

The Python implementation is based on the R package [uci](https://github.com/ipeaGIT/uci) by Pereira et al.


## Install
```bash
pip install git+https://github.com/ai4up/urban-centrality-index@v0.1.0
```

## Usage
```Python
>>> import uci

>>> uci.uci(gdf, 'column_of_interest')
UCI                            0.089
location_coef                  0.492
proximity_index                0.181
spatial_separation             146.196
spatial_separation_max         179.015
dtype: float64
```

## Development
Build from source using [poetry](https://python-poetry.org/):
```
poetry build
pip install dist/urban_centrality_index-*.whl
```


## Citation

 The original R package [uci](https://github.com/ipeaGIT/uci) is developed by a team at the Institute for Applied Economic Research (Ipea), Brazil. If you use this package in research publications, please cite it as:

* Pereira, R. H. M., Nadalin, V., Monasterio, L., & Albuquerque, P. H. (2013). **Urban centrality: a simple index**. *Geographical analysis*, 45(1), 77-89. [https://www.doi.org/10.1111/gean.12002](https://www.doi.org/10.1111/gean.12002)


BibTeX:
```
@article{pereira2013urbancentrality,
  title = {Urban {{Centrality}}: {{A Simple Index}}},
  author = {Pereira, Rafael H. M. and Nadalin, Vanessa and Monasterio, Leonardo and Albuquerque, Pedro H. M.},
  year = {2013},
  journal = {Geographical Analysis},
  volume = {45},
  number = {1},
  pages = {77--89},
  issn = {1538-4632},
  doi = {10.1111/gean.12002}
}
```