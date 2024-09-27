# Urban Centrality Index (UCI)

Calculate Urban Centrality Index (UCI) as described in [Pereira et al. (2013)](https://www.doi.org/10.1111/gean.12002).

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
