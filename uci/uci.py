import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix


def uci(
    gdf: gpd.GeoDataFrame,
    var_name: str,
    euclidean: bool = True,
    bootstrap_border: bool = False
) -> pd.Series:
    """
    Calculate the Urban Centrality Index (UCI) as described in Pereira et al. (2013) \doi{10.1111/gean.12002}.
    The UCI quantifies the degree of spatial organization of a city or region on a continuous scale from 0 to 1,
    where values closer to 0 indicate a more polycentric pattern and values closer to 1 indicate a more monocentric urban form.

    Implementation based on the original R code by Pereira et al. available at: https://github.com/ipeaGIT/uci

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        A GeoDataFrame containing the study area as polygon geometries.

    var_name : str
        The name of the column in `gdf` that contains the number of activities, opportunities, resources,
        or services to be considered when calculating urban centrality levels. `NaN` values are treated as `0`.

    euclidean : bool, optional
        If `True` (default), calculates Euclidean distances between spatial units.
        If `False`, calculates distances based on spatial neighbor links.

        It is recommended to use "euclidean=False" for areas with concave shapes (e.g., bays), as it can provide
        more accurate UCI estimates despite being computationally more expensive.

    bootstrap_border : bool, optional
        If 'False' (default), a heuristic approach proposed by Pereira et al. (2013) is used to estimate the maximum spatial
        separation index, which assumes that it takes the maximum value if activities are homogeneously distributed along the
        boundary of the study area.
        If `True`, the spatial separation index is calculated for different value distributions along the boundary using a
        bootstrapping approach and the maximum value is selected.

    Returns
    -------
    pd.Series
        A Series containing the following values:
        - 'UCI': The calculated Urban Centrality Index.
        - 'location_coef': The calculated location coefficient.
        - 'spatial_separation_ratio': The proximity index based on spatial separation.
        - 'spatial_separation': The observed spatial separation index (Venables).
        - 'spatial_separation_max': The maximum spatial separation index found.

    """
    # Check inputs
    _assert_var_name(gdf, var_name)

    # Change projection to UTM (Web Mercator)
    gdf = gdf.to_crs(epsg=3857)

    # Calculate Location Coefficient
    var = gdf[var_name].fillna(0).values
    var_norm = _normalize_distribution(var)
    LC = _calc_location_coef(var_norm)

    # Calculate distance matrix
    if euclidean:
        distance = _calc_euclidean_dist_matrix(gdf)
    else:
        distance = _calc_spatial_link_dist_matrix(gdf)

    # Calculate spatial separation index (Venables)
    V = _calc_venables(var_norm, distance)
    V_max = _estimate_max_venables(gdf, distance, bootstrap_border)

    # Calculate UCI
    proximity_index = 1 - (V / max(V, V_max))
    UCI = LC * proximity_index

    return pd.Series({
        'UCI': UCI,
        'location_coef': LC,
        'proximity_index': proximity_index,
        'spatial_separation': V,
        'spatial_separation_max': V_max
    })


def _assert_var_name(gdf: gpd.GeoDataFrame, var_name: str) -> None:
    """
    Check if the variable name is in the GeoDataFrame.
    """
    assert isinstance(gdf, gpd.GeoDataFrame), "Input must be a GeoDataFrame."
    assert var_name in gdf.columns, f"Variable '{var_name}' must be in the GeoDataFrame."


def _calc_location_coef(x: np.ndarray) -> float:
    """
    Calculate Location Coefficient (LC).
    """
    return np.sum(np.abs(x - (1 / len(x)))) / 2


def _calc_venables(b: np.ndarray, distance: np.ndarray) -> float:
    """
    Calculate the Venables spatial separation index.
    """
    v = np.dot(np.dot(b.T, distance), b)
    return v


def _estimate_max_venables(
    gdf: gpd.GeoDataFrame,
    distance: np.ndarray,
    bootstrap_border: bool
) -> float:
    """
    Estimate maximum Venables spatial separation index using heuristic approach.
    """
    # Identify border cells
    boundary = gdf.unary_union.boundary
    boundary_mask = gdf.intersects(boundary)
    sf_border = gdf[boundary_mask]
    n_border = len(sf_border)
    distance_border = distance[np.ix_(boundary_mask, boundary_mask)]

    # Determine max venables based on bootstrap or heuristic
    if bootstrap_border:
        v = np.max([
            _calc_venables(_simulate_border_values(n_border, n), distance_border)
            for n in range(2, n_border, (n_border // 50) + 1)
        ])
    else:
        some_border_values = np.full(n_border, 1 / n_border)
        v = _calc_venables(some_border_values, distance_border)

    return v


def _calc_euclidean_dist_matrix(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """
    Calculate the Euclidean distance matrix.
    """
    centroid = gdf.geometry.centroid
    coords = list(zip(centroid.x, centroid.y))
    distance = distance_matrix(coords, coords)
    distance += _calc_self_distance(gdf)

    return distance


def _calc_self_distance(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """
    Calculate the self-distance (within a spatial unit) based on the unit's area.
    """
    areas = gdf.area.values
    self_distance = np.diag(np.sqrt(areas / np.pi))

    return self_distance


def _create_spatial_link_graph(gdf: gpd.GeoDataFrame) -> nx.Graph:
    """
    Create a graph of connected spatial units, where nodes represent the centroids of units
    and edges represent the distance between connected units.
    """
    G = nx.Graph()

    # Add nodes
    for idx in gdf.index:
        G.add_node(idx)

    # Add edges between intersecting geometries
    for i, geom1 in gdf.geometry.items():
        for j, geom2 in gdf.geometry.items():
            if i != j and geom1.intersects(geom2):
                distance = geom1.centroid.distance(geom2.centroid)
                G.add_edge(i, j, weight=distance)

    return G


def _calc_spatial_link_dist_matrix(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """
    Calculate a distance matrix using the shortest path along the geometries of a GeoPandas GeoDataFrame, following the connections between them (i.e., taking into account the actual shapes and paths)
    """
    G = _create_spatial_link_graph(gdf)
    shortest_paths = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
    dis_matrix = pd.DataFrame(shortest_paths).sort_index(axis=1).sort_index(axis=0).to_numpy()
    dis_matrix += _calc_self_distance(gdf)

    return dis_matrix


def _normalize_distribution(vec: np.ndarray) -> np.ndarray:
    """
    Normalize the distribution of a variable.
    """
    return vec / np.sum(vec)


def _simulate_border_values(length: int, n: int) -> np.ndarray:
    """
    Simulate equal value distributions along a random subset of the border.
    """
    np.random.seed(0)
    positions = np.random.choice(range(length), size=n)
    values = _create_weight_vector(length, positions)
    return values


def _create_weight_vector(length: int, positions: np.ndarray) -> np.ndarray:
    """
    Create a normalized vector of specified length, where values at positions sum up to 1.
    """
    b = np.zeros(length)
    b[positions] = 1
    return b / np.sum(b)
