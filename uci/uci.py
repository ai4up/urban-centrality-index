import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix


def uci(gdf, var_name, dist_type="euclidean", bootstrap_border=False):
    """
    Calculate the Urban Centrality Index (UCI) as described in Pereira et al. (2013) \doi{10.1111/gean.12002}. 
    The UCI quantifies the degree of spatial organization of a city or region on a continuous scale from 0 to 1, 
    where values closer to 0 indicate a more polycentric pattern and values closer to 1 indicate a more monocentric urban form.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        A GeoDataFrame containing the study area as polygon geometries.
        
    var_name : str
        The name of the column in `gdf` that contains the number of activities, opportunities, resources, 
        or services to be considered when calculating urban centrality levels. `NaN` values are treated as `0`.

    dist_type : str, optional
        A string indicating the type of distance calculations to use. 
        Acceptable values are:
        - "euclidean" (default): calculates Euclidean distances.
        - "spatial_link": calculates distances based on spatial neighbor links.
        
        It is recommended to use "spatial_link" for areas with concave shapes (e.g., bays), as it can provide 
        more accurate UCI estimates despite being computationally more expensive.

    bootstrap_border : bool, optional
        If `True`, the function uses a bootstrap approach to simulate random distributions of activities along the border 
        of the study area to find the maximum value of the Venables spatial separation index. Defaults to `False`, 
        where a heuristic approach is used that assumes maximum spatial separation occurs with evenly distributed 
        activities along the border.

    Returns
    -------
    pandas.Series
        A Series containing the following values:
        - 'UCI': The calculated Urban Centrality Index.
        - 'location_coef': The calculated location coefficient.
        - 'spatial_separation_ratio': The proximity index based on spatial separation.
        - 'spatial_separation': The observed spatial separation index (Venables).
        - 'spatial_separation_max': The maximum spatial separation index found.
    
    Raises
    ------
    ValueError
        If `dist_type` is not one of the acceptable values ("euclidean" or "spatial_link").
    """
    # Check inputs
    _assert_var_name(gdf, var_name)

    # Change projection to UTM (Web Mercator)
    gdf = gdf.to_crs(epsg=3857)

    # Location Coefficient
    var = gdf[var_name].fillna(0).values
    var_norm = _normalize_distribution(var)
    LC = _location_coef(var_norm)

    # Calculate distance matrix based on dist_type
    if dist_type == 'euclidean':
        distance = _get_euclidean_dist_matrix(gdf)
    elif dist_type == 'spatial_link':
        distance = _get_spatial_link_dist_matrix(gdf)
    else:
        raise ValueError("dist_type must be either 'spatial_link' or 'euclidean'")

    # Spatial separation index (Venables)
    v_observed = venables(var_norm, distance)

    # Identify border cells
    boundary = gdf.unary_union.boundary
    boundary_mask = gdf.intersects(boundary)
    sf_border = gdf[boundary_mask]
    n_border = len(sf_border)
    distance_border = distance[np.ix_(boundary_mask, boundary_mask)]

    # Determine max venables based on bootstrap or heuristic
    if bootstrap_border:
        # Try different border values (at most ~50 times) and find the max venables
        max_venables = np.max([venables(_simulate_border_values(n_border, n), distance_border) for n in range(2, n_border, (n_border // 50) + 1)])
    else:
        some_border_values = np.full(n_border, 1 / n_border)
        max_venables = venables(some_border_values, distance_border)

    # Calculate UCI
    proximity_index = 1 - (v_observed / max_venables)
    UCI = LC * proximity_index

    return pd.Series({
        'UCI': UCI,
        'location_coef': LC,
        'spatial_separation_ratio': proximity_index,
        'spatial_separation': v_observed,
        'spatial_separation_max': max_venables
    })


def _assert_var_name(gdf, var_name):
    """
    Check if the variable name is in the GeoDataFrame.
    """
    assert isinstance(gdf, gpd.GeoDataFrame), "Input must be a GeoDataFrame."
    assert var_name in gdf.columns, f"Variable '{var_name}' must be in the GeoDataFrame."


def _location_coef(x):
    """
    Calculate Location Coefficient (LC).
    """
    return np.sum(np.abs(x - (1 / len(x)))) / 2


def venables(b, distance):
    """
    Calculate the Venables spatial separation index.
    """
    v = np.dot(np.dot(b.T, distance), b)
    return v


def _get_euclidean_dist_matrix(gdf):
    """
    Calculate the Euclidean distance matrix.
    """
    centroid = gdf.geometry.centroid
    coords = list(zip(centroid.x, centroid.y))
    distance = distance_matrix(coords, coords)
    distance += _calc_self_distance(gdf)

    return distance
    

def _calc_self_distance(gdf):
    areas = gdf.area.values
    self_distance = np.diag(np.sqrt(areas / np.pi))
    
    return self_distance


def _create_spatial_link_graph(gdf):
    # Create empty graph
    G = nx.Graph()

    # Add nodes
    for idx in gdf.index:
        G.add_node(idx)

    # Add edges between on intersecting geometries
    for i, geom1 in gdf.geometry.items():
        for j, geom2 in gdf.geometry.items():
            if i != j and geom1.intersects(geom2):
                distance = geom1.centroid.distance(geom2.centroid)
                G.add_edge(i, j, weight=distance)

    return G


def _get_spatial_link_dist_matrix(gdf):
    """
    Calculate a distance matrix using the shortest path along the geometries of a GeoPandas GeoDataFrame, following the connections between them (i.e., taking into account the actual shapes and paths)
    """
    G = _create_spatial_link_graph(gdf)
    shortest_paths = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
    dis_matrix = pd.DataFrame(shortest_paths).to_numpy()
    dis_matrix += _calc_self_distance(gdf)

    return dis_matrix


def _normalize_distribution(vec):
    """
    Normalize the distribution of a variable.
    """
    return vec / np.sum(vec)


def _simulate_border_values(length, n):
    """
    Simulate equal value distributions along random subset of the border.
    """
    positions = np.random.choice(range(length), size=n)
    values = _create_weight_vector(length, positions)
    return values


def _create_weight_vector(length, positions):
    """
    Create a normalized vector of specified length, where values at positions sum up to 1.
    """
    b = np.zeros(length)
    b[positions] = 1
    return b / np.sum(b)
