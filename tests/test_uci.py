import unittest
from unittest.mock import patch

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

from uci.uci import uci, _calc_location_coef, _calc_venables, _estimate_max_venables, _normalize_distribution, _calc_euclidean_dist_matrix

class TestUCIModule(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Sample polygons for GeoDataFrame
        polygons = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),  # (0, 0)
            Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),  # (1, 0)
            Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),  # (2, 0)
            Polygon([(0, 1), (1, 1), (1, 2), (0, 2)]),  # (0, 1)
            Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),  # (1, 1)
            Polygon([(2, 1), (3, 1), (3, 2), (2, 2)]),  # (2, 1)
            Polygon([(0, 2), (1, 2), (1, 3), (0, 3)]),  # (0, 2)
            Polygon([(1, 2), (2, 2), (2, 3), (1, 3)]),  # (1, 2)
            Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])   # (2, 2)
        ]
        cls.gdf = gpd.GeoDataFrame(geometry=polygons, data={'activities': [20, 100, 20, 100, 500, 0, 20, 0, 20]}, crs="EPSG:4326")
        cls.gdf_poly = cls.gdf.copy()
        cls.gdf_poly.loc[4, 'activities'] = 0  # Set central spatial unit to zero (more polycentric activity distribution)
        cls.gdf_extrem_poly = cls.gdf.copy()
        cls.gdf_extrem_poly.loc[[1, 3, 4, 5, 7], 'activities'] = 0  # Keep only values at the corner
        cls.gdf_convex = cls.gdf.drop([7])  # Slightly U-shaped geometry (dropped spatial unit had no activity)
        cls.gdf_convex_poly = cls.gdf.drop([4, 7])  # U-shaped geometry (more polycentric activity distribution)
        cls.gdf_isolated = cls.gdf.drop([4, 5, 7])  # Isolated spatial unit


    def test_location_coef(self):
        data = np.array([0.1, 0.4, 0.3, 0.2])

        lc = _calc_location_coef(data)

        self.assertEqual(lc, 0.2)


    def test_normalize_distribution(self):
        vec = np.array([100, 200, 300])

        norm_vec = _normalize_distribution(vec)

        expected_norm = np.array([0.16666667, 0.33333333, 0.5])
        np.testing.assert_almost_equal(norm_vec, expected_norm, decimal=5)


    def test_venables(self):
        b = np.array([0.5, 0.25, 0.25])
        distance = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])

        v = _calc_venables(b, distance)

        self.assertEqual(v, 0.875)


    def test_estimate_max_venables(self):
        distance = _calc_euclidean_dist_matrix(self.gdf)

        max_ven = _estimate_max_venables(self.gdf, distance, bootstrap_border=False)

        self.assertAlmostEqual(max_ven, 1.60809, places=5)


    def test_uci_with_invalid_var_name(self):
        # Ensure that passing an invalid column raises an assertion error
        with self.assertRaises(AssertionError):
            uci(self.gdf, 'invalid_column')


    def test_uci(self):
        result = uci(self.gdf, 'activities')

        # Ensure the output has all expected fields
        self.assertIn('UCI', result)
        self.assertIn('location_coef', result)
        self.assertIn('proximity_index', result)
        self.assertIn('spatial_separation', result)
        self.assertIn('spatial_separation_max', result)

        # Ensure the UCI value is non-negative and bounded
        self.assertGreaterEqual(result['UCI'], 0)
        self.assertLessEqual(result['UCI'], 1)


    def test_uci_euclidean(self):
        result = uci(self.gdf, 'activities')
        result_poly = uci(self.gdf_poly, 'activities')
        result_convex = uci(self.gdf_convex, 'activities')
        result_convex_poly = uci(self.gdf_convex_poly, 'activities')

        self.assertAlmostEqual(result['UCI'], 0.24276, places=5)
        self.assertLess(result_poly['proximity_index'], result['proximity_index'])  # Set activity of central spatial unit to zero should result in lower proximity index
        self.assertEqual(result_convex['spatial_separation'], result['spatial_separation'])  # Dropped spatial unit had no value and should not influence the spatial separation index
        self.assertGreater(result_convex_poly['spatial_separation'], result_convex['spatial_separation'])  # Removing central spatial unit should result in higher spatial separation index


    def test_uci_spatial_link(self):
        result = uci(self.gdf, 'activities', euclidean=False)
        result_poly = uci(self.gdf_poly, 'activities', euclidean=False)
        result_convex = uci(self.gdf_convex, 'activities', euclidean=False)
        result_convex_poly = uci(self.gdf_convex_poly, 'activities', euclidean=False)

        self.assertAlmostEqual(result['UCI'], 0.249821, places=5)
        self.assertLess(result_poly['proximity_index'], result['proximity_index'])  # Set activity of central spatial unit to zero should result in lower proximity index
        self.assertGreater(result_convex['spatial_separation'], result['spatial_separation'])  # More convex shape should increase distances and thus spatial separation index
        self.assertGreater(result_convex_poly['spatial_separation'], result_convex['spatial_separation'])  # Removing central spatial unit should result in higher spatial separation index


    def test_uci_spatial_link_edge_case(self):
        """
        Handle edge case of spatially extremely separated distributions, where the actual
        spatial separation index might be larger than the heuristic or bootstrapped maximum.
        """
        result = uci(self.gdf_extrem_poly, 'activities', euclidean=False)

        self.assertEqual(result['UCI'], 0)


    def test_uci_spatial_link_isolated(self):
        """
        Handle edge case of isolated spatial units for which no spatial neighbor link exists.
        """
        result = uci(self.gdf_isolated, 'activities', euclidean=False)

        np.testing.assert_equal(result['UCI'], np.nan)
        np.testing.assert_equal(result['proximity_index'], np.nan)
        self.assertAlmostEqual(result['location_coef'], 0.3809523, places=5)


    def test_bootstrap_border(self):

        result = uci(self.gdf, 'activities')
        result_bootstrap = uci(self.gdf, 'activities', bootstrap_border=True)

        self.assertGreater(result_bootstrap['spatial_separation_max'], result['spatial_separation_max'])  # Bootstrapping should in most cases increase the maximum estimate of the spatial separation index
