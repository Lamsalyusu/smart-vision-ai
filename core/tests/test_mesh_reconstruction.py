import unittest
import numpy as np

from core.domain.spatial.mesh_reconstruction import (
    PointCloudNormalEstimator,
    VolumetricTSDFBuilder,
    MarchingCubesReconstructor,
    TriangleMesh,
    MeshPostProcessor,
    MeshExporter,
)


class TestMeshReconstructionPipeline(unittest.TestCase):
    def setUp(self):
        # Generate 3D point cloud of a sphere: radius 1.0, centered at origin
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 10)
        x = (np.outer(np.cos(u), np.sin(v))).flatten()
        y = (np.outer(np.sin(u), np.sin(v))).flatten()
        z = (np.outer(np.ones(np.size(u)), np.cos(v))).flatten()
        self.sphere_points = np.stack([x, y, z], axis=-1)

    def test_point_cloud_normal_estimation(self):
        estimator = PointCloudNormalEstimator(k_neighbors=6)
        normals = estimator.estimate_normals(self.sphere_points)

        self.assertEqual(normals.shape, self.sphere_points.shape)
        # Verify unit norm of computed normals
        norms = np.linalg.norm(normals, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_volumetric_tsdf_builder(self):
        estimator = PointCloudNormalEstimator(k_neighbors=6)
        normals = estimator.estimate_normals(self.sphere_points)

        builder = VolumetricTSDFBuilder(voxel_size=0.15, truncation_distance=0.3)
        tsdf, meta = builder.build_tsdf(self.sphere_points, normals)

        self.assertEqual(len(tsdf.shape), 3)
        self.assertEqual(tsdf.shape, meta.dimensions)
        self.assertLessEqual(np.max(tsdf), 0.3)
        self.assertGreaterEqual(np.min(tsdf), -0.3)

    def test_marching_cubes_isosurface_extraction(self):
        estimator = PointCloudNormalEstimator(k_neighbors=6)
        normals = estimator.estimate_normals(self.sphere_points)

        builder = VolumetricTSDFBuilder(voxel_size=0.15, truncation_distance=0.3)
        tsdf, meta = builder.build_tsdf(self.sphere_points, normals)

        reconstructor = MarchingCubesReconstructor(isovalue=0.0)
        mesh = reconstructor.reconstruct(tsdf, meta)

        self.assertIsInstance(mesh, TriangleMesh)
        self.assertGreater(len(mesh.vertices), 0)
        self.assertGreater(len(mesh.faces), 0)
        self.assertEqual(len(mesh.normals), len(mesh.vertices))

    def test_laplacian_smoothing_and_area(self):
        estimator = PointCloudNormalEstimator(k_neighbors=6)
        normals = estimator.estimate_normals(self.sphere_points)
        builder = VolumetricTSDFBuilder(voxel_size=0.15, truncation_distance=0.3)
        tsdf, meta = builder.build_tsdf(self.sphere_points, normals)
        reconstructor = MarchingCubesReconstructor(isovalue=0.0)
        mesh = reconstructor.reconstruct(tsdf, meta)

        smoothed = MeshPostProcessor.laplacian_smooth(mesh, iterations=2, lambda_factor=0.3)
        self.assertEqual(len(smoothed.vertices), len(mesh.vertices))
        self.assertEqual(len(smoothed.faces), len(mesh.faces))

        area = MeshPostProcessor.compute_surface_area(smoothed)
        self.assertGreater(area, 0.0)

    def test_mesh_export_obj_ply_and_threejs(self):
        estimator = PointCloudNormalEstimator(k_neighbors=6)
        normals = estimator.estimate_normals(self.sphere_points)
        builder = VolumetricTSDFBuilder(voxel_size=0.15, truncation_distance=0.3)
        tsdf, meta = builder.build_tsdf(self.sphere_points, normals)
        reconstructor = MarchingCubesReconstructor(isovalue=0.0)
        mesh = reconstructor.reconstruct(tsdf, meta)

        # Test OBJ export
        obj_str = MeshExporter.to_obj(mesh, object_name="TestSphere")
        self.assertIn("o TestSphere", obj_str)
        self.assertIn("v ", obj_str)
        self.assertIn("f ", obj_str)

        # Test PLY export
        ply_str = MeshExporter.to_ply(mesh)
        self.assertIn("ply\nformat ascii 1.0", ply_str)
        self.assertIn(f"element vertex {len(mesh.vertices)}", ply_str)
        self.assertIn(f"element face {len(mesh.faces)}", ply_str)

        # Test Three.js JSON format
        three_json = MeshExporter.to_threejs_json(mesh)
        self.assertEqual(three_json["metadata"]["type"], "BufferGeometry")
        self.assertEqual(three_json["statistics"]["vertex_count"], len(mesh.vertices))
        self.assertEqual(three_json["statistics"]["face_count"], len(mesh.faces))
        self.assertIn("position", three_json["data"]["attributes"])
        self.assertIn("normal", three_json["data"]["attributes"])


class TestMeshReconstructionApi(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import httpx
        from core.infrastructure.api.main import app
        self.transport = httpx.ASGITransport(app=app)
        self.client = httpx.AsyncClient(transport=self.transport, base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_reconstruct_endpoint_canonical_chair(self):
        res = await self.client.post(
            "/mesh/reconstruct",
            json={
                "object_class": "chair",
                "voxel_size": 0.1,
                "smoothing_iterations": 1,
                "output_format": "threejs",
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertIn("threejs_geometry", data)
        self.assertGreater(data["statistics"]["vertex_count"], 0)

    async def test_reconstruct_endpoint_with_custom_points(self):
        pts = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
        res = await self.client.post(
            "/mesh/reconstruct",
            json={
                "points": pts,
                "voxel_size": 0.2,
                "smoothing_iterations": 1,
                "output_format": "obj",
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertIn("obj_data", data)
        self.assertIn("v ", data["obj_data"])


if __name__ == "__main__":
    unittest.main()
