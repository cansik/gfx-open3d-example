from pathlib import Path
from typing import Tuple, Callable

import imageio as iio
import numpy as np
import pygfx as gfx
import trimesh
from open3d import geometry, io
from open3d.visualization import rendering
from trimesh import Trimesh
from wgpu.gui.auto import WgpuCanvas

LoaderResult = Tuple[gfx.Geometry, gfx.Material]


def load_from_trimesh(path: str) -> LoaderResult:
    # load geometry
    t_mesh: Trimesh = trimesh.load(path)  # doesn't work with: process=False, maintain_order=True
    uvs = np.array(t_mesh.visual.uv, np.float32)
    t_geo = gfx.geometry_from_trimesh(t_mesh)

    # load texture
    texture_path = Path(path).with_suffix(".png")
    im = iio.v3.imread(texture_path).astype("float32") / 255
    tex = gfx.Texture(im, dim=2)
    material = gfx.MeshBasicMaterial(map=tex)

    return t_geo, material


def load_from_open3d(path: str) -> LoaderResult:
    o3d_model: rendering.TriangleMeshModel = io.read_triangle_model(path)
    o3d_mesh_info: rendering.TriangleMeshModel.MeshInfo = o3d_model.meshes[0]
    o3d_mesh: geometry.TriangleMesh = o3d_mesh_info.mesh
    o3d_material = o3d_model.materials[o3d_mesh_info.material_idx]

    # o3d_temp = io.read_triangle_mesh(path, enable_post_processing=True)

    # visualize o3d model
    # visualization.draw([o3d_model], "Test")

    # create geometry
    triangle_uvs = np.array(o3d_mesh.triangle_uvs, dtype=np.float32)
    triangles = np.array(o3d_mesh.triangles, dtype=np.uint32)

    vertex_normals = np.array(o3d_mesh.vertex_normals, dtype=np.float32)
    vertex_colors = np.array(o3d_mesh.vertex_colors, dtype=np.float32)
    vertices = np.array(o3d_mesh.vertices, dtype=np.float32)

    triangle_uvs_wgpu = (triangle_uvs * np.array([1, -1]) + np.array([0, 1])).astype(np.float32)  # uv.y = 1 - uv.y

    gfx_geometry = gfx.Geometry(indices=triangles, positions=vertices,
                                normals=vertex_normals, texcoords=triangle_uvs_wgpu)

    # create material
    gfx_material = gfx.MeshBasicMaterial()
    gfx_material.flat_shading = False

    texture = np.array(o3d_material.albedo_img).astype("float32") / 255

    tex = gfx.Texture(texture, dim=2)
    gfx_material.map_interpolation = "linear"
    gfx_material.side = "FRONT"
    gfx_material.map = tex

    return gfx_geometry, gfx_material


def create_mesh(path: str, loader: Callable[[str], LoaderResult]) -> gfx.Mesh:
    gfx_geometry, gfx_material = loader(path)
    gfx_mesh = gfx.Mesh(gfx_geometry, gfx_material)

    # upscale mesh
    up_scale_ratio = 100
    gfx_mesh.local.scale = np.array(np.full((3,), up_scale_ratio))

    return gfx_mesh


def main():
    object_path = "data/cube.obj"

    # load different objs
    o3d_mesh = create_mesh(object_path, load_from_open3d)
    o3d_mesh.world.x += 200

    t_mesh = create_mesh(object_path, load_from_trimesh)
    t_mesh.world.x -= 200

    print(t_mesh.geometry.texcoords.data.shape)
    print(o3d_mesh.geometry.texcoords.data.shape)

    # setup 3d scene
    canvas = WgpuCanvas()
    renderer = gfx.renderers.WgpuRenderer(canvas)
    scene = gfx.Scene()

    scene.add(gfx.AxesHelper(size=250))
    scene.add(gfx.AmbientLight(gfx.Color("#ffffff"), 0.5))

    # add objects
    scene.add(o3d_mesh, t_mesh)

    camera = gfx.PerspectiveCamera(70, 16 / 9)
    camera.show_object(scene)
    controller = gfx.OrbitController(camera, register_events=renderer)

    disp = gfx.Display(canvas=canvas)
    disp.stats = True
    disp.show(scene)


if __name__ == "__main__":
    main()
