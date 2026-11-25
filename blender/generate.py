"""Render a synthetic orange in Blender.

Run headless:
    blender -b -P blender/generate.py -- --out data/synthetic/step2_1_fresh.jpg

All randomness lives outside Blender (src/make_scene_params.py, from Step 2.4).
This script only turns a dict of parameters into an image, so a render is fully
determined by its parameters and the label never depends on the image.
"""

import argparse
import colorsys
import os
import sys
import time

import bpy
from mathutils import Euler, Vector

IMAGE_SIZE = 224
RENDER_SAMPLES = 32  # 16 left visible grain on the shadow edge in the benchmark
FRUIT_RADIUS = 1.0

# One fresh orange; Step 2.2 replaces these with sampled values per grade
DEFAULT_PARAMS = {
    "hue": 0.075,             # HSV hue of the peel, 0.075 is orange
    "saturation": 0.9,
    "value": 0.9,
    "roughness": 0.4,         # waxy peel: some shine, not mirror-like
    "pit_scale": 18.0,        # Voronoi cells per unit; above ~40 the pits shrink below a pixel at 224 px
    "bump_strength": 0.3,
    "squash": 0.9,            # z-scale; oranges are slightly flatter than a sphere
    "lumpiness": 0.03,        # strength of the low-frequency shape noise
    "rotation": (0.5, 0.3, 0.0),  # radians, tilted so the stem is visible but off-centre
    "light_energy": 400.0,    # watts
    "light_location": (2.0, -2.0, 6.0),
    "fill": 0.7,              # fraction of the image width the fruit covers
    "stem_mix": 0.5,          # 0 = olive green, 1 = dark brown
}

# Stem colour range (sRGB), from stems seen in the real train split: olive-grey to near-black brown
STEM_OLIVE = (0.30, 0.32, 0.12)
STEM_BROWN = (0.20, 0.12, 0.05)


def srgb_to_linear(c):
    # Blender colours are linear; the hue we sample is meant as the sRGB hue seen in the image
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def look_at(obj, target):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def camera_height(fill, lens_mm, sensor_mm=36.0):
    # Distance at which the fruit's diameter covers `fill` of the frame width
    visible_width = 2 * FRUIT_RADIUS / fill
    return visible_width * lens_mm / sensor_mm


def setup_render(scene):
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = RENDER_SAMPLES
    scene.eevee.use_raytracing = False  # not needed for one diffuse-ish object, and costs time
    scene.render.resolution_x = IMAGE_SIZE
    scene.render.resolution_y = IMAGE_SIZE
    scene.render.resolution_percentage = 100
    # Standard keeps the sampled hue close to the output hue; AgX would desaturate bright orange
    scene.view_settings.view_transform = "Standard"
    scene.render.image_settings.file_format = "JPEG"  # real images are JPEG (D7)
    scene.render.image_settings.quality = 95
    scene.render.image_settings.color_mode = "RGB"
    # Blender writes the render date into the file by default; drop all metadata so reruns match
    scene.render.use_stamp_date = False
    scene.render.use_stamp_time = False
    scene.render.use_stamp_render_time = False
    scene.render.use_stamp_frame = False
    scene.render.use_stamp_filename = False


def make_peel_material():
    mat = bpy.data.materials.new("peel")
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes["Principled BSDF"]

    coords = nodes.new("ShaderNodeTexCoord")
    voronoi = nodes.new("ShaderNodeTexVoronoi")
    voronoi.name = "pits"
    bump = nodes.new("ShaderNodeBump")
    bump.name = "pit_bump"
    # Height is Voronoi distance (0 to ~0.7); the 0.001 default made pits about 1000x too shallow to see
    bump.inputs["Distance"].default_value = 0.02
    # Object coordinates so the pits stay fixed on the peel when the fruit rotates
    links.new(coords.outputs["Object"], voronoi.inputs["Vector"])
    links.new(voronoi.outputs["Distance"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def make_stem_material():
    mat = bpy.data.materials.new("stem")
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.5  # some sheen so the raised edge catches the light

    mix = nodes.new("ShaderNodeMix")
    mix.name = "stem_colour"
    mix.data_type = "RGBA"
    mix.inputs["A"].default_value = (*[srgb_to_linear(c) for c in STEM_OLIVE], 1.0)
    mix.inputs["B"].default_value = (*[srgb_to_linear(c) for c in STEM_BROWN], 1.0)

    # Fine noise darkens patches and wrinkles the surface, so it reads as a dry scar, not a smooth hole
    coords = nodes.new("ShaderNodeTexCoord")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 80.0  # a few blotches across a ~6 px stem
    shade = nodes.new("ShaderNodeMix")
    shade.data_type = "RGBA"
    shade.blend_type = "MULTIPLY"
    shade.inputs["Factor"].default_value = 0.7
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Distance"].default_value = 0.01
    links.new(coords.outputs["Object"], noise.inputs["Vector"])
    links.new(mix.outputs["Result"], shade.inputs["A"])
    links.new(noise.outputs["Color"], shade.inputs["B"])
    links.new(shade.outputs["Result"], bsdf.inputs["Base Color"])
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def make_flat_material(name, rgb, roughness):
    mat = bpy.data.materials.new(name)
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def build_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    setup_render(scene)

    world = bpy.data.worlds.new("world")
    # Dim grey ambient so shadowed peel is dark but not black
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.3, 0.3, 0.3, 1.0)
    scene.world = world

    bpy.ops.mesh.primitive_plane_add(size=20)
    belt = bpy.context.object
    belt.name = "belt"
    belt.data.materials.append(make_flat_material("belt", (0.03, 0.03, 0.03), 0.8))

    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=FRUIT_RADIUS)
    fruit = bpy.context.object
    fruit.name = "fruit"
    bpy.ops.object.shade_smooth()
    fruit.modifiers.new("subdiv", "SUBSURF").levels = 2
    fruit.modifiers["subdiv"].render_levels = 2
    lumps = fruit.modifiers.new("lumps", "DISPLACE")
    lumps.texture = bpy.data.textures.new("lumps", "CLOUDS")
    lumps.texture.noise_scale = 0.6  # large blobs: shape irregularity, not peel texture
    fruit.data.materials.append(make_peel_material())

    # Stem or calyx at the +z pole on every grade, so a small dark mark is not a defect cue (D6)
    # Low dome rather than a flat disk: a raised scar catches light on its edge instead of reading as a hole.
    # Radius 0.045 gives about 4 to 8 px at 224 px, the size of stems in the real train images.
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=0.045,
                                         location=(0, 0, FRUIT_RADIUS + 0.005))
    stem = bpy.context.object
    stem.name = "stem"
    stem.scale.z = 0.8
    bpy.ops.object.shade_smooth()
    stem.data.materials.append(make_stem_material())
    stem.parent = fruit

    bpy.ops.object.light_add(type="AREA")
    light = bpy.context.object
    light.name = "light"
    light.data.size = 3.0

    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.data.lens = 50
    scene.camera = camera
    return scene


def rest_on_ground(fruit):
    # D3: after rotation, drop the fruit so its lowest point touches the belt (z = 0)
    fruit.location = (0, 0, 0)
    bpy.context.view_layer.update()
    evaluated = fruit.evaluated_get(bpy.context.evaluated_depsgraph_get())
    lowest = min((evaluated.matrix_world @ v.co).z for v in evaluated.data.vertices)
    fruit.location.z = -lowest


def apply_params(scene, params):
    fruit = bpy.data.objects["fruit"]
    fruit.scale = (1.0, 1.0, params["squash"])
    fruit.rotation_euler = Euler(params["rotation"])
    fruit.modifiers["lumps"].strength = params["lumpiness"]
    rest_on_ground(fruit)

    nodes = fruit.data.materials[0].node_tree.nodes
    rgb = colorsys.hsv_to_rgb(params["hue"], params["saturation"], params["value"])
    bsdf = nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*[srgb_to_linear(c) for c in rgb], 1.0)
    bsdf.inputs["Roughness"].default_value = params["roughness"]
    nodes["pits"].inputs["Scale"].default_value = params["pit_scale"]
    nodes["pit_bump"].inputs["Strength"].default_value = params["bump_strength"]

    stem_nodes = bpy.data.objects["stem"].data.materials[0].node_tree.nodes
    stem_nodes["stem_colour"].inputs["Factor"].default_value = params["stem_mix"]

    light = bpy.data.objects["light"]
    light.data.energy = params["light_energy"]
    light.location = params["light_location"]
    look_at(light, fruit.location)

    camera = scene.camera
    camera.location = (0, 0, fruit.location.z + camera_height(params["fill"], camera.data.lens))
    look_at(camera, fruit.location)


def render(scene, out_path):
    scene.render.filepath = out_path
    start = time.time()
    bpy.ops.render.render(write_still=True)
    return time.time() - start


def parse_args():
    # Blender passes its own arguments first; ours come after "--"
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/synthetic/step2_1_fresh.jpg")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    scene = build_scene()
    apply_params(scene, DEFAULT_PARAMS)
    # Absolute path: Blender would otherwise resolve relative paths against the .blend location
    seconds = render(scene, os.path.abspath(args.out))
    print(f"Rendered {args.out} in {seconds:.2f} s")


if __name__ == "__main__":
    main()
