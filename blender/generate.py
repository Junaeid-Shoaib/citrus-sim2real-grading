# pyright: reportMissingModuleSource=false
"""Render a synthetic orange in Blender.

Run headless:
    blender -b -P blender/generate.py -- --out data/synthetic/step2_1_fresh.jpg

All randomness lives outside Blender (src/make_scene_params.py, from Step 2.4).
This script only turns a dict of parameters into an image, so a render is fully
determined by its parameters and the label never depends on the image.
"""

import argparse
import colorsys
import math
import os
import sys
import time

import bpy
from mathutils import Euler, Vector  # pyright: ignore[reportMissingImports]

IMAGE_SIZE = 224
RENDER_SAMPLES = 32  # 16 left visible grain on the shadow edge in the benchmark
FRUIT_RADIUS = 1.0

# One fresh orange. Real values come from src/make_scene_params.py (Step 2.4), one dict per image.
# There is no "grade" switch here: grade only decides which ranges the sampler draws from.
DEFAULT_PARAMS = {
    "hue": 0.075,             # HSV hue of the peel, 0.075 is orange
    "saturation": 0.9,
    "value": 0.9,
    "blotch_hue": 0.1,        # second peel colour mixed in large patches (yellow patches on green fruit)
    "blotch_amount": 0.2,     # 0 = uniform peel
    "pattern_offset": (0.0, 0.0, 0.0),  # shifts every procedural texture so each image gets new patterns
    "roughness": 0.4,         # waxy peel: some shine, not mirror-like
    # Global shift of defective fruit (spec 2.2); each starts at 0, so 0 means no shift
    "shift_dull": 0.0,        # fraction of saturation removed
    "shift_brown": 0.0,       # fraction of the way the hue moves toward brown-orange
    "shift_rough": 0.0,       # added to roughness (less shine)
    "defect_type": "none",    # none | spots | lesions | rot
    "spot_scale": 4.0,        # Voronoi cells per unit; about 3 gives ~5 visible spots, 5.5 gives ~30
    "spot_radius": 0.03,      # spot radius in fruit-radius units (about 2 px at 224 px)
    # Lesion or rot patches: (degrees from the view direction, azimuth degrees, radius); up to 3, radius at most MAX_PATCH_RADIUS
    "patches": [],
    "mould_amount": 0.0,      # rot only: how much of the patch is covered in mould
    "mould_mix": 0.0,         # 0 = all-white mould, 1 = grey-green spore centres inside white margins
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

# Defect colours (sRGB), matched by eye to the real defective train images
SPOT_COLOUR = (0.10, 0.07, 0.04)          # blackspot: near-black pits
LESION_COLOUR = (0.45, 0.38, 0.28)        # canker: grey-brown raised lesion
HALO_COLOUR = (0.75, 0.68, 0.35)          # canker: faint yellow ring around each lesion
# Rot and mould colours start from real rotten train images (10x10 px averages), adjusted so renders match
ROT_DARK = (0.26, 0.15, 0.03)             # dark brown patch
ROT_OLIVE = (0.45, 0.33, 0.15)            # olive-brown patch (more saturated than measured: renders came out grey)
MOULD_WHITE = (0.90, 0.89, 0.85)          # mycelium margin
MOULD_BLUE_GREEN = (0.40, 0.48, 0.42)     # muted grey-green spore centre (darker than measured: top light brightens it)
BROWN_HUE = 0.07                          # global shift pulls the peel hue toward this

# Caps on the global shift: beyond these a greenish base turned beige, which would make
# "beige means defective" a shortcut and undo D5. Checked by rendering greenish bases at the caps.
MAX_SHIFT_BROWN = 0.15
MAX_SHIFT_DULL = 0.2

DEFECT_TYPES = ["spots", "lesions", "rot"]
MAX_PATCHES = 3  # fixed number of patch slots, so the shader layout never changes

# Patch edge shape per defect type: lesions are crisp with a halo; rot is soft, irregular,
# with a water-soaked rim. "outer" is where the halo or rim ends, as a multiple of the radius.
PATCH_SHAPE = {
    "lesions": {"wobble": 0.1, "core_soft": 0.8, "outer": 1.4},
    "rot": {"wobble": 0.25, "core_soft": 0.55, "outer": 1.15},
}
MAX_PATCH_RADIUS = 0.7   # chord length on the unit sphere, about 41 degrees: a local patch, not the whole fruit
UNUSED_RADIUS = 0.01
# An unused slot has its centre at the fruit's centre, so every surface point is 1.0 from it.
# It can only show if its outer edge plus the worst-case edge wobble reaches 1.0.
assert max(s["outer"] for s in PATCH_SHAPE.values()) * UNUSED_RADIUS \
    + max(s["wobble"] for s in PATCH_SHAPE.values()) / 2 < 0.5


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


def linear_rgba(rgb):
    return (*[srgb_to_linear(c) for c in rgb], 1.0)


def connect(links, value, socket):
    # Inputs are either a constant or the output socket of another node
    if isinstance(value, bpy.types.NodeSocket):
        links.new(value, socket)
    else:
        socket.default_value = value


def math_op(nodes, links, operation, a, b=0.0):
    node = nodes.new("ShaderNodeMath")
    node.operation = operation
    connect(links, a, node.inputs[0])
    connect(links, b, node.inputs[1])
    return node.outputs[0]


def mix_colour(nodes, links, a, b, factor):
    node = nodes.new("ShaderNodeMix")
    node.data_type = "RGBA"
    connect(links, factor, node.inputs["Factor"])
    connect(links, a, node.inputs["A"])
    connect(links, b, node.inputs["B"])
    return node.outputs["Result"]


def map_range(nodes, links, value, name):
    # Smoothstep from From Min to From Max; apply_params sets the edges per image
    node = nodes.new("ShaderNodeMapRange")
    node.name = name
    node.interpolation_type = "SMOOTHSTEP"
    connect(links, value, node.inputs["Value"])
    return node.outputs["Result"]


def named(nodes, node_type, name):
    node = nodes.new(node_type)
    node.name = name
    return node


def make_peel_material():
    """Peel shader with every grade and defect type built in.

    The layout is fixed: images differ only in input values. Changing the layout makes EEVEE
    recompile the shader (about 6 s), so defect types are switched on and off with 0/1 factors.
    """
    mat = bpy.data.materials.new("peel")
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes["Principled BSDF"]

    # Object coordinates so patterns stay fixed on the peel when the fruit rotates
    coords = nodes.new("ShaderNodeTexCoord").outputs["Object"]
    offset = named(nodes, "ShaderNodeCombineXYZ", "pattern_offset")
    shifted = nodes.new("ShaderNodeVectorMath")
    shifted.operation = "ADD"
    links.new(coords, shifted.inputs[0])
    links.new(offset.outputs[0], shifted.inputs[1])
    shifted = shifted.outputs[0]
    # Direction from the fruit centre; patches and the camera cap are defined on the unit sphere
    direction = nodes.new("ShaderNodeVectorMath")
    direction.operation = "NORMALIZE"
    links.new(coords, direction.inputs[0])
    direction = direction.outputs[0]

    # Base peel colour, with large soft blotches of a second colour
    blotch_noise = nodes.new("ShaderNodeTexNoise")
    blotch_noise.inputs["Scale"].default_value = 2.5
    links.new(shifted, blotch_noise.inputs["Vector"])
    blotch = map_range(nodes, links, blotch_noise.outputs["Fac"], "blotch_edges")
    blotch.node.inputs["From Min"].default_value = 0.45
    blotch.node.inputs["From Max"].default_value = 0.6
    blotch = math_op(nodes, links, "MULTIPLY", blotch, named(nodes, "ShaderNodeValue", "blotch_amount").outputs[0])
    colour = mix_colour(nodes, links, named(nodes, "ShaderNodeRGB", "base_colour").outputs[0],
                        named(nodes, "ShaderNodeRGB", "blotch_colour").outputs[0], blotch)

    spots_on = named(nodes, "ShaderNodeValue", "spots_on").outputs[0]
    lesions_on = named(nodes, "ShaderNodeValue", "lesions_on").outputs[0]
    rot_on = named(nodes, "ShaderNodeValue", "rot_on").outputs[0]

    # Spots: thresholded Voronoi, masked to the camera-facing cap (within 60 degrees, cos 60 = 0.5)
    spot_tex = named(nodes, "ShaderNodeTexVoronoi", "spot_cells")
    links.new(shifted, spot_tex.inputs["Vector"])
    spots = map_range(nodes, links, spot_tex.outputs["Distance"], "spot_edges")
    facing = nodes.new("ShaderNodeVectorMath")
    facing.operation = "DOT_PRODUCT"
    links.new(direction, facing.inputs[0])
    links.new(named(nodes, "ShaderNodeCombineXYZ", "camera_direction").outputs[0], facing.inputs[1])
    cap = math_op(nodes, links, "GREATER_THAN", facing.outputs["Value"], 0.5)
    spots = math_op(nodes, links, "MULTIPLY", math_op(nodes, links, "MULTIPLY", spots, cap), spots_on)
    colour = mix_colour(nodes, links, colour, linear_rgba(SPOT_COLOUR), spots)

    # Patches for lesions and rot: distance from each centre, roughened by noise so edges are irregular
    edge_noise = nodes.new("ShaderNodeTexNoise")
    edge_noise.inputs["Scale"].default_value = 6.0
    edge_noise.inputs["Detail"].default_value = 0.0  # one octave: smooth irregular lobes; finer octaves scalloped the edge at 224 px
    links.new(shifted, edge_noise.inputs["Vector"])
    wobble = math_op(nodes, links, "MULTIPLY", math_op(nodes, links, "SUBTRACT", edge_noise.outputs["Fac"], 0.5),
                     named(nodes, "ShaderNodeValue", "edge_wobble").outputs[0])
    core, outer = 0.0, 0.0
    for i in range(MAX_PATCHES):
        distance = nodes.new("ShaderNodeVectorMath")
        distance.operation = "DISTANCE"
        links.new(direction, distance.inputs[0])
        links.new(named(nodes, "ShaderNodeCombineXYZ", f"patch{i}_centre").outputs[0], distance.inputs[1])
        d = math_op(nodes, links, "ADD", distance.outputs["Value"], wobble)
        core = math_op(nodes, links, "MAXIMUM", core, map_range(nodes, links, d, f"patch{i}_core"))
        outer = math_op(nodes, links, "MAXIMUM", outer, map_range(nodes, links, d, f"patch{i}_outer"))
    halo = math_op(nodes, links, "SUBTRACT", outer, core)  # ring just outside the core
    halo.node.use_clamp = True

    # Canker: yellow halo then grey-brown core
    colour = mix_colour(nodes, links, colour, linear_rgba(HALO_COLOUR),
                        math_op(nodes, links, "MULTIPLY", halo, math_op(nodes, links, "MULTIPLY", lesions_on, 0.6)))
    lesion = math_op(nodes, links, "MULTIPLY", core, lesions_on)
    colour = mix_colour(nodes, links, colour, linear_rgba(LESION_COLOUR), lesion)

    # Rot: dark-brown to olive-brown patch (mottled by the edge noise), ringed by a darker water-soaked rim
    rot = math_op(nodes, links, "MULTIPLY", core, rot_on)
    rot_colour = mix_colour(nodes, links, linear_rgba(ROT_DARK), linear_rgba(ROT_OLIVE), edge_noise.outputs["Fac"])
    colour = mix_colour(nodes, links, colour, rot_colour, rot)
    rim = math_op(nodes, links, "MULTIPLY", halo, rot_on)
    darken = nodes.new("ShaderNodeMix")
    darken.data_type = "RGBA"
    darken.blend_type = "MULTIPLY"
    links.new(rim, darken.inputs["Factor"])
    links.new(colour, darken.inputs["A"])
    darken.inputs["B"].default_value = (0.5, 0.5, 0.5, 1.0)
    colour = darken.outputs["Result"]

    # Mould grows out from the patch centre: a continuous felt, white at its margin and grey-green
    # toward the middle. The field is the patch core (1 inside, falling to 0 at the edge) plus noise
    # that only breaks up the boundaries; noise alone gave a camouflage pattern of separate blobs.
    mould_noise = nodes.new("ShaderNodeTexNoise")
    mould_noise.inputs["Scale"].default_value = 8.0
    links.new(shifted, mould_noise.inputs["Vector"])
    mould_field = math_op(nodes, links, "ADD", core,
                          math_op(nodes, links, "MULTIPLY",
                                  math_op(nodes, links, "SUBTRACT", mould_noise.outputs["Fac"], 0.5), 0.5))
    mould = math_op(nodes, links, "MULTIPLY", map_range(nodes, links, mould_field, "mould_edges"), rot_on)
    spores = map_range(nodes, links, mould_field, "mould_centre_edges")
    spores = math_op(nodes, links, "MULTIPLY", math_op(nodes, links, "MULTIPLY", spores, mould),
                     named(nodes, "ShaderNodeValue", "mould_mix").outputs[0])
    colour = mix_colour(nodes, links, colour, linear_rgba(MOULD_WHITE), mould)
    colour = mix_colour(nodes, links, colour, linear_rgba(MOULD_BLUE_GREEN), spores)
    links.new(colour, bsdf.inputs["Base Color"])

    # Rot and mould are matte, the wet rim is glossy; the rest of the peel keeps its sampled roughness
    roughness = map_range(nodes, links, math_op(nodes, links, "MAXIMUM", rot, mould), "roughness")
    roughness.node.interpolation_type = "LINEAR"
    roughness.node.inputs["To Max"].default_value = 0.9
    roughness = math_op(nodes, links, "SUBTRACT", roughness, math_op(nodes, links, "MULTIPLY", rim, 0.4))
    roughness.node.use_clamp = True
    links.new(roughness, bsdf.inputs["Roughness"])

    # Height: oil-gland pits, plus raised lesions and sunken spots and rot
    pits = named(nodes, "ShaderNodeTexVoronoi", "pits")
    links.new(coords, pits.inputs["Vector"])
    height = math_op(nodes, links, "ADD", pits.outputs["Distance"], math_op(nodes, links, "MULTIPLY", lesion, 0.5))
    height = math_op(nodes, links, "SUBTRACT", height, math_op(nodes, links, "MULTIPLY", rot, 1.0))
    height = math_op(nodes, links, "SUBTRACT", height, math_op(nodes, links, "MULTIPLY", rim, 0.3))
    height = math_op(nodes, links, "SUBTRACT", height, math_op(nodes, links, "MULTIPLY", spots, 0.5))
    bump = named(nodes, "ShaderNodeBump", "pit_bump")
    # Height is mostly Voronoi distance (0 to ~0.7); the 0.001 default made pits about 1000x too shallow to see
    bump.inputs["Distance"].default_value = 0.02
    links.new(height, bump.inputs["Height"])
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

    light = bpy.data.objects["light"]
    light.data.energy = params["light_energy"]
    light.location = params["light_location"]
    look_at(light, fruit.location)

    camera = scene.camera
    camera.location = (0, 0, fruit.location.z + camera_height(params["fill"], camera.data.lens))
    look_at(camera, fruit.location)
    # The peel needs the final camera position to place defects on the visible side
    bpy.context.view_layer.update()

    apply_peel(fruit, camera, params)
    stem_nodes = bpy.data.objects["stem"].data.materials[0].node_tree.nodes
    stem_nodes["stem_colour"].inputs["Factor"].default_value = params["stem_mix"]


def shifted_rgba(hue, saturation, value, params):
    # Global shift of defective fruit: duller and browner. All zero for fresh and green.
    hue = hue + (BROWN_HUE - hue) * params["shift_brown"]
    saturation = saturation * (1 - params["shift_dull"])
    # Brown is a dark orange, so moving toward it also darkens; desaturating alone gave pastel peach
    value = value * (1 - 2 * params["shift_brown"])
    return linear_rgba(colorsys.hsv_to_rgb(hue, saturation, value))


def set_vector(node, vector):
    node.inputs["X"].default_value, node.inputs["Y"].default_value, node.inputs["Z"].default_value = vector


def view_to_object(fruit, view, degrees_from_view, azimuth):
    """Turn a point given relative to the view direction into the fruit's own coordinates.

    The sampler only says "this far from the centre of the visible face"; converting here means
    a defect lands on the visible side whatever rotation and camera the image uses.
    """
    side = view.orthogonal().normalized()
    up = view.cross(side)
    theta, phi = math.radians(degrees_from_view), math.radians(azimuth)
    world = view * math.cos(theta) + (side * math.cos(phi) + up * math.sin(phi)) * math.sin(theta)
    return (fruit.matrix_world.to_3x3().inverted() @ world).normalized()


def apply_peel(fruit, camera, params):
    # Guards for the 2.4 sampler: generate.py refuses values outside the agreed ranges
    assert 0 <= params["shift_brown"] <= MAX_SHIFT_BROWN, f"shift_brown above cap: {params['shift_brown']}"
    assert 0 <= params["shift_dull"] <= MAX_SHIFT_DULL, f"shift_dull above cap: {params['shift_dull']}"
    assert len(params["patches"]) <= MAX_PATCHES, f"at most {MAX_PATCHES} patches"
    assert all(0 < p[2] <= MAX_PATCH_RADIUS for p in params["patches"]), "patch radius out of range"

    nodes = fruit.data.materials[0].node_tree.nodes
    nodes["base_colour"].outputs[0].default_value = shifted_rgba(
        params["hue"], params["saturation"], params["value"], params)
    nodes["blotch_colour"].outputs[0].default_value = shifted_rgba(
        params["blotch_hue"], params["saturation"], params["value"], params)
    nodes["blotch_amount"].outputs[0].default_value = params["blotch_amount"]
    set_vector(nodes["pattern_offset"], params["pattern_offset"])
    nodes["roughness"].inputs["To Min"].default_value = min(params["roughness"] + params["shift_rough"], 0.9)
    nodes["pits"].inputs["Scale"].default_value = params["pit_scale"]
    nodes["pit_bump"].inputs["Strength"].default_value = params["bump_strength"]

    for name in DEFECT_TYPES:
        nodes[f"{name}_on"].outputs[0].default_value = 1.0 if params["defect_type"] == name else 0.0

    view = (camera.location - fruit.location).normalized()
    set_vector(nodes["camera_direction"], view_to_object(fruit, view, 0.0, 0.0))

    # Voronoi distances are in cell units, so scale the threshold to keep spot size independent of count
    edge = params["spot_radius"] * params["spot_scale"]
    nodes["spot_cells"].inputs["Scale"].default_value = params["spot_scale"]
    nodes["spot_edges"].inputs["From Min"].default_value = edge
    nodes["spot_edges"].inputs["From Max"].default_value = edge * 0.6

    # Only one defect type is on per image, so the patch slots take that type's edge shape
    shape = PATCH_SHAPE["rot"] if params["defect_type"] == "rot" else PATCH_SHAPE["lesions"]
    nodes["edge_wobble"].outputs[0].default_value = shape["wobble"]
    for i in range(MAX_PATCHES):
        if i < len(params["patches"]):
            degrees_from_view, azimuth, radius = params["patches"][i]
            centre = view_to_object(fruit, view, degrees_from_view, azimuth)
        else:
            # Unused slot: centre at the origin is 1.0 from every surface point (see the assert at the top)
            centre, radius = (0.0, 0.0, 0.0), UNUSED_RADIUS
        set_vector(nodes[f"patch{i}_centre"], centre)
        nodes[f"patch{i}_core"].inputs["From Min"].default_value = radius
        nodes[f"patch{i}_core"].inputs["From Max"].default_value = radius * shape["core_soft"]
        nodes[f"patch{i}_outer"].inputs["From Min"].default_value = radius * shape["outer"]
        nodes[f"patch{i}_outer"].inputs["From Max"].default_value = radius * 0.95  # wide, smooth transition; 1.05 gave a sawtooth rim

    # Lower threshold = more of the noise passes = more mould; at 0 the noise never reaches 0.8.
    # Spore centres use a higher threshold on the same noise, so each blob keeps a white margin.
    mould_edge = 1.3 - 0.8 * params["mould_amount"]  # field peaks at 1.25, so amount 0 gives no mould
    nodes["mould_edges"].inputs["From Min"].default_value = mould_edge
    nodes["mould_edges"].inputs["From Max"].default_value = mould_edge + 0.3  # soft, fluffy margin
    nodes["mould_centre_edges"].inputs["From Min"].default_value = mould_edge + 0.35
    nodes["mould_centre_edges"].inputs["From Max"].default_value = mould_edge + 0.55
    nodes["mould_mix"].outputs[0].default_value = params["mould_mix"]


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
