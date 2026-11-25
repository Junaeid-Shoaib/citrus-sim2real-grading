# pyright: reportMissingModuleSource=false
"""Render a synthetic orange in Blender.

Run headless, rendering rows of a labels.csv written by src/make_scene_params.py:
    blender -b -P blender/generate.py -- --labels data/synthetic/wide/labels.csv --start 0 --count 500

Without --labels it renders one default fresh orange to --out.

All randomness lives outside Blender (src/make_scene_params.py). This script only turns a
row of parameters into an image, so a render is fully determined by its parameters and the
label never depends on the image.
"""

import argparse
import colorsys
import csv
import json
import math
import os
import sys
import time

import bpy
from mathutils import Euler, Vector

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
    "shift_dark": 0.0,        # fraction of brightness removed
    "shift_rough": 0.0,       # added to roughness (less shine)
    "defect_type": "none",    # none | spots (blackspot) | lesions (canker) | rot
    "spot_scale": 4.0,        # Voronoi cells per unit; about 3 gives ~5 visible spots, 5.5 gives ~30
    "spot_radius": 0.03,      # spot radius in fruit-radius units (about 2 px at 224 px)
    "spot_distort": 0.0,      # canker: noise distortion of the cells so marks are irregular, not round
    "wrinkle": 0.0,           # depth of shrivelled creases in the peel (defective only; strongest for blackspot)
    "canker_density": 0.5,    # canker: share of the visible face where marks cluster (0 to 1)
    # Rot patches: (degrees from the view direction, azimuth degrees, radius); up to 3, radius at most MAX_PATCH_RADIUS
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
    "light_temperature": 6500.0,  # kelvin; 6500 is close to neutral white
    "fill_ratio": 0.0,        # white fill light at the camera, as a fraction of the key light on the fruit
    "fill": 0.7,              # fraction of the image width the fruit covers
    "camera_tilt": 0.0,       # degrees from straight down
    "camera_azimuth": 0.0,    # degrees around the fruit
    "camera_shift": (0.0, 0.0),  # lens shift as a fraction of the frame, moves the fruit off-centre
    "stem_mix": 0.5,          # 0 = olive green, 1 = dark brown
    "bg_colour": (0.03, 0.03, 0.03),  # sRGB; default is the dark belt
    "bg_pattern": "plain",    # plain | speckle (worktop) | noise (cloth or paper folds)
    "bg_pattern_strength": 0.0,
    "bg_pattern_scale": 10.0,
}

# Stem colour range (sRGB), from stems seen in the real train split: olive-grey to near-black brown
STEM_OLIVE = (0.30, 0.32, 0.12)
STEM_BROWN = (0.20, 0.12, 0.05)

# Defect colours (sRGB), matched by eye to the real defective train images
SPOT_COLOUR = (0.10, 0.07, 0.04)          # blackspot: near-black pits
CANKER_COLOUR = (0.42, 0.30, 0.16)        # canker: light-brown corky marks
SCAB_COLOUR = (0.60, 0.50, 0.25)          # canker: yellow-brown scabby mottle (the grey-tan 0.55, 0.50, 0.36 cut saturation to 0.70)
# Rot and mould colours start from real rotten train images (10x10 px averages), adjusted so renders match
ROT_DARK = (0.26, 0.15, 0.03)             # dark brown patch
ROT_OLIVE = (0.45, 0.33, 0.15)            # olive-brown patch (more saturated than measured: renders came out grey)
MOULD_WHITE = (0.90, 0.89, 0.85)          # mycelium margin
MOULD_BLUE_GREEN = (0.40, 0.48, 0.42)     # muted grey-green spore centre (darker than measured: top light brightens it)

# Caps on the global shift. Real defective train fruit is 0.64-0.76x as bright as fresh and
# 0.89-0.97x as saturated, with no shift toward brown, so the shift darkens and slightly dulls.
# Keeping dulling small stops a greenish base turning beige ("beige means defective" shortcut, D5).
MAX_SHIFT_DARK = 0.65
MAX_SHIFT_DULL = 0.05   # 0.12 gave synthetic defective/fresh saturation 0.84 in the wide pilot; real is 0.89-0.97

DEFECT_TYPES = ["spots", "lesions", "rot"]   # blackspot, canker, rot
MAX_PATCHES = 3  # fixed number of rot patch slots, so the shader layout never changes

# Rot patch edge: soft, irregular, with a water-soaked rim. "outer" is where the rim ends, as a
# multiple of the radius. Canker no longer uses patches: it reuses the spot machinery.
PATCH_WOBBLE = 0.25
PATCH_CORE_SOFT = 0.55
PATCH_OUTER = 1.15
MAX_PATCH_RADIUS = 0.7   # chord length on the unit sphere, about 41 degrees: a local patch, not the whole fruit
UNUSED_RADIUS = 0.01
# An unused slot has its centre at the fruit's centre, so every surface point is 1.0 from it.
# It can only show if its outer edge plus the worst-case edge wobble reaches 1.0.
assert PATCH_OUTER * UNUSED_RADIUS + PATCH_WOBBLE / 2 < 0.5


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
    # Half the default white sheen: at 0.5 the white reflection greyed the peel colour
    bsdf.inputs["Specular IOR Level"].default_value = 0.25

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

    # Blackspot and canker share one mechanism: thresholded Voronoi cells, masked to the
    # camera-facing cap (within 60 degrees, cos 60 = 0.5). Canker distorts the cells so marks are
    # irregular, colours them light brown with corky mottling, and raises them instead of sinking them.
    cork_noise = nodes.new("ShaderNodeTexNoise")
    cork_noise.inputs["Scale"].default_value = 30.0
    links.new(shifted, cork_noise.inputs["Vector"])
    centred = nodes.new("ShaderNodeVectorMath")
    centred.operation = "SUBTRACT"
    links.new(cork_noise.outputs["Color"], centred.inputs[0])
    centred.inputs[1].default_value = (0.5, 0.5, 0.5)
    distortion = nodes.new("ShaderNodeVectorMath")
    distortion.operation = "SCALE"
    links.new(centred.outputs[0], distortion.inputs[0])
    links.new(named(nodes, "ShaderNodeValue", "spot_distort").outputs[0], distortion.inputs["Scale"])
    spot_coords = nodes.new("ShaderNodeVectorMath")
    spot_coords.operation = "ADD"
    links.new(shifted, spot_coords.inputs[0])
    links.new(distortion.outputs[0], spot_coords.inputs[1])

    spot_tex = named(nodes, "ShaderNodeTexVoronoi", "spot_cells")
    links.new(spot_coords.outputs[0], spot_tex.inputs["Vector"])
    marks = map_range(nodes, links, spot_tex.outputs["Distance"], "spot_edges")
    facing = nodes.new("ShaderNodeVectorMath")
    facing.operation = "DOT_PRODUCT"
    links.new(direction, facing.inputs[0])
    links.new(named(nodes, "ShaderNodeCombineXYZ", "camera_direction").outputs[0], facing.inputs[1])
    cap = math_op(nodes, links, "GREATER_THAN", facing.outputs["Value"], 0.5)
    marks = math_op(nodes, links, "MULTIPLY", math_op(nodes, links, "MULTIPLY", marks, cap),
                    math_op(nodes, links, "ADD", spots_on, lesions_on))
    # Canker marks cluster unevenly: a large-scale noise mask thins them in some areas, so the
    # cells do not read as a regular polka-dot grid. canker_density sets how much area keeps marks.
    cluster_noise = nodes.new("ShaderNodeTexNoise")
    cluster_noise.inputs["Scale"].default_value = 1.5
    links.new(shifted, cluster_noise.inputs["Vector"])
    cluster = map_range(nodes, links, cluster_noise.outputs["Fac"], "canker_cluster")
    # Blackspot is unaffected: mask = 1 - lesions_on x (1 - cluster)
    keep = math_op(nodes, links, "SUBTRACT", 1.0,
                   math_op(nodes, links, "MULTIPLY", lesions_on, math_op(nodes, links, "SUBTRACT", 1.0, cluster)))
    marks = math_op(nodes, links, "MULTIPLY", marks, keep)
    cork_colour = mix_colour(nodes, links, linear_rgba(CANKER_COLOUR),
                             linear_rgba(tuple(c * 0.5 for c in CANKER_COLOUR)), cork_noise.outputs["Fac"])
    mark_colour = mix_colour(nodes, links, linear_rgba(SPOT_COLOUR), cork_colour, lesions_on)
    colour = mix_colour(nodes, links, colour, mark_colour, marks)

    # Canker also roughens the whole peel: a scabby grey-tan mottle, not only separate marks.
    # Real cankered fruit reads as a rough mottled surface at 224 px; dots alone did not.
    scab_noise = nodes.new("ShaderNodeTexNoise")
    scab_noise.inputs["Scale"].default_value = 14.0
    scab_noise.inputs["Detail"].default_value = 6.0
    links.new(shifted, scab_noise.inputs["Vector"])
    scab = map_range(nodes, links, scab_noise.outputs["Fac"], "scab_edges")
    scab.node.inputs["From Min"].default_value = 0.5
    scab.node.inputs["From Max"].default_value = 0.7
    scab = math_op(nodes, links, "MULTIPLY", math_op(nodes, links, "MULTIPLY", scab, lesions_on), 0.6)
    colour = mix_colour(nodes, links, colour, linear_rgba(SCAB_COLOUR), scab)

    # Shrivelled creases: ridged noise (1 - |2n - 1|, sharpened) gives thin connected lines
    wrinkle_noise = nodes.new("ShaderNodeTexNoise")
    wrinkle_noise.inputs["Scale"].default_value = 4.0
    wrinkle_noise.inputs["Detail"].default_value = 3.0
    wrinkle_noise.inputs["Distortion"].default_value = 1.0
    links.new(shifted, wrinkle_noise.inputs["Vector"])
    ridge = math_op(nodes, links, "ABSOLUTE",
                    math_op(nodes, links, "SUBTRACT", math_op(nodes, links, "MULTIPLY", wrinkle_noise.outputs["Fac"], 2.0), 1.0))
    ridge = math_op(nodes, links, "POWER", math_op(nodes, links, "SUBTRACT", 1.0, ridge), 6.0)
    creases = math_op(nodes, links, "MULTIPLY", ridge, named(nodes, "ShaderNodeValue", "wrinkle").outputs[0])

    # Rot patches: distance from each centre, roughened by noise so edges are irregular
    edge_noise = nodes.new("ShaderNodeTexNoise")
    edge_noise.inputs["Scale"].default_value = 6.0
    edge_noise.inputs["Detail"].default_value = 0.0  # one octave: smooth irregular lobes; finer octaves scalloped the edge at 224 px
    links.new(shifted, edge_noise.inputs["Vector"])
    wobble = math_op(nodes, links, "MULTIPLY", math_op(nodes, links, "SUBTRACT", edge_noise.outputs["Fac"], 0.5),
                     PATCH_WOBBLE)
    core, outer = 0.0, 0.0
    for i in range(MAX_PATCHES):
        distance = nodes.new("ShaderNodeVectorMath")
        distance.operation = "DISTANCE"
        links.new(direction, distance.inputs[0])
        links.new(named(nodes, "ShaderNodeCombineXYZ", f"patch{i}_centre").outputs[0], distance.inputs[1])
        d = math_op(nodes, links, "ADD", distance.outputs["Value"], wobble)
        core = math_op(nodes, links, "MAXIMUM", core, map_range(nodes, links, d, f"patch{i}_core"))
        outer = math_op(nodes, links, "MAXIMUM", outer, map_range(nodes, links, d, f"patch{i}_outer"))
    ring = math_op(nodes, links, "SUBTRACT", outer, core)  # band just outside the core
    ring.node.use_clamp = True

    # Rot: dark-brown to olive-brown patch (mottled by the edge noise), ringed by a darker water-soaked rim
    rot = math_op(nodes, links, "MULTIPLY", core, rot_on)
    rot_colour = mix_colour(nodes, links, linear_rgba(ROT_DARK), linear_rgba(ROT_OLIVE), edge_noise.outputs["Fac"])
    colour = mix_colour(nodes, links, colour, rot_colour, rot)
    rim = math_op(nodes, links, "MULTIPLY", ring, rot_on)
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
    # Powdery look: fine noise thins the mould everywhere, so the patch shows through in places
    # and the wide soft edge fades out unevenly instead of ending like a flat decal
    powder_noise = nodes.new("ShaderNodeTexNoise")
    powder_noise.inputs["Scale"].default_value = 40.0
    powder_noise.inputs["Detail"].default_value = 4.0
    links.new(shifted, powder_noise.inputs["Vector"])
    powder = map_range(nodes, links, powder_noise.outputs["Fac"], "powder")
    powder.node.inputs["From Min"].default_value = 0.3
    powder.node.inputs["From Max"].default_value = 0.65
    powder.node.inputs["To Min"].default_value = 0.35
    mould = math_op(nodes, links, "MULTIPLY", mould, powder)
    spores = map_range(nodes, links, mould_field, "mould_centre_edges")
    spores = math_op(nodes, links, "MULTIPLY", math_op(nodes, links, "MULTIPLY", spores, mould),
                     named(nodes, "ShaderNodeValue", "mould_mix").outputs[0])
    colour = mix_colour(nodes, links, colour, linear_rgba(MOULD_WHITE), mould)
    colour = mix_colour(nodes, links, colour, linear_rgba(MOULD_BLUE_GREEN), spores)
    links.new(colour, bsdf.inputs["Base Color"])

    # Rot, mould and corky canker are matte, the wet rim is glossy; the rest of the peel keeps its sampled roughness
    canker = math_op(nodes, links, "MULTIPLY", marks, lesions_on)
    matte = math_op(nodes, links, "MAXIMUM", math_op(nodes, links, "MAXIMUM", rot, mould),
                    math_op(nodes, links, "MAXIMUM", canker, scab))
    roughness = map_range(nodes, links, matte, "roughness")
    roughness.node.interpolation_type = "LINEAR"
    roughness.node.inputs["To Max"].default_value = 0.9
    roughness = math_op(nodes, links, "SUBTRACT", roughness, math_op(nodes, links, "MULTIPLY", rim, 0.4))
    roughness.node.use_clamp = True
    links.new(roughness, bsdf.inputs["Roughness"])

    # Height: oil-gland pits, plus raised rough canker marks and sunken black spots and rot
    pits = named(nodes, "ShaderNodeTexVoronoi", "pits")
    links.new(coords, pits.inputs["Vector"])
    corky_relief = math_op(nodes, links, "ADD", 0.5, cork_noise.outputs["Fac"])
    height = math_op(nodes, links, "ADD", pits.outputs["Distance"],
                     math_op(nodes, links, "MULTIPLY", canker, corky_relief))
    height = math_op(nodes, links, "SUBTRACT", height, math_op(nodes, links, "MULTIPLY", rot, 1.0))
    height = math_op(nodes, links, "SUBTRACT", height, math_op(nodes, links, "MULTIPLY", rim, 0.3))
    black_spots = math_op(nodes, links, "MULTIPLY", marks, spots_on)
    height = math_op(nodes, links, "SUBTRACT", height, math_op(nodes, links, "MULTIPLY", black_spots, 0.5))
    # Scab relief over the canker peel, sunken creases, and a little fluffy relief in the mould
    height = math_op(nodes, links, "ADD", height,
                     math_op(nodes, links, "MULTIPLY", math_op(nodes, links, "MULTIPLY", scab_noise.outputs["Fac"], lesions_on), 0.8))
    height = math_op(nodes, links, "SUBTRACT", height, math_op(nodes, links, "MULTIPLY", creases, 3.0))
    height = math_op(nodes, links, "ADD", height, math_op(nodes, links, "MULTIPLY", math_op(nodes, links, "MULTIPLY", mould, powder_noise.outputs["Fac"]), 0.6))
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


def make_background_material():
    """Surface under the fruit: plain, speckled worktop, or cloth/paper with soft folds.

    Like the peel, the layout is fixed and patterns are switched with 0/1 factors.
    """
    mat = bpy.data.materials.new("background")
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.8
    coords = nodes.new("ShaderNodeTexCoord").outputs["Object"]
    strength = named(nodes, "ShaderNodeValue", "bg_strength").outputs[0]
    colour = named(nodes, "ShaderNodeRGB", "bg_colour").outputs[0]

    # Speckle: small dark dots, like the speckled worktop in some real photos
    speckle_tex = named(nodes, "ShaderNodeTexVoronoi", "bg_speckle")
    links.new(coords, speckle_tex.inputs["Vector"])
    speckle = map_range(nodes, links, speckle_tex.outputs["Distance"], "bg_speckle_edges")
    speckle.node.inputs["From Min"].default_value = 0.12
    speckle.node.inputs["From Max"].default_value = 0.08
    speckle_on = named(nodes, "ShaderNodeValue", "bg_speckle_on").outputs[0]
    speckle = math_op(nodes, links, "MULTIPLY", math_op(nodes, links, "MULTIPLY", speckle, speckle_on), strength)
    colour = mix_colour(nodes, links, colour, (0.05, 0.05, 0.06, 1.0), speckle)

    # Noise: soft light and dark areas plus a bump, like folds in a cloth or paper
    noise_tex = named(nodes, "ShaderNodeTexNoise", "bg_noise")
    links.new(coords, noise_tex.inputs["Vector"])
    noise_on = named(nodes, "ShaderNodeValue", "bg_noise_on").outputs[0]
    shade = math_op(nodes, links, "MULTIPLY", math_op(nodes, links, "MULTIPLY", noise_tex.outputs["Fac"], noise_on), strength)
    colour = mix_colour(nodes, links, colour, (0.0, 0.0, 0.0, 1.0), math_op(nodes, links, "MULTIPLY", shade, 0.5))
    links.new(colour, bsdf.inputs["Base Color"])
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Distance"].default_value = 0.05
    links.new(shade, bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def build_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    setup_render(scene)

    world = bpy.data.worlds.new("world")
    # Dim grey ambient so shadowed peel is dark but not black. 0.3 washed green and defective
    # fruit out to 0.07-0.1 below the real saturation; 0.15 (with less sheen) matched it.
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.15, 0.15, 0.15, 1.0)
    scene.world = world

    # Large plane so a tilted camera never sees past its edge
    bpy.ops.mesh.primitive_plane_add(size=100)
    ground = bpy.context.object
    ground.name = "ground"
    ground.data.materials.append(make_background_material())

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

    # Weak fill from the camera position: large and neutral white, so it lifts shadows without
    # the greying a stronger grey ambient caused
    bpy.ops.object.light_add(type="AREA")
    fill_light = bpy.context.object
    fill_light.name = "fill_light"
    fill_light.data.size = 4.0
    fill_light.data.color = (1.0, 1.0, 1.0)

    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.data.lens = 50
    scene.camera = camera
    return scene


def rest_on_ground(fruit):
    # D3: after rotation, drop the fruit so its lowest point touches the ground (z = 0)
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
    light.data.use_temperature = True
    light.data.temperature = params["light_temperature"]
    light.location = params["light_location"]
    look_at(light, fruit.location)

    # Camera on a sphere around the fruit: tilt 0 is straight down
    camera = scene.camera
    distance = camera_height(params["fill"], camera.data.lens)
    tilt, azimuth = math.radians(params["camera_tilt"]), math.radians(params["camera_azimuth"])
    camera.location = fruit.location + distance * Vector((math.sin(tilt) * math.cos(azimuth),
                                                          math.sin(tilt) * math.sin(azimuth),
                                                          math.cos(tilt)))
    look_at(camera, fruit.location)
    camera.data.shift_x, camera.data.shift_y = params["camera_shift"]

    # Fill light sits at the camera. Light on the fruit falls with distance squared, so scale its
    # power by (camera distance / key distance)^2 to make fill_ratio a ratio of light on the fruit.
    fill_light = bpy.data.objects["fill_light"]
    key_distance = (light.location - fruit.location).length
    fill_light.data.energy = params["fill_ratio"] * params["light_energy"] * (distance / key_distance) ** 2
    fill_light.location = camera.location
    look_at(fill_light, fruit.location)

    apply_background(params)
    # The peel needs the final camera position to place defects on the visible side
    bpy.context.view_layer.update()

    apply_peel(fruit, camera, params)
    stem_nodes = bpy.data.objects["stem"].data.materials[0].node_tree.nodes
    stem_nodes["stem_colour"].inputs["Factor"].default_value = params["stem_mix"]


def apply_background(params):
    nodes = bpy.data.objects["ground"].data.materials[0].node_tree.nodes
    nodes["bg_colour"].outputs[0].default_value = linear_rgba(params["bg_colour"])
    nodes["bg_strength"].outputs[0].default_value = params["bg_pattern_strength"]
    nodes["bg_speckle_on"].outputs[0].default_value = 1.0 if params["bg_pattern"] == "speckle" else 0.0
    nodes["bg_noise_on"].outputs[0].default_value = 1.0 if params["bg_pattern"] == "noise" else 0.0
    nodes["bg_speckle"].inputs["Scale"].default_value = params["bg_pattern_scale"]
    nodes["bg_noise"].inputs["Scale"].default_value = params["bg_pattern_scale"]


def shifted_rgba(hue, saturation, value, params):
    # Global shift of defective fruit: darker and slightly duller, hue unchanged (real defective
    # fruit is not browner in hue; canker's yellower hue comes from its own base range).
    # All zero for fresh and green.
    saturation = saturation * (1 - params["shift_dull"])
    value = value * (1 - params["shift_dark"])
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
    assert 0 <= params["shift_dark"] <= MAX_SHIFT_DARK, f"shift_dark above cap: {params['shift_dark']}"
    assert 0 <= params["shift_dull"] <= MAX_SHIFT_DULL, f"shift_dull above cap: {params['shift_dull']}"
    assert len(params["patches"]) <= MAX_PATCHES, f"at most {MAX_PATCHES} patches"
    assert not params["patches"] or params["defect_type"] == "rot", "only rot uses patches"
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
    nodes["spot_distort"].outputs[0].default_value = params["spot_distort"]
    nodes["wrinkle"].outputs[0].default_value = params["wrinkle"]
    # Higher density lowers the threshold on the cluster noise (noise Fac sits mostly in 0.3-0.7)
    cluster_edge = 0.7 - 0.4 * params["canker_density"]
    nodes["canker_cluster"].inputs["From Min"].default_value = cluster_edge
    nodes["canker_cluster"].inputs["From Max"].default_value = cluster_edge + 0.1

    for i in range(MAX_PATCHES):
        if i < len(params["patches"]):
            degrees_from_view, azimuth, radius = params["patches"][i]
            centre = view_to_object(fruit, view, degrees_from_view, azimuth)
        else:
            # Unused slot: centre at the origin is 1.0 from every surface point (see the assert at the top)
            centre, radius = (0.0, 0.0, 0.0), UNUSED_RADIUS
        set_vector(nodes[f"patch{i}_centre"], centre)
        nodes[f"patch{i}_core"].inputs["From Min"].default_value = radius
        nodes[f"patch{i}_core"].inputs["From Max"].default_value = radius * PATCH_CORE_SOFT
        nodes[f"patch{i}_outer"].inputs["From Min"].default_value = radius * PATCH_OUTER
        nodes[f"patch{i}_outer"].inputs["From Max"].default_value = radius * 0.95  # wide, smooth transition; 1.05 gave a sawtooth rim

    # Lower threshold on the mould field = more mould. Spore centres use a higher threshold on the
    # same field, so the grey-green sits inside a white margin.
    mould_edge = 1.3 - 0.8 * params["mould_amount"]  # field peaks at 1.25, so amount 0 gives no mould
    nodes["mould_edges"].inputs["From Min"].default_value = mould_edge
    nodes["mould_edges"].inputs["From Max"].default_value = mould_edge + 0.6  # wide, soft margin; 0.3 still ended like a decal
    nodes["mould_centre_edges"].inputs["From Min"].default_value = mould_edge + 0.35
    nodes["mould_centre_edges"].inputs["From Max"].default_value = mould_edge + 0.55
    nodes["mould_mix"].outputs[0].default_value = params["mould_mix"]


def render(scene, out_path):
    scene.render.filepath = out_path
    start = time.time()
    bpy.ops.render.render(write_still=True)
    return time.time() - start


# labels.csv stores these as JSON text because they are lists or tuples
JSON_COLUMNS = ["pattern_offset", "rotation", "light_location", "camera_shift", "bg_colour", "patches"]
TEXT_COLUMNS = ["path", "render_path", "grade", "defect_type", "randomisation", "bg_type", "bg_pattern"]


def row_to_params(row):
    # Blender's Python has no pandas, so labels.csv is read with csv and converted here
    params = dict(DEFAULT_PARAMS)
    for key, text in row.items():
        if key in JSON_COLUMNS:
            params[key] = json.loads(text)
        elif key in TEXT_COLUMNS:
            params[key] = text
        else:
            params[key] = float(text)
    return params


def parse_args():
    # Blender passes its own arguments first; ours come after "--"
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", help="labels.csv from src/make_scene_params.py")
    parser.add_argument("--start", type=int, default=0, help="first row to render")
    parser.add_argument("--count", type=int, default=500, help="rows per run; one Blender process per chunk")
    parser.add_argument("--out", default="data/synthetic/step2_1_fresh.jpg", help="output when --labels is not given")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    scene = build_scene()
    if not args.labels:
        apply_params(scene, DEFAULT_PARAMS)
        # Absolute path: Blender would otherwise resolve relative paths against the .blend location
        seconds = render(scene, os.path.abspath(args.out))
        print(f"Rendered {args.out} in {seconds:.2f} s")
        return

    with open(args.labels) as f:
        rows = list(csv.DictReader(f))[args.start:args.start + args.count]
    rendered, skipped, total = 0, 0, 0.0
    for row in rows:
        # Raw render; src/camera_process.py turns it into the final image at row["path"]
        out_path = os.path.abspath(row["render_path"])
        # Skip finished images so a crashed or interrupted chunk can simply be re-run
        if os.path.exists(out_path):
            skipped += 1
            continue
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        apply_params(scene, row_to_params(row))
        total += render(scene, out_path)
        rendered += 1
    mean = total / rendered if rendered else 0.0
    print(f"Rows {args.start}-{args.start + len(rows) - 1}: rendered {rendered}, skipped {skipped}, "
          f"mean {mean:.2f} s per image")


if __name__ == "__main__":
    main()
