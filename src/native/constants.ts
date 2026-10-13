// SPDX-License-Identifier: GPL-3.0-only
// Enumerations shared by the native engine and its TS faces (api.ts,
// api.web.ts); they mirror the enums in gea3d_native.h.

export const NODE_GROUP = 0
export const NODE_MESH = 1
export const NODE_LINE_SEGMENTS = 2
export const NODE_LINE = 3
export const NODE_POINTS = 4
export const NODE_SPRITE = 5
export const NODE_CAMERA = 6
export const NODE_LIGHT = 7

export const LIGHT_AMBIENT = 1
export const LIGHT_DIRECTIONAL = 2
export const LIGHT_HEMISPHERE = 3
export const LIGHT_POINT = 4

export const MATERIAL_VERTEX_COLORS = 1
export const MATERIAL_TRANSPARENT = 2
export const MATERIAL_FOG = 4
export const MATERIAL_FLAT_SHADING = 8
export const MATERIAL_WIREFRAME = 16
export const MATERIAL_DEPTH_TEST = 32
export const MATERIAL_DEPTH_WRITE = 64
export const MATERIAL_SIZE_ATTENUATION = 128
export const MATERIAL_VISIBLE = 256
export const MATERIAL_ADDITIVE = 512

export const BACKGROUND_NONE = 0
export const BACKGROUND_FLAT = 1
export const BACKGROUND_GRADIENT = 2

export const FOG_NONE = 0
export const FOG_LINEAR = 1
export const FOG_EXP2 = 2

export const STAT_SUBMITTED = 0
export const STAT_NEAR_REJECTED = 1
export const STAT_CULLED = 2
export const STAT_CLIP_DROPPED = 3
export const STAT_DROPPED = 4
export const STAT_MESHES_DRAWN = 5
export const STAT_MESHES_CULLED = 6
export const STAT_PRIMITIVES = 7
export const STAT_COMPOSITED = 8

export const RAY_DISTANCE = 0
export const RAY_POINT_X = 1
export const RAY_POINT_Y = 2
export const RAY_POINT_Z = 3
export const RAY_FACE = 4
export const RAY_INSTANCE = 5
