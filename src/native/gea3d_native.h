// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <cstddef>
#include <cstdint>

// Native engine behind gea3d's three.js-style TS API (declared as the
// `gea3dNative` global in gea3d/src/renderer/native.ts and bound through the
// geatsc-plugin-gea host shims to these symbols).
//
// The scene graph lives here in float32: nodes carry TRS or an explicit local
// matrix, world matrices update through dirty flags, and ONE drawScene call
// per frame culls (bounding sphere vs the six frustum planes), transforms each
// vertex once, lights, fogs, clips, sorts and rasterizes. The TS classes are
// thin wrappers that write a node only when its transform or material changed.
//
// Arrays arrive as TS typed-array views: the templates read data()/size()
// directly, so no call copies an array into a temporary.
//
// Compiled into the firmware through the app's `gea.nativeSources`.

namespace gea::host::gea3d {

// ---- Buffers: flat float / uint32 arrays shared by geometries and instances.

double createBuffer();
void deleteBuffer(double buffer);
void bufferDataF32Raw(double buffer, const float *data, std::size_t count);
void bufferDataU32Raw(double buffer, const std::uint32_t *data, std::size_t count);

template <typename FloatArray>
void bufferDataF32(double buffer, const FloatArray &data)
{
	bufferDataF32Raw(buffer, data.data(), data.size());
}

template <typename Uint32Array>
void bufferDataU32(double buffer, const Uint32Array &data)
{
	bufferDataU32Raw(buffer, data.data(), data.size());
}

// ---- Geometry: buffer ids per attribute (-1 = absent; no index = sequential).

double createGeometry();
void deleteGeometry(double geometry);
void geometryBuffers(double geometry, double position, double index, double color, double normal);

// ---- Materials.
//
// shade: 0 unlit, 1 lambert, 2 normal. side: 0 front, 1 back, 2 double.
// flags: MaterialFlag bits below. size: point size in px (PointsMaterial).

enum MaterialFlag : int {
	kMaterialVertexColors = 1,
	kMaterialTransparent = 2,
	kMaterialFog = 4,
	kMaterialFlatShading = 8,
	kMaterialWireframe = 16,
	kMaterialDepthTest = 32,
	kMaterialDepthWrite = 64,
	kMaterialSizeAttenuation = 128,
	kMaterialVisible = 256,
	kMaterialAdditive = 512,
};

double createMaterial();
void deleteMaterial(double material);
void materialSet(double material, double shade, double r, double g, double b, double opacity, double side, double flags, double size);

// ---- Nodes.

enum NodeKind : int {
	kNodeGroup = 0,
	kNodeMesh = 1,
	kNodeLineSegments = 2,
	kNodeLine = 3,
	kNodePoints = 4,
	kNodeSprite = 5,
	kNodeCamera = 6,
	kNodeLight = 7,
};

enum LightKind : int {
	kLightAmbient = 1,
	kLightDirectional = 2,
	kLightHemisphere = 3,
	kLightPoint = 4,
};

double createNode(double kind);
// Frees the node; its children become roots.
void deleteNode(double node);
// parent < 0 detaches.
void nodeParent(double node, double parent);
void nodeTransform(double node, double px, double py, double pz, double qx, double qy, double qz, double qw, double sx, double sy, double sz);
void nodeMatrixRaw(double node, const float *elements, std::size_t count);
void nodeFlags(double node, double visible, double frustumCulled, double renderOrder);
void nodeGeometry(double node, double geometry, double material);
// InstancedMesh: `matrices` holds count column-major 4x4s; colors (optional,
// -1) holds count RGB triples.
void nodeInstances(double node, double matrices, double colors, double count);
void nodeLight(double node, double kind, double r, double g, double b, double intensity, double groundR, double groundG, double groundB, double distance, double decay);
// Directional lights shine from their world position toward this point.
void nodeLightTarget(double node, double x, double y, double z);
void cameraProjectionRaw(double node, const float *elements, std::size_t count, double near, double far);

template <typename FloatArray>
void nodeMatrix(double node, const FloatArray &elements)
{
	nodeMatrixRaw(node, elements.data(), elements.size());
}

template <typename FloatArray>
void cameraProjection(double node, const FloatArray &elements, double near, double far)
{
	cameraProjectionRaw(node, elements.data(), elements.size(), near, far);
}

// ---- Scene environment (stored on the scene's root node).
//
// background: 0 none (keep the pixels under the 3D view), 1 flat (top), 2
// vertical gradient (top to bottom). Colours are 0xRRGGBB. fog: 0 none,
// 1 linear (near..far), 2 exponential squared (density).
void sceneEnvironment(double scene, double background, double top, double bottom, double fog, double fogColor, double fogNear, double fogFar, double fogDensity);

// ---- Frame.

// depthBuffer: 1 = per-pixel depth test for opaque geometry (exact
// intersections), 0 = nearest-first coverage (each pixel written once).
// capacity: primitive store size per frame.
void renderOptions(double depthBuffer, double capacity);
// Where frames go: -1 = the display (composited under the retained UI when
// the engine supports it), otherwise an image surface from createSurface.
void renderTarget(double surface);
// A native pixel surface registered as an image (for imageFromId + a canvas
// drawImage); returns its image id, or -1.
double createSurface(double width, double height);
// Renders `scene` seen by `camera` into a width x height viewport at the
// display origin. Returns the primitives rasterized, or -1 if the present
// was rejected.
double drawScene(double scene, double camera, double width, double height);

// Diagnostics for the last frame: 0 submitted triangles, 1 near rejected,
// 2 backface culled, 3 clip dropped, 4 dropped (capacity), 5 meshes drawn,
// 6 meshes frustum-culled, 7 primitives, 8 composited under the UI (0/1).
double stat(double which);

// ---- Raycasting against uploaded geometry (world space).
//
// Tests one node (instances included) and returns 1 on a hit within
// [near, far], else 0. rayResult: 0 distance, 1..3 point, 4 face index,
// 5 instance id.
double raycast(double node, double ox, double oy, double oz, double dx, double dy, double dz, double near, double far);
double rayResult(double which);

// ---- Immediate API (the pre-scene-graph surface, kept for existing code).

void beginFrameRaw(const float *projection, std::size_t count, double near, double bgColor, double viewportW, double viewportH);
void ambientLight(double r, double g, double b);
void directionalLight(double x, double y, double z, double r, double g, double b);
void drawElementsRaw(double positionBuffer,
                     double indexBuffer,
                     double colorBuffer,
                     double triangleCount,
                     const float *modelView,
                     std::size_t modelViewCount,
                     double shadeMode,
                     double baseR,
                     double baseG,
                     double baseB,
                     double doubleSide);
double endFrame();

template <typename FloatArray>
void beginFrame(const FloatArray &projection, double near, double bgColor, double viewportW, double viewportH)
{
	beginFrameRaw(projection.data(), projection.size(), near, bgColor, viewportW, viewportH);
}

template <typename FloatArray>
void drawElements(double positionBuffer,
                  double indexBuffer,
                  double colorBuffer,
                  double triangleCount,
                  const FloatArray &modelView,
                  double shadeMode,
                  double baseR,
                  double baseG,
                  double baseB,
                  double doubleSide)
{
	drawElementsRaw(positionBuffer, indexBuffer, colorBuffer, triangleCount, modelView.data(), modelView.size(), shadeMode, baseR, baseG, baseB, doubleSide);
}

}  // namespace gea::host::gea3d
