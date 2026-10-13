// SPDX-License-Identifier: GPL-3.0-only
// The per-vertex and per-pixel paths must be optimized even when the
// firmware is built for size.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("O2")
#endif

#include "gea3d_native.h"

#include "gea3d_math.h"
#include "gea3d_raster.h"

#include "canvas.h"
#include "display.h"
#include "image.h"
#include "pixel.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <new>
#include <vector>

// Retained-UI compositing: with an engine that knows the underlay hook, a
// mounted UI tree repaints the 3D view under its own nodes in the same flush.
#if __has_include("display_underlay.h")
#include "display_underlay.h"
#define GEA3D_UNDERLAY 1
#else
#define GEA3D_UNDERLAY 0
#endif

// Async present on ESP32: the raster + panel stream of frame N runs on core 0
// while the app task builds frame N+1 (radios are off in the gea3d profile).
#if defined(ESP_PLATFORM) && __has_include("freertos/FreeRTOS.h")
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#define GEA3D_ASYNC_PRESENT 1
#else
#define GEA3D_ASYNC_PRESENT 0
#endif

// Band copy-out by the STM32F7's DMA2D, overlapping the next band's raster.
#if defined(STM32F746xx) && __has_include("stm32f7xx.h")
#include "stm32f7xx.h"
#define GEA3D_DMA2D_COPY 1
#else
#define GEA3D_DMA2D_COPY 0
#endif

// Framebuffers the rasterizer cannot address directly (sub-byte packed gray,
// rotated panels) present flat triangles through the display's own command.
#if GEA_PIXEL_STORAGE_PACKED || (defined(GEA_EMBEDDED_DISPLAY_ROTATE_LANDSCAPE) && GEA_EMBEDDED_DISPLAY_ROTATE_LANDSCAPE)
#define GEA3D_PRESENT_COMMANDS 1
#else
#define GEA3D_PRESENT_COMMANDS 0
#endif

namespace gea::host::gea3d {

namespace {

namespace pixel = gea::framework::graphics::pixel;
namespace pd = gea::platform::display;
using raster::Pixel;
using raster::Prim;
using raster::detail::clampFloat;
using raster::detail::maxInt;
using raster::detail::minInt;
using raster::detail::roundToInt;

constexpr int kMaxLights = 4;
constexpr int kDepthBuckets = 1024;
// renderOrder spans up to this wide sort by counting; wider ones by stable_sort.
constexpr int kLayerCountingRange = 256;
// Screen coordinates stay within +-16000 px, so 16.16 edge positions fit int32.
constexpr float kCoordLimit = 16000.0f;
constexpr int kDefaultCapacity = 4096;

// ---------------------------------------------------------------- resources

struct Buffer {
	bool live = false;
	std::vector<float> f32;
	std::vector<std::uint32_t> u32;
	std::uint32_t version = 0;
};

struct Geometry {
	bool live = false;
	int position = -1;
	int index = -1;
	int color = -1;
	int normal = -1;
	// Bounding sphere of `position` at `boundsVersion`.
	int boundsBuffer = -1;
	std::uint32_t boundsVersion = 0;
	float center[3] = {0, 0, 0};
	float radius = -1.0f;
};

struct Material {
	bool live = false;
	int shade = 1;
	float color[3] = {255, 255, 255};  // 0..255
	float opacity = 1.0f;
	int side = 0;
	int flags = kMaterialVisible | kMaterialDepthTest | kMaterialDepthWrite | kMaterialFog;
	float size = 1.0f;
};

struct Environment {
	raster::Background background;
	int fog = 0;
	float fogColor[3] = {0, 0, 0};
	float fogNear = 1.0f;
	float fogFar = 1000.0f;
	float fogDensity = 0.00025f;
};

struct Node {
	bool live = false;
	int kind = kNodeGroup;
	int parent = -1;
	int firstChild = -1;
	int nextSibling = -1;
	int prevSibling = -1;

	float position[3] = {0, 0, 0};
	float quaternion[4] = {0, 0, 0, 1};
	float scale[3] = {1, 1, 1};
	float local[16];
	float world[16];
	bool useMatrix = false;
	bool localDirty = true;
	std::uint32_t worldVersion = 1;
	std::uint32_t parentWorldVersion = 0;

	bool visible = true;
	bool frustumCulled = true;
	int renderOrder = 0;

	int geometry = -1;
	int material = -1;
	// An InstancedMesh draws its instances only (none at a count of 0);
	// any other mesh draws once at its own transform.
	bool instanced = false;
	int instanceMatrices = -1;
	int instanceColors = -1;
	int instanceCount = 0;

	int lightKind = 0;
	float lightColor[3] = {1, 1, 1};
	float intensity = 1.0f;
	float groundColor[3] = {1, 1, 1};
	float distance = 0.0f;
	float decay = 2.0f;
	float target[3] = {0, 0, 0};

	float projection[16];
	float near = 0.1f;
	float far = 2000.0f;

	Environment environment;

	Node()
	{
		math::identity(local);
		math::identity(world);
		math::identity(projection);
	}
};

template <typename T>
int allocate(std::vector<T> &pool)
{
	for (std::size_t i = 0; i < pool.size(); i++) {
		if (!pool[i].live) {
			pool[i] = T();
			pool[i].live = true;
			return static_cast<int>(i);
		}
	}
	pool.emplace_back();
	pool.back().live = true;
	return static_cast<int>(pool.size() - 1);
}

template <typename T>
T *lookup(std::vector<T> &pool, double id)
{
	const int index = static_cast<int>(id);
	if (index < 0 || index >= static_cast<int>(pool.size()) || !pool[static_cast<std::size_t>(index)].live) return nullptr;
	return &pool[static_cast<std::size_t>(index)];
}

std::vector<Buffer> g_buffers;
std::vector<Geometry> g_geometries;
std::vector<Material> g_materials;
std::vector<Node> g_nodes;

// ---------------------------------------------------------------- scene graph

void unlink(int id)
{
	Node &node = g_nodes[static_cast<std::size_t>(id)];
	if (node.parent < 0) return;

	Node &parent = g_nodes[static_cast<std::size_t>(node.parent)];
	if (node.prevSibling >= 0) g_nodes[static_cast<std::size_t>(node.prevSibling)].nextSibling = node.nextSibling;
	else parent.firstChild = node.nextSibling;
	if (node.nextSibling >= 0) g_nodes[static_cast<std::size_t>(node.nextSibling)].prevSibling = node.prevSibling;
	node.parent = -1;
	node.prevSibling = -1;
	node.nextSibling = -1;
	node.localDirty = true;
}

void link(int id, int parentId)
{
	Node &node = g_nodes[static_cast<std::size_t>(id)];
	Node &parent = g_nodes[static_cast<std::size_t>(parentId)];
	node.parent = parentId;
	node.prevSibling = -1;
	node.nextSibling = parent.firstChild;
	if (parent.firstChild >= 0) g_nodes[static_cast<std::size_t>(parent.firstChild)].prevSibling = id;
	parent.firstChild = id;
	node.localDirty = true;
}

bool isAncestor(int ancestor, int id)
{
	for (int cursor = id; cursor >= 0; cursor = g_nodes[static_cast<std::size_t>(cursor)].parent) {
		if (cursor == ancestor) return true;
	}
	return false;
}

// Recomputes the node's world matrix if it or its parent moved. The parent
// must already be current.
void refreshWorld(Node &node)
{
	const Node *parent = node.parent >= 0 ? &g_nodes[static_cast<std::size_t>(node.parent)] : nullptr;
	const std::uint32_t parentVersion = parent ? parent->worldVersion : 0;
	if (!node.localDirty && node.parentWorldVersion == parentVersion) return;

	if (node.localDirty && !node.useMatrix) math::compose(node.local, node.position, node.quaternion, node.scale);
	if (parent) math::multiplyAffine(node.world, parent->world, node.local);
	else std::memcpy(node.world, node.local, sizeof(node.world));
	node.localDirty = false;
	node.parentWorldVersion = parentVersion;
	node.worldVersion++;
}

// World matrix of a node anywhere in a graph (walks the ancestor chain).
void ensureWorld(int id)
{
	Node &node = g_nodes[static_cast<std::size_t>(id)];
	if (node.parent >= 0) ensureWorld(node.parent);
	refreshWorld(node);
}

// ---------------------------------------------------------------- frame state

struct Lights {
	float ambient[3] = {0, 0, 0};
	int directionalCount = 0;
	float direction[kMaxLights][3];
	float directionalColor[kMaxLights][3];
	int pointCount = 0;
	float pointPosition[kMaxLights][3];
	float pointColor[kMaxLights][3];
	float pointDistance[kMaxLights];
	float pointDecay[kMaxLights];
	int hemisphereCount = 0;
	float hemisphereUp[kMaxLights][3];
	float sky[kMaxLights][3];
	float ground[kMaxLights][3];

	void reset() { *this = Lights(); }
};

struct Stats {
	int submitted = 0;
	int nearRejected = 0;
	int culled = 0;
	int clipDropped = 0;
	int dropped = 0;
	int meshesDrawn = 0;
	int meshesCulled = 0;
	int primitives = 0;
	int composited = 0;
};

struct View {
	float view[16];
	float projection[16];
	float planes[6][4];  // view-space frustum
	float near = 0.1f;
	float far = 2000.0f;
	bool orthographic = false;
	float halfW = 0.0f;
	float halfH = 0.0f;
	int width = 0;
	int height = 0;
};

// The material of one draw, resolved to what the vertex stage needs.
struct Surface {
	int shade = 1;
	float color[3] = {255, 255, 255};  // 0..255
	int side = 0;
	bool vertexColors = false;
	bool blend = false;
	bool additive = false;
	std::uint8_t alpha = 255;
	// The drawing object's renderOrder.
	int layer = 0;
	bool fog = true;
	bool flat = false;
	bool wireframe = false;
	float size = 1.0f;
	bool sizeAttenuation = true;
};

// Per-vertex results of the current draw, one record per vertex so a
// triangle's corners sit in few cache lines.
struct Vertex {
	float view[3];    // view space
	float color[3];   // final colour 0..255, or the unlit base where faces are shaded
	float fog;        // fog factor where faces are shaded (1 = no fog)
	std::int32_t screen[2];  // 28.4
	float depth;      // near / w
	std::uint8_t rgb[3];  // `color` as bytes, when shaded per vertex
	std::uint8_t front;   // in front of the near plane (screen and depth are set)
};

// `fixed` (optional) is a pool in faster memory used when a draw fits.
struct VertexScratch {
	std::vector<Vertex> items;
	Vertex *fixed = nullptr;
	std::size_t fixedCount = 0;
	Vertex *current = nullptr;

	Vertex *prepare(std::size_t count)
	{
		if (count <= fixedCount) {
			current = fixed;
			return current;
		}
		if (items.size() < count) items.resize(count);
		current = items.data();
		return current;
	}
};

// Band rendering: rows are rasterized into a small buffer in fast memory,
// then copied to the target in one sequential pass, so a framebuffer in slow
// (e.g. write-through SDRAM) memory takes each pixel once, in order.
struct Bands {
	Pixel *pixels = nullptr;
	// A second buffer where bands are copied out in the background: the
	// raster fills one while the other drains.
	Pixel *spare = nullptr;
	int width = 0;
	int rows = 0;
	// Per band of the region being drawn, its primitives (buildBandLists).
	std::vector<std::int32_t> items;
	std::vector<int> starts;
	std::vector<int> opaque;
	std::vector<int> next;
	std::vector<int> bandOfRow;
};

// A clipped polygon corner (view space + attributes).
struct Corner {
	float x, y, z;
	float r, g, b;
};

struct Renderer {
	raster::Frame frames[2];
	int building = 0;
	raster::Scratch scratch;
	int capacity = kDefaultCapacity;
	bool depthBuffer = false;
	int surface = -1;
	View view;
	Lights lights;
	Environment environment;
	Stats stats;
	VertexScratch vertices;
	Bands bands;
	std::vector<int> stack;
	std::vector<int> drawables;
	std::vector<int> lightNodes;
	std::vector<std::int32_t> keys;
	std::vector<std::int32_t> bucketStarts;
	std::vector<std::int32_t> layerCounts;
	std::vector<std::int32_t> layerOrder;
	bool legacyActive = false;

	raster::Frame &frame() { return frames[building]; }
};

Renderer g_renderer;

// Fast memory for the raster. On the STM32F7 the heap and framebuffers are
// SDRAM (framebuffers write-through), so two band buffers fill the zero-wait
// DTCM beside the stack (never zeroed; the rasterizer initialises what it
// uses) while DMA2D drains the other, and the vertex scratch and primitive
// records live in internal SRAM. Bands draw painter's-order, so they need no
// coverage bookkeeping. GEA3D_BAND_ROWS lets host tests exercise bands.
#if defined(STM32F746xx) && !defined(GEA3D_BAND_ROWS)
#define GEA3D_BAND_ROWS 15
#define GEA3D_BAND_BUFFERS 2
#define GEA3D_FAST_MEMORY __attribute__((section(".dtcm.gea3d"), aligned(32)))
#define GEA3D_VERTEX_POOL 384
#define GEA3D_PRIM_POOL 1024
#elif defined(GEA3D_BAND_ROWS)
#define GEA3D_BAND_BUFFERS 1
#define GEA3D_FAST_MEMORY alignas(32)
#define GEA3D_VERTEX_POOL 384
#define GEA3D_PRIM_POOL 1024
#endif

#if defined(GEA3D_BAND_ROWS)
constexpr int kBandWidth = 480;
constexpr int kBandRows = GEA3D_BAND_ROWS;

GEA3D_FAST_MEMORY Pixel g_bandPixels[GEA3D_BAND_BUFFERS][kBandWidth * kBandRows];
Vertex g_vertexPool[GEA3D_VERTEX_POOL];
Prim g_primPool[GEA3D_PRIM_POOL];

struct FastMemoryBinding {
	FastMemoryBinding()
	{
		Renderer &r = g_renderer;
		r.bands.pixels = g_bandPixels[0];
		r.bands.spare = GEA3D_BAND_BUFFERS > 1 && GEA3D_DMA2D_COPY ? g_bandPixels[GEA3D_BAND_BUFFERS - 1] : nullptr;
		r.bands.width = kBandWidth;
		r.bands.rows = kBandRows;
		r.vertices.fixed = g_vertexPool;
		r.vertices.fixedCount = GEA3D_VERTEX_POOL;
		// Only frame 0: no asynchronous present where fast memory is bound.
		r.frames[0].fixedPrims = g_primPool;
		r.frames[0].fixedCapacity = GEA3D_PRIM_POOL;
	}
};

FastMemoryBinding g_fastMemoryBinding;
#endif

struct Surfaces {
	int ids[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
	Pixel *pixels[8] = {};
	int width[8] = {};
	int height[8] = {};
	int count = 0;

	int find(int id) const
	{
		for (int i = 0; i < count; i++) {
			if (ids[i] == id) return i;
		}
		return -1;
	}
};

Surfaces g_surfaces;

struct RayHit {
	float distance = 0.0f;
	float point[3] = {0, 0, 0};
	int face = -1;
	int instance = -1;
};

RayHit g_rayHit;

// ---------------------------------------------------------------- helpers

// VMAXNM on FPv5: no float compare stalls the pipeline.
GEA3D_INLINE float maxFloat(float a, float b) { return __builtin_fmaxf(a, b); }

// 0..255 by truncation; clamped in integer registers (USAT), not with float
// compares.
GEA3D_INLINE std::uint8_t toByte(float value)
{
	return static_cast<std::uint8_t>(raster::detail::clampInt(static_cast<int>(value), 0, 255));
}

void hexToRgb(double hex, std::uint8_t *out)
{
	const std::uint32_t value = static_cast<std::uint32_t>(hex);
	out[0] = static_cast<std::uint8_t>((value >> 16) & 0xFF);
	out[1] = static_cast<std::uint8_t>((value >> 8) & 0xFF);
	out[2] = static_cast<std::uint8_t>(value & 0xFF);
}

const Buffer *bufferAt(int id)
{
	if (id < 0 || id >= static_cast<int>(g_buffers.size()) || !g_buffers[static_cast<std::size_t>(id)].live) return nullptr;
	return &g_buffers[static_cast<std::size_t>(id)];
}

void updateBounds(Geometry &geometry)
{
	const Buffer *positions = bufferAt(geometry.position);
	if (!positions) {
		geometry.radius = 0.0f;
		return;
	}
	if (geometry.boundsBuffer == geometry.position && geometry.boundsVersion == positions->version && geometry.radius >= 0.0f) return;

	const std::size_t count = positions->f32.size() / 3;
	const float *p = positions->f32.data();
	geometry.boundsBuffer = geometry.position;
	geometry.boundsVersion = positions->version;
	if (count == 0) {
		geometry.center[0] = geometry.center[1] = geometry.center[2] = 0.0f;
		geometry.radius = 0.0f;
		return;
	}

	float lo[3] = {p[0], p[1], p[2]};
	float hi[3] = {p[0], p[1], p[2]};
	for (std::size_t i = 1; i < count; i++) {
		for (int k = 0; k < 3; k++) {
			lo[k] = std::min(lo[k], p[i * 3 + k]);
			hi[k] = std::max(hi[k], p[i * 3 + k]);
		}
	}
	for (int k = 0; k < 3; k++) geometry.center[k] = (lo[k] + hi[k]) * 0.5f;
	float maxSq = 0.0f;
	for (std::size_t i = 0; i < count; i++) {
		const float dx = p[i * 3] - geometry.center[0];
		const float dy = p[i * 3 + 1] - geometry.center[1];
		const float dz = p[i * 3 + 2] - geometry.center[2];
		maxSq = std::max(maxSq, dx * dx + dy * dy + dz * dz);
	}
	geometry.radius = std::sqrt(maxSq);
}

Surface resolveSurface(const Material &material)
{
	Surface surface;
	surface.shade = material.shade;
	for (int k = 0; k < 3; k++) surface.color[k] = material.color[k];
	surface.side = material.side;
	surface.vertexColors = (material.flags & kMaterialVertexColors) != 0;
	surface.additive = (material.flags & kMaterialAdditive) != 0;
	surface.blend = surface.additive || ((material.flags & kMaterialTransparent) != 0 && material.opacity < 1.0f);
	surface.alpha = toByte(material.opacity * 255.0f);
	surface.fog = (material.flags & kMaterialFog) != 0;
	surface.flat = (material.flags & kMaterialFlatShading) != 0;
	surface.wireframe = (material.flags & kMaterialWireframe) != 0;
	surface.size = material.size;
	surface.sizeAttenuation = (material.flags & kMaterialSizeAttenuation) != 0;
	return surface;
}

// ---------------------------------------------------------------- lighting

void lightNormal(const float *n, const float *p, float *out)
{
	const Lights &lights = g_renderer.lights;
	float r = lights.ambient[0], g = lights.ambient[1], b = lights.ambient[2];
	for (int i = 0; i < lights.directionalCount; i++) {
		const float *d = lights.direction[i];
		const float dot = n[0] * d[0] + n[1] * d[1] + n[2] * d[2];
		if (dot <= 0.0f) continue;
		r += dot * lights.directionalColor[i][0];
		g += dot * lights.directionalColor[i][1];
		b += dot * lights.directionalColor[i][2];
	}
	for (int i = 0; i < lights.hemisphereCount; i++) {
		const float *up = lights.hemisphereUp[i];
		const float t = (n[0] * up[0] + n[1] * up[1] + n[2] * up[2]) * 0.5f + 0.5f;
		r += lights.ground[i][0] + (lights.sky[i][0] - lights.ground[i][0]) * t;
		g += lights.ground[i][1] + (lights.sky[i][1] - lights.ground[i][1]) * t;
		b += lights.ground[i][2] + (lights.sky[i][2] - lights.ground[i][2]) * t;
	}
	for (int i = 0; i < lights.pointCount; i++) {
		float l[3] = {lights.pointPosition[i][0] - p[0], lights.pointPosition[i][1] - p[1], lights.pointPosition[i][2] - p[2]};
		const float distance = std::sqrt(l[0] * l[0] + l[1] * l[1] + l[2] * l[2]);
		if (distance <= 0.0f) continue;
		const float inv = 1.0f / distance;
		const float dot = (n[0] * l[0] + n[1] * l[1] + n[2] * l[2]) * inv;
		if (dot <= 0.0f) continue;
		float falloff = lights.pointDecay[i] > 0.0f ? 1.0f / std::max(std::pow(distance, lights.pointDecay[i]), 0.01f) : 1.0f;
		if (lights.pointDistance[i] > 0.0f) {
			const float ratio = distance / lights.pointDistance[i];
			const float window = std::clamp(1.0f - ratio * ratio * ratio * ratio, 0.0f, 1.0f);
			falloff *= window * window;
		}
		r += dot * falloff * lights.pointColor[i][0];
		g += dot * falloff * lights.pointColor[i][1];
		b += dot * falloff * lights.pointColor[i][2];
	}
	out[0] = r;
	out[1] = g;
	out[2] = b;
}

GEA3D_INLINE float fogFactor(float distance)
{
	const Environment &env = g_renderer.environment;
	if (env.fog == 1) {
		if (env.fogFar <= env.fogNear) return 1.0f;
		return clampFloat((env.fogFar - distance) / (env.fogFar - env.fogNear), 0.0f, 1.0f);
	}
	if (env.fog == 2) {
		const float d = env.fogDensity * distance;
		return clampFloat(std::exp(-d * d), 0.0f, 1.0f);
	}
	return 1.0f;
}

// Final 0..255 colour of a lit, fogged corner.
GEA3D_INLINE void shadeCorner(const float *base, const float *light, float fog, float *out)
{
	const Environment &env = g_renderer.environment;
	for (int k = 0; k < 3; k++) out[k] = base[k] * light[k];
	if (fog >= 1.0f) return;

	for (int k = 0; k < 3; k++) out[k] = env.fogColor[k] + (out[k] - env.fogColor[k]) * fog;
}

// MeshNormalMaterial: the view-space normal as a colour.
void normalColor(const float *n, float *out)
{
	for (int k = 0; k < 3; k++) out[k] = (n[k] * 0.5f + 0.5f) * 255.0f;
}

// ---------------------------------------------------------------- projection

GEA3D_INLINE void project(const View &v, float x, float y, float z, std::int32_t *sx, std::int32_t *sy, float *depth)
{
	const float *p = v.projection;
	const float cx = p[0] * x + p[4] * y + p[8] * z + p[12];
	const float cy = p[1] * x + p[5] * y + p[9] * z + p[13];
	float cw = p[3] * x + p[7] * y + p[11] * z + p[15];
	if (cw < 0.000001f && cw > -0.000001f) cw = 0.000001f;
	const float invW = 1.0f / cw;
	const float fx = clampFloat((cx * invW + 1.0f) * v.halfW, -kCoordLimit, kCoordLimit);
	const float fy = clampFloat((1.0f - cy * invW) * v.halfH, -kCoordLimit, kCoordLimit);
	*sx = roundToInt(fx * 16.0f);
	*sy = roundToInt(fy * 16.0f);
	if (v.orthographic) {
		const float range = v.far - v.near;
		*depth = range > 0.0f ? clampFloat(1.0f - (-z - v.near) / range, 0.0f, 1.0f) : 1.0f;
	} else {
		*depth = clampFloat(v.near * invW, 0.0f, 1.0f);
	}
}

// ---------------------------------------------------------------- emission

// Room for more primitives: the fixed pool first, then the heap (doubling;
// whatever the pool already holds moves along).
void growPrims(raster::Frame &frame)
{
	int capacity = 0;
	if (!frame.prims && frame.fixedPrims) {
		frame.prims = frame.fixedPrims;
		capacity = frame.fixedCapacity;
	} else {
		capacity = maxInt(256, frame.primCapacity * 2);
		const bool fromPool = frame.prims == frame.fixedPrims;
		frame.primStore.resize(static_cast<std::size_t>(capacity));
		if (fromPool && frame.count > 0) std::memcpy(frame.primStore.data(), frame.prims, static_cast<std::size_t>(frame.count) * sizeof(Prim));
		frame.prims = frame.primStore.data();
	}
	frame.primCapacity = capacity;
	frame.sortKeys.resize(static_cast<std::size_t>(capacity));
	frame.blended.resize(static_cast<std::size_t>(capacity));
	frame.layers.resize(static_cast<std::size_t>(capacity));
	frame.emittedRows.resize(static_cast<std::size_t>(capacity));
}

GEA3D_INLINE Prim *nextPrim()
{
	raster::Frame &frame = g_renderer.frame();
	if (frame.count >= g_renderer.capacity) {
		g_renderer.stats.dropped++;
		return nullptr;
	}
	if (frame.count >= frame.primCapacity) growPrims(frame);
	return &frame.prims[frame.count++];
}

GEA3D_INLINE void finishPrim(Prim &prim, const Surface &surface, int corners)
{
	prim.alpha = surface.alpha;
	prim.flags = surface.blend ? raster::kBlend : 0;
	if (surface.additive) prim.flags |= raster::kAdd;
	// Corners that pack to one pixel colour make a flat primitive: every
	// colour between them packs to that pixel too.
	prim.color = raster::detail::packColor(prim.r[0], prim.g[0], prim.b[0]);
	bool uniform = true;
	for (int i = 1; i < corners; i++) uniform = uniform && raster::detail::packColor(prim.r[i], prim.g[i], prim.b[i]) == prim.color;
	if (!uniform) prim.flags |= raster::kSmooth;

	int lo = static_cast<int>(prim.y[0]), hi = lo;
	for (int i = 1; i < corners; i++) {
		lo = minInt(lo, static_cast<int>(prim.y[i]));
		hi = maxInt(hi, static_cast<int>(prim.y[i]));
	}
	// A wide line reaches past its ends' rows.
	const int margin = 1 + (prim.type == raster::kLine ? prim.width >> 1 : 0);
	prim.rowY0 = static_cast<std::int16_t>((lo >> 4) - margin);
	prim.rowY1 = static_cast<std::int16_t>((hi >> 4) + margin);

	raster::Frame &frame = g_renderer.frame();
	const std::size_t index = static_cast<std::size_t>(&prim - frame.prims);
	frame.sortKeys[index] = prim.key;
	frame.blended[index] = surface.blend ? 1 : 0;
	frame.layers[index] = surface.layer;
	frame.minLayer = minInt(frame.minLayer, surface.layer);
	frame.maxLayer = maxInt(frame.maxLayer, surface.layer);
	frame.emittedRows[index] = static_cast<std::uint16_t>(prim.rowY0) | (static_cast<std::uint32_t>(static_cast<std::uint16_t>(prim.rowY1)) << 16);
}

// Whether a primitive with these screen corners (28.4) may cover a pixel of
// the frame, give or take `margin` pixels.
GEA3D_INLINE bool onScreen(const std::int32_t *sx, const std::int32_t *sy, int corners, int margin)
{
	int xl = sx[0], xh = xl, yl = sy[0], yh = yl;
	for (int i = 1; i < corners; i++) {
		xl = minInt(xl, sx[i]);
		xh = maxInt(xh, sx[i]);
		yl = minInt(yl, sy[i]);
		yh = maxInt(yh, sy[i]);
	}
	const raster::Frame &frame = g_renderer.frame();
	const int edge = margin * 16;
	return xh >= -edge && yh >= -edge && xl <= (frame.width + margin) * 16 && yl <= (frame.height + margin) * 16;
}

void emitTriangle(const Surface &surface, const Corner *c, const std::int32_t *sx, const std::int32_t *sy, const float *depth, float key)
{
	if (!onScreen(sx, sy, 3, 0)) {
		g_renderer.stats.culled++;
		return;
	}

	Prim *prim = nextPrim();
	if (!prim) return;

	prim->type = raster::kTriangle;
	for (int i = 0; i < 3; i++) {
		prim->x[i] = sx[i];
		prim->y[i] = sy[i];
		prim->depth[i] = depth[i];
		prim->r[i] = toByte(c[i].r);
		prim->g[i] = toByte(c[i].g);
		prim->b[i] = toByte(c[i].b);
	}
	prim->key = key;
	finishPrim(*prim, surface, 3);
}

void emitLine(const Surface &surface, const Corner &a, const Corner &b)
{
	const float nearZ = -g_renderer.view.near;
	Corner p = a, q = b;
	if (p.z > nearZ && q.z > nearZ) {
		g_renderer.stats.nearRejected++;
		return;
	}
	if (p.z > nearZ || q.z > nearZ) {
		Corner &out = p.z > nearZ ? p : q;
		const Corner &in = p.z > nearZ ? q : p;
		const float t = (nearZ - in.z) / (out.z - in.z);
		out.x = in.x + (out.x - in.x) * t;
		out.y = in.y + (out.y - in.y) * t;
		out.z = nearZ;
		out.r = in.r + (out.r - in.r) * t;
		out.g = in.g + (out.g - in.g) * t;
		out.b = in.b + (out.b - in.b) * t;
	}

	std::int32_t sx[2], sy[2];
	float depth[2];
	project(g_renderer.view, p.x, p.y, p.z, &sx[0], &sy[0], &depth[0]);
	project(g_renderer.view, q.x, q.y, q.z, &sx[1], &sy[1], &depth[1]);
	const int width = raster::detail::clampInt(roundToInt(surface.size), 1, 3);
	if (!onScreen(sx, sy, 2, width)) {
		g_renderer.stats.culled++;
		return;
	}

	Prim *prim = nextPrim();
	if (!prim) return;

	prim->type = raster::kLine;
	prim->width = static_cast<std::uint8_t>(width);
	for (int i = 0; i < 2; i++) {
		prim->x[i] = sx[i];
		prim->y[i] = sy[i];
		prim->depth[i] = depth[i];
	}
	prim->x[2] = prim->x[1];
	prim->y[2] = prim->y[1];
	prim->depth[2] = prim->depth[1];
	prim->r[0] = toByte(p.r);
	prim->g[0] = toByte(p.g);
	prim->b[0] = toByte(p.b);
	prim->r[1] = prim->r[2] = toByte(q.r);
	prim->g[1] = prim->g[2] = toByte(q.g);
	prim->b[1] = prim->b[2] = toByte(q.b);
	prim->key = std::max(-p.z, -q.z);
	finishPrim(*prim, surface, 2);
}

// Clips a triangle against the near plane (view space) and fans it out.
void emitClipped(const Surface &surface, const Corner *corners)
{
	const float nearZ = -g_renderer.view.near;
	Corner polygon[4];
	int count = 0;
	for (int e = 0; e < 3; e++) {
		const Corner &a = corners[e];
		const Corner &b = corners[(e + 1) % 3];
		const bool aIn = a.z <= nearZ;
		const bool bIn = b.z <= nearZ;
		if (aIn) polygon[count++] = a;
		if (aIn != bIn) {
			const float t = (nearZ - a.z) / (b.z - a.z);
			Corner &c = polygon[count++];
			c.x = a.x + (b.x - a.x) * t;
			c.y = a.y + (b.y - a.y) * t;
			c.z = nearZ;
			c.r = a.r + (b.r - a.r) * t;
			c.g = a.g + (b.g - a.g) * t;
			c.b = a.b + (b.b - a.b) * t;
		}
	}
	if (count < 3) {
		g_renderer.stats.clipDropped++;
		return;
	}

	std::int32_t sx[4], sy[4];
	float depth[4];
	float key = 0.0f;
	for (int i = 0; i < count; i++) {
		project(g_renderer.view, polygon[i].x, polygon[i].y, polygon[i].z, &sx[i], &sy[i], &depth[i]);
		key = maxFloat(key, -polygon[i].z);
	}
	for (int i = 2; i < count; i++) {
		const Corner fan[3] = {polygon[0], polygon[i - 1], polygon[i]};
		const std::int32_t fx[3] = {sx[0], sx[i - 1], sx[i]};
		const std::int32_t fy[3] = {sy[0], sy[i - 1], sy[i]};
		const float fd[3] = {depth[0], depth[i - 1], depth[i]};
		emitTriangle(surface, fan, fx, fy, fd, key);
	}
}

// Whether a mesh draw shades per face (flat shading, or lit without vertex
// normals); otherwise every corner takes its vertex's colour.
GEA3D_INLINE bool shadesPerFace(const Surface &surface, bool smoothLit)
{
	return !surface.wireframe && !smoothLit && (surface.shade != 0 || surface.flat);
}

// Transforms, lights and projects every vertex of a draw once. Unless faces
// are shaded, each vertex also gets its final (lit, fogged) colour.
void transformVertices(const float *mv, const float *positions, std::size_t count, const float *colors, const float *normals, const Surface &surface, const float *tint, bool smoothLit, bool perVertex)
{
	Vertex *vertices = g_renderer.vertices.prepare(count);
	// Local copies: the compiler keeps them in registers, as vertex stores
	// cannot alias them.
	const View view = g_renderer.view;
	float m[16];
	std::memcpy(m, mv, sizeof(m));
	const float nearZ = -view.near;
	const bool fogged = surface.fog && g_renderer.environment.fog != 0;
	const float tinted[3] = {surface.color[0] * tint[0], surface.color[1] * tint[1], surface.color[2] * tint[2]};

	float normalMatrix[16];
	if (normals) {
		float inverse[16];
		math::invertAffine(inverse, mv);
		// Transpose of the inverse's 3x3.
		normalMatrix[0] = inverse[0];
		normalMatrix[4] = inverse[1];
		normalMatrix[8] = inverse[2];
		normalMatrix[1] = inverse[4];
		normalMatrix[5] = inverse[5];
		normalMatrix[9] = inverse[6];
		normalMatrix[2] = inverse[8];
		normalMatrix[6] = inverse[9];
		normalMatrix[10] = inverse[10];
	}

	for (std::size_t i = 0; i < count; i++) {
		// The attribute buffers live in slow memory: fetch a few vertices ahead.
		__builtin_prefetch(positions + (i + 8) * 3);
		if (colors) __builtin_prefetch(colors + (i + 8) * 3);
		Vertex &vertex = vertices[i];
		float *p = vertex.view;
		math::transformPoint(m, positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2], p);
		vertex.front = p[2] <= nearZ;
		if (vertex.front) project(view, p[0], p[1], p[2], &vertex.screen[0], &vertex.screen[1], &vertex.depth);

		float base[3] = {tinted[0], tinted[1], tinted[2]};
		if (colors) {
			base[0] *= colors[i * 3];
			base[1] *= colors[i * 3 + 1];
			base[2] *= colors[i * 3 + 2];
		}
		const float fog = fogged ? fogFactor(-p[2]) : 1.0f;
		if (!perVertex) {
			for (int k = 0; k < 3; k++) vertex.color[k] = base[k];
			vertex.fog = fog;
			continue;
		}

		float light[3] = {1.0f, 1.0f, 1.0f};
		if (smoothLit) {
			float n[3];
			math::transformDirection(normalMatrix, normals[i * 3], normals[i * 3 + 1], normals[i * 3 + 2], n);
			math::normalize3(n);
			if (surface.shade == 2) normalColor(n, base);
			else lightNormal(n, p, light);
		}
		shadeCorner(base, light, fog, vertex.color);
		for (int k = 0; k < 3; k++) vertex.rgb[k] = toByte(vertex.color[k]);
		vertex.fog = 1.0f;
	}
}

// A triangle whose corners are all in front of the near plane and shaded
// per vertex: its record comes straight from the vertices.
void emitVertexTriangle(const Surface &surface, const Vertex &a, const Vertex &b, const Vertex &c)
{
	Prim *prim = nextPrim();
	if (!prim) return;

	prim->type = raster::kTriangle;
	const Vertex *corners[3] = {&a, &b, &c};
	for (int i = 0; i < 3; i++) {
		const Vertex &v = *corners[i];
		prim->x[i] = v.screen[0];
		prim->y[i] = v.screen[1];
		prim->depth[i] = v.depth;
		prim->r[i] = v.rgb[0];
		prim->g[i] = v.rgb[1];
		prim->b[i] = v.rgb[2];
	}
	prim->key = maxFloat(-a.view[2], maxFloat(-b.view[2], -c.view[2]));
	finishPrim(*prim, surface, 3);
}

GEA3D_INLINE Corner cornerOf(const Vertex &vertex, const float *rgb)
{
	const float *p = vertex.view;
	return Corner{p[0], p[1], p[2], rgb[0], rgb[1], rgb[2]};
}

void emitTriangles(const Surface &surface, std::size_t vertexCount, const std::uint32_t *indices, std::size_t indexCount, bool smoothLit)
{
	const Vertex *v = g_renderer.vertices.current;
	Stats &stats = g_renderer.stats;
	const bool orthographic = g_renderer.view.orthographic;
	const std::size_t triangles = (indices ? indexCount : vertexCount) / 3;
	const bool perFace = shadesPerFace(surface, smoothLit);
	// Only culling and face lighting need to know which way a face points.
	const bool needsFacing = surface.side != 2 || perFace || surface.wireframe;
	stats.submitted += static_cast<int>(triangles);

	for (std::size_t t = 0; t < triangles; t++) {
		std::uint32_t id[3];
		for (int k = 0; k < 3; k++) id[k] = indices ? indices[t * 3 + k] : static_cast<std::uint32_t>(t * 3 + k);
		if (id[0] >= vertexCount || id[1] >= vertexCount || id[2] >= vertexCount) continue;

		const Vertex &v0 = v[id[0]], &v1 = v[id[1]], &v2 = v[id[2]];
		const float *p0 = v0.view;
		const float *p1 = v1.view;
		const float *p2 = v2.view;
		if ((v0.front | v1.front | v2.front) == 0) {
			stats.nearRejected++;
			continue;
		}

		const bool inFront = (v0.front & v1.front & v2.front) != 0;
		if (inFront && !surface.wireframe) {
			const std::int32_t sx[3] = {v0.screen[0], v1.screen[0], v2.screen[0]};
			const std::int32_t sy[3] = {v0.screen[1], v1.screen[1], v2.screen[1]};
			if (!onScreen(sx, sy, 3, 0)) {
				stats.culled++;
				continue;
			}
		}

		float n[3] = {0, 0, 0};
		bool back = false;
		if (needsFacing) {
			const float e1[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
			const float e2[3] = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
			n[0] = e1[1] * e2[2] - e1[2] * e2[1];
			n[1] = e1[2] * e2[0] - e1[0] * e2[2];
			n[2] = e1[0] * e2[1] - e1[1] * e2[0];
			const float facing = orthographic ? -n[2] : n[0] * p0[0] + n[1] * p0[1] + n[2] * p0[2];
			back = facing >= 0.0f;
		}

		if (surface.wireframe) {
			const Corner c[3] = {cornerOf(v0, v0.color), cornerOf(v1, v1.color), cornerOf(v2, v2.color)};
			for (int k = 0; k < 3; k++) emitLine(surface, c[k], c[(k + 1) % 3]);
			continue;
		}

		if ((surface.side == 0 && back) || (surface.side == 1 && !back)) {
			stats.culled++;
			continue;
		}

		if (!perFace) {
			if (inFront) {
				emitVertexTriangle(surface, v0, v1, v2);
				continue;
			}
			const Corner c[3] = {cornerOf(v0, v0.color), cornerOf(v1, v1.color), cornerOf(v2, v2.color)};
			emitClipped(surface, c);
			continue;
		}

		if (back) {
			n[0] = -n[0];
			n[1] = -n[1];
			n[2] = -n[2];
		}

		float faceLight[3] = {1, 1, 1};
		float faceBase[3] = {0, 0, 0};
		const bool faceNormalColor = surface.shade == 2;
		if (surface.shade != 0 && math::normalize3(n)) {
			if (faceNormalColor) {
				normalColor(n, faceBase);
			} else {
				const float centroid[3] = {(p0[0] + p1[0] + p2[0]) / 3.0f, (p0[1] + p1[1] + p2[1]) / 3.0f, (p0[2] + p1[2] + p2[2]) / 3.0f};
				lightNormal(n, centroid, faceLight);
			}
		}

		float faceBaseAverage[3] = {0, 0, 0};
		if (surface.flat && !faceNormalColor) {
			for (int j = 0; j < 3; j++) faceBaseAverage[j] = v0.color[j] * (1.0f / 3.0f);
			for (int j = 0; j < 3; j++) faceBaseAverage[j] += v1.color[j] * (1.0f / 3.0f);
			for (int j = 0; j < 3; j++) faceBaseAverage[j] += v2.color[j] * (1.0f / 3.0f);
		}
		Corner c[3];
		const Vertex *corners[3] = {&v0, &v1, &v2};
		for (int k = 0; k < 3; k++) {
			const Vertex &vertex = *corners[k];
			const float *base = faceNormalColor ? faceBase : (surface.flat ? faceBaseAverage : vertex.color);
			float rgb[3];
			shadeCorner(base, faceLight, vertex.fog, rgb);
			c[k] = cornerOf(vertex, rgb);
		}

		if (inFront) {
			const std::int32_t sx[3] = {v0.screen[0], v1.screen[0], v2.screen[0]};
			const std::int32_t sy[3] = {v0.screen[1], v1.screen[1], v2.screen[1]};
			const float depth[3] = {v0.depth, v1.depth, v2.depth};
			const float key = maxFloat(-p0[2], maxFloat(-p1[2], -p2[2]));
			emitTriangle(surface, c, sx, sy, depth, key);
		} else {
			emitClipped(surface, c);
		}
	}
}

void emitLines(const Surface &surface, std::size_t vertexCount, const std::uint32_t *indices, std::size_t indexCount, bool strip)
{
	const Vertex *v = g_renderer.vertices.current;
	const std::size_t count = indices ? indexCount : vertexCount;
	const std::size_t segments = strip ? (count > 0 ? count - 1 : 0) : count / 2;
	for (std::size_t s = 0; s < segments; s++) {
		const std::size_t a = strip ? s : s * 2;
		const std::uint32_t ia = indices ? indices[a] : static_cast<std::uint32_t>(a);
		const std::uint32_t ib = indices ? indices[a + 1] : static_cast<std::uint32_t>(a + 1);
		if (ia >= vertexCount || ib >= vertexCount) continue;

		emitLine(surface, cornerOf(v[ia], v[ia].color), cornerOf(v[ib], v[ib].color));
	}
}

// A screen-aligned square of `sizePx` around a view-space point.
void emitQuad(const Surface &surface, const float *p, float halfX, float halfY, const float *rgb, bool screenSpace)
{
	const float nearZ = -g_renderer.view.near;
	if (p[2] > nearZ) {
		g_renderer.stats.nearRejected++;
		return;
	}

	std::int32_t sx[4], sy[4];
	float depth[4];
	if (screenSpace) {
		std::int32_t cx = 0, cy = 0;
		float d = 0.0f;
		project(g_renderer.view, p[0], p[1], p[2], &cx, &cy, &d);
		const std::int32_t hx = static_cast<std::int32_t>(halfX * 16.0f);
		const std::int32_t hy = static_cast<std::int32_t>(halfY * 16.0f);
		const std::int32_t xs[4] = {cx - hx, cx + hx, cx + hx, cx - hx};
		const std::int32_t ys[4] = {cy - hy, cy - hy, cy + hy, cy + hy};
		for (int i = 0; i < 4; i++) {
			sx[i] = xs[i];
			sy[i] = ys[i];
			depth[i] = d;
		}
	} else {
		const float xs[4] = {p[0] - halfX, p[0] + halfX, p[0] + halfX, p[0] - halfX};
		const float ys[4] = {p[1] + halfY, p[1] + halfY, p[1] - halfY, p[1] - halfY};
		for (int i = 0; i < 4; i++) project(g_renderer.view, xs[i], ys[i], p[2], &sx[i], &sy[i], &depth[i]);
	}

	const Corner c{p[0], p[1], p[2], rgb[0], rgb[1], rgb[2]};
	const Corner corners[3] = {c, c, c};
	const float key = -p[2];
	const std::int32_t ax[3] = {sx[0], sx[1], sx[2]}, ay[3] = {sy[0], sy[1], sy[2]};
	const float ad[3] = {depth[0], depth[1], depth[2]};
	emitTriangle(surface, corners, ax, ay, ad, key);
	const std::int32_t bx[3] = {sx[0], sx[2], sx[3]}, by[3] = {sy[0], sy[2], sy[3]};
	const float bd[3] = {depth[0], depth[2], depth[3]};
	emitTriangle(surface, corners, bx, by, bd, key);
}

void emitPoints(const Surface &surface, std::size_t vertexCount)
{
	const Vertex *v = g_renderer.vertices.current;
	const View &view = g_renderer.view;
	for (std::size_t i = 0; i < vertexCount; i++) {
		const float *p = v[i].view;
		float size = surface.size;
		if (surface.sizeAttenuation && !view.orthographic && p[2] < 0.0f) size = surface.size * view.halfH / -p[2];
		const float half = std::max(size * 0.5f, 0.5f);
		emitQuad(surface, p, half, half, v[i].color, true);
	}
}

// ---------------------------------------------------------------- draws

// One draw of a node's geometry through modelView `mv` (instance `tint`).
void drawGeometry(int kind, const Geometry &geometry, const Surface &surface, const float *mv, const float *tint)
{
	const Buffer *positions = bufferAt(geometry.position);
	if (!positions || positions->f32.size() < 3) return;

	const std::size_t vertexCount = positions->f32.size() / 3;
	const Buffer *indexBuffer = bufferAt(geometry.index);
	const std::uint32_t *indices = indexBuffer && !indexBuffer->u32.empty() ? indexBuffer->u32.data() : nullptr;
	const std::size_t indexCount = indices ? indexBuffer->u32.size() : 0;
	const Buffer *colorBuffer = surface.vertexColors ? bufferAt(geometry.color) : nullptr;
	const float *colors = colorBuffer && colorBuffer->f32.size() >= vertexCount * 3 ? colorBuffer->f32.data() : nullptr;
	const Buffer *normalBuffer = bufferAt(geometry.normal);
	const float *normals = normalBuffer && normalBuffer->f32.size() >= vertexCount * 3 ? normalBuffer->f32.data() : nullptr;
	const bool triangles = kind == kNodeMesh;
	const bool smoothLit = triangles && normals && !surface.flat && surface.shade != 0 && !surface.wireframe;

	const bool perVertex = !triangles || !shadesPerFace(surface, smoothLit);
	transformVertices(mv, positions->f32.data(), vertexCount, colors, normals, surface, tint, smoothLit, perVertex);
	if (kind == kNodeMesh) emitTriangles(surface, vertexCount, indices, indexCount, smoothLit);
	else if (kind == kNodeLineSegments) emitLines(surface, vertexCount, indices, indexCount, false);
	else if (kind == kNodeLine) emitLines(surface, vertexCount, indices, indexCount, true);
	else if (kind == kNodePoints) emitPoints(surface, vertexCount);
}

bool sphereVisible(const Geometry &geometry, const float *mv)
{
	float center[3];
	math::transformPoint(mv, geometry.center[0], geometry.center[1], geometry.center[2], center);
	const float radius = geometry.radius * math::maxScaleOnAxis(mv);
	return math::sphereInFrustum(g_renderer.view.planes, center, radius);
}

void drawSprite(const Surface &surface, const float *mv)
{
	float center[3];
	math::transformPoint(mv, 0.0f, 0.0f, 0.0f, center);
	const float sx = std::sqrt(mv[0] * mv[0] + mv[1] * mv[1] + mv[2] * mv[2]);
	const float sy = std::sqrt(mv[4] * mv[4] + mv[5] * mv[5] + mv[6] * mv[6]);
	if (!math::sphereInFrustum(g_renderer.view.planes, center, std::max(sx, sy))) {
		g_renderer.stats.meshesCulled++;
		return;
	}

	const float base[3] = {surface.color[0], surface.color[1], surface.color[2]};
	const float unlit[3] = {1, 1, 1};
	float rgb[3];
	shadeCorner(base, unlit, surface.fog && g_renderer.environment.fog != 0 ? fogFactor(-center[2]) : 1.0f, rgb);
	emitQuad(surface, center, sx * 0.5f, sy * 0.5f, rgb, false);
	g_renderer.stats.meshesDrawn++;
}

void drawNode(const Node &node)
{
	Material *material = lookup(g_materials, node.material);
	if (!material || !(material->flags & kMaterialVisible)) return;

	Surface surface = resolveSurface(*material);
	surface.layer = node.renderOrder;
	float mv[16];
	if (node.kind == kNodeSprite) {
		math::multiplyAffine(mv, g_renderer.view.view, node.world);
		drawSprite(surface, mv);
		return;
	}

	Geometry *geometry = lookup(g_geometries, node.geometry);
	if (!geometry) return;

	updateBounds(*geometry);
	const float white[3] = {1, 1, 1};
	if (!node.instanced) {
		math::multiplyAffine(mv, g_renderer.view.view, node.world);
		if (node.frustumCulled && !sphereVisible(*geometry, mv)) {
			g_renderer.stats.meshesCulled++;
			return;
		}
		drawGeometry(node.kind, *geometry, surface, mv, white);
		g_renderer.stats.meshesDrawn++;
		return;
	}

	const Buffer *instances = bufferAt(node.instanceMatrices);
	if (!instances) return;

	const Buffer *tints = bufferAt(node.instanceColors);
	const int count = std::min(node.instanceCount, static_cast<int>(instances->f32.size() / 16));
	float viewWorld[16];
	math::multiplyAffine(viewWorld, g_renderer.view.view, node.world);
	for (int i = 0; i < count; i++) {
		math::multiplyAffine(mv, viewWorld, &instances->f32[static_cast<std::size_t>(i) * 16]);
		if (node.frustumCulled && !sphereVisible(*geometry, mv)) {
			g_renderer.stats.meshesCulled++;
			continue;
		}
		const float *tint = tints && tints->f32.size() >= static_cast<std::size_t>(i + 1) * 3 ? &tints->f32[static_cast<std::size_t>(i) * 3] : white;
		drawGeometry(node.kind, *geometry, surface, mv, tint);
		g_renderer.stats.meshesDrawn++;
	}
}

// ---------------------------------------------------------------- frame setup

void beginFrameState(int width, int height)
{
	Renderer &r = g_renderer;
	raster::Frame &frame = r.frame();
	frame.count = 0;
	frame.opaqueCount = 0;
	frame.blendCount = 0;
	frame.minLayer = INT_MAX;
	frame.maxLayer = INT_MIN;
	frame.width = width;
	frame.height = height;
	frame.depthBuffer = r.depthBuffer;
	r.stats = Stats();
	r.lights.reset();
	r.view.width = width;
	r.view.height = height;
	r.view.halfW = static_cast<float>(width) * 0.5f;
	r.view.halfH = static_cast<float>(height) * 0.5f;
}

void setProjection(const float *projection, float near, float far)
{
	View &v = g_renderer.view;
	std::memcpy(v.projection, projection, sizeof(v.projection));
	v.near = near < 0.0001f ? 0.0001f : near;
	v.far = far;
	v.orthographic = projection[15] == 1.0f && projection[11] == 0.0f;
	math::frustumPlanes(v.projection, v.planes);
}

void addLight(const Node &node)
{
	Lights &lights = g_renderer.lights;
	const float *view = g_renderer.view.view;
	const float color[3] = {node.lightColor[0] * node.intensity, node.lightColor[1] * node.intensity, node.lightColor[2] * node.intensity};
	const float *world = node.world;

	if (node.lightKind == kLightDirectional && lights.directionalCount < kMaxLights) {
		float direction[3];
		math::transformDirection(view, world[12] - node.target[0], world[13] - node.target[1], world[14] - node.target[2], direction);
		if (!math::normalize3(direction)) return;
		const int i = lights.directionalCount++;
		std::memcpy(lights.direction[i], direction, sizeof(direction));
		std::memcpy(lights.directionalColor[i], color, sizeof(color));
	} else if (node.lightKind == kLightPoint && lights.pointCount < kMaxLights) {
		const int i = lights.pointCount++;
		math::transformPoint(view, world[12], world[13], world[14], lights.pointPosition[i]);
		std::memcpy(lights.pointColor[i], color, sizeof(color));
		lights.pointDistance[i] = node.distance;
		lights.pointDecay[i] = node.decay;
	} else if (node.lightKind == kLightHemisphere && lights.hemisphereCount < kMaxLights) {
		float up[3];
		math::transformDirection(view, world[12], world[13], world[14], up);
		if (!math::normalize3(up)) {
			math::transformDirection(view, 0.0f, 1.0f, 0.0f, up);
			math::normalize3(up);
		}
		const int i = lights.hemisphereCount++;
		std::memcpy(lights.hemisphereUp[i], up, sizeof(up));
		std::memcpy(lights.sky[i], color, sizeof(color));
		for (int k = 0; k < 3; k++) lights.ground[i][k] = node.groundColor[k] * node.intensity;
	} else if (node.lightKind == kLightAmbient) {
		for (int k = 0; k < 3; k++) lights.ambient[k] += color[k];
	}
}

// Updates world matrices below `root` and collects what it draws and lights.
void collect(int root)
{
	std::vector<int> &stack = g_renderer.stack;
	std::vector<int> &drawables = g_renderer.drawables;
	std::vector<int> &lightNodes = g_renderer.lightNodes;
	stack.clear();
	drawables.clear();
	lightNodes.clear();
	stack.push_back(root);
	while (!stack.empty()) {
		const int id = stack.back();
		stack.pop_back();
		Node &node = g_nodes[static_cast<std::size_t>(id)];
		if (!node.visible) continue;

		refreshWorld(node);
		if (node.kind == kNodeLight) lightNodes.push_back(id);
		else if (node.kind != kNodeGroup && node.kind != kNodeCamera) drawables.push_back(id);
		for (int child = node.firstChild; child >= 0; child = g_nodes[static_cast<std::size_t>(child)].nextSibling) stack.push_back(child);
	}
}

// Frame::rows for the sorted order.
void packRowRanges(raster::Frame &frame)
{
	frame.rowStore.resize(static_cast<std::size_t>(frame.count));
	std::uint32_t *rows = frame.rowStore.data();
	for (int i = 0; i < frame.count; i++) rows[i] = frame.emittedRows[static_cast<std::size_t>(frame.order[static_cast<std::size_t>(i)])];
	frame.rows = rows;
}

// Stably reorders frame.order[first, last) by renderOrder, keeping the depth
// order within each layer: highest layer first for the nearest-first opaque
// run, lowest first for the farthest-first blended run.
void orderLayers(raster::Frame &frame, int first, int last, bool highestFirst)
{
	const std::int32_t *layers = frame.layers.data();
	std::int32_t *order = frame.order.data() + first;
	const int count = last - first;
	const int minLayer = frame.minLayer;
	const int range = frame.maxLayer - minLayer;
	if (count < 2) return;

	if (range >= kLayerCountingRange) {
		std::stable_sort(order, order + count, [layers, highestFirst](std::int32_t a, std::int32_t b) { return highestFirst ? layers[a] > layers[b] : layers[a] < layers[b]; });
		return;
	}

	Renderer &r = g_renderer;
	std::vector<std::int32_t> &counts = r.layerCounts;
	counts.assign(static_cast<std::size_t>(range + 2), 0);
	for (int i = 0; i < count; i++) {
		const int rank = layers[order[i]] - minLayer;
		counts[static_cast<std::size_t>((highestFirst ? range - rank : rank) + 1)]++;
	}
	for (int b = 1; b <= range + 1; b++) counts[static_cast<std::size_t>(b)] += counts[static_cast<std::size_t>(b - 1)];
	std::vector<std::int32_t> &sorted = r.layerOrder;
	sorted.resize(static_cast<std::size_t>(count));
	for (int i = 0; i < count; i++) {
		const int rank = layers[order[i]] - minLayer;
		sorted[static_cast<std::size_t>(counts[static_cast<std::size_t>(highestFirst ? range - rank : rank)]++)] = order[i];
	}
	std::memcpy(order, sorted.data(), static_cast<std::size_t>(count) * sizeof(std::int32_t));
}

// Nearest-first order for opaque primitives, farthest-first for blended;
// by renderOrder first when the frame has several (three.js's order).
// Reads only the dense key arrays, never the primitive records.
void sortFrame()
{
	Renderer &r = g_renderer;
	raster::Frame &frame = r.frame();
	const int n = frame.count;
	frame.order.resize(static_cast<std::size_t>(n));
	r.keys.resize(static_cast<std::size_t>(n));
	r.bucketStarts.assign(2 * (kDepthBuckets + 1), 0);
	frame.opaqueCount = 0;
	frame.blendCount = 0;
	if (n == 0) {
		packRowRanges(frame);
		return;
	}

	const float *keys = frame.sortKeys.data();
	const std::uint8_t *blended = frame.blended.data();
	float lo = keys[0], hi = lo;
	for (int i = 1; i < n; i++) {
		lo = keys[i] < lo ? keys[i] : lo;
		hi = keys[i] > hi ? keys[i] : hi;
	}
	const float range = hi - lo;
	const float scale = range > 0.000001f ? static_cast<float>(kDepthBuckets - 1) / range : 0.0f;

	// One counting sort over [opaque buckets | blended buckets]. Opaque is the
	// exact reverse of the painter's order, ties included (later primitives
	// win a tie, as they would painting back-to-front), so opaque keys count
	// down from the far end and are placed walking the store backwards.
	std::int32_t *counts = r.bucketStarts.data();
	int blendedCount = 0;
	for (int i = 0; i < n; i++) {
		const int painter = minInt(static_cast<int>((hi - keys[i]) * scale), kDepthBuckets - 1);
		const int bucket = blended[i] ? kDepthBuckets + 1 + painter : kDepthBuckets - 1 - painter;
		r.keys[static_cast<std::size_t>(i)] = bucket;
		counts[bucket + 1]++;
		blendedCount += blended[i];
	}
	frame.opaqueCount = n - blendedCount;
	frame.blendCount = blendedCount;
	for (int b = 1; b <= 2 * kDepthBuckets + 1; b++) counts[b] += counts[b - 1];
	for (int k = 0; k < n; k++) {
		const int i = n - 1 - k;
		if (blended[i]) continue;
		frame.order[static_cast<std::size_t>(counts[r.keys[static_cast<std::size_t>(i)]]++)] = i;
	}
	for (int i = 0; i < n; i++) {
		if (!blended[i]) continue;
		frame.order[static_cast<std::size_t>(counts[r.keys[static_cast<std::size_t>(i)]]++)] = i;
	}
	if (frame.minLayer < frame.maxLayer) {
		orderLayers(frame, 0, frame.opaqueCount, true);
		orderLayers(frame, frame.opaqueCount, n, false);
	}
	packRowRanges(frame);
}

// ---------------------------------------------------------------- present

// One band row out to the target: 32-byte blocks (LDM/STM bursts on
// Cortex-M) when both ends are word aligned, else memcpy.
GEA3D_INLINE void copyRow(Pixel *dst, const Pixel *src, std::size_t bytes)
{
	struct Block {
		std::uint32_t words[8];
	};
	if (((reinterpret_cast<std::uintptr_t>(dst) | reinterpret_cast<std::uintptr_t>(src)) & 3u) != 0) {
		std::memcpy(dst, src, bytes);
		return;
	}
	Block *to = reinterpret_cast<Block *>(dst);
	const Block *from = reinterpret_cast<const Block *>(src);
	const std::size_t blocks = bytes / sizeof(Block);
	for (std::size_t i = 0; i < blocks; i++) to[i] = from[i];
	const std::size_t done = blocks * sizeof(Block);
	std::memcpy(reinterpret_cast<std::uint8_t *>(dst) + done, reinterpret_cast<const std::uint8_t *>(src) + done, bytes - done);
}

#if GEA3D_DMA2D_COPY
// Band copy-out on DMA2D (RGB565, memory to memory). A copy starts once the
// previous one (or the engine's own DMA2D work) is done. The data cache is
// cleaned before the first copy (no dirty line may later overwrite what DMA2D
// wrote) and dropped after the last (no stale line may be read): whole-cache
// operations, since the 4 KB cache holds far fewer lines than the target.
static_assert(sizeof(Pixel) == 2, "DMA2D band copies are RGB565");

void waitBandCopy()
{
	while ((DMA2D->CR & DMA2D_CR_START) != 0) {}
}

void startBandCopy(Pixel *dst, int stride, const Pixel *src, int width, int rows)
{
	waitBandCopy();
	__DSB();
	constexpr std::uint32_t kRgb565 = 2;
	DMA2D->CR = 0;
	DMA2D->FGPFCCR = kRgb565;
	DMA2D->FGMAR = reinterpret_cast<std::uint32_t>(src);
	DMA2D->FGOR = 0;
	DMA2D->OPFCCR = kRgb565;
	DMA2D->OMAR = reinterpret_cast<std::uint32_t>(dst);
	DMA2D->OOR = static_cast<std::uint32_t>(stride - width);
	DMA2D->NLR = (static_cast<std::uint32_t>(width) << DMA2D_NLR_PL_Pos) | static_cast<std::uint32_t>(rows);
	DMA2D->CR |= DMA2D_CR_START;
}
#endif

// Each band's primitives in sorted order (Bands::items[starts[b]..starts[b+1]),
// the first opaque[b] of them opaque), so a band never visits the primitives
// that miss it and reads no sort order.
void buildBandLists(const raster::Frame &frame, int y0, int y1, Bands &bands)
{
	const int count = (y1 - y0) / bands.rows + 1;
	bands.bandOfRow.resize(static_cast<std::size_t>(y1 - y0 + 1));
	for (int y = y0; y <= y1; y++) bands.bandOfRow[static_cast<std::size_t>(y - y0)] = (y - y0) / bands.rows;
	bands.starts.assign(static_cast<std::size_t>(count + 1), 0);
	bands.opaque.assign(static_cast<std::size_t>(count), 0);

	const int *bandOf = bands.bandOfRow.data();
	int *starts = bands.starts.data();
	for (int i = 0; i < frame.count; i++) {
		const int first = maxInt(static_cast<std::int16_t>(frame.rows[i] & 0xFFFFu), y0);
		const int last = minInt(static_cast<std::int16_t>(frame.rows[i] >> 16), y1);
		if (first > last) continue;
		for (int b = bandOf[first - y0]; b <= bandOf[last - y0]; b++) starts[b + 1]++;
	}
	for (int b = 0; b < count; b++) starts[b + 1] += starts[b];

	bands.items.resize(static_cast<std::size_t>(starts[count]));
	bands.next.assign(bands.starts.begin(), bands.starts.end() - 1);
	int *next = bands.next.data();
	int *opaque = bands.opaque.data();
	std::int32_t *items = bands.items.data();
	for (int i = 0; i < frame.count; i++) {
		const int first = maxInt(static_cast<std::int16_t>(frame.rows[i] & 0xFFFFu), y0);
		const int last = minInt(static_cast<std::int16_t>(frame.rows[i] >> 16), y1);
		if (first > last) continue;
		const bool isOpaque = i < frame.opaqueCount;
		const std::int32_t prim = frame.order[static_cast<std::size_t>(i)];
		for (int b = bandOf[first - y0]; b <= bandOf[last - y0]; b++) {
			items[next[b]++] = prim;
			opaque[b] += isOpaque;
		}
	}
}

raster::BandList bandList(const Bands &bands, int band)
{
	const std::size_t at = static_cast<std::size_t>(band);
	const int start = bands.starts[at];
	return raster::BandList{bands.items.data() + start, bands.starts[at + 1] - start, bands.opaque[at]};
}

// Rasterizes the frame's viewport part of [x0..x1] x [y0..y1].
void rasterizeInto(const raster::Frame &frame, Pixel *base, int stride, int x0, int y0, int x1, int y1)
{
	x1 = std::min(x1, frame.width - 1);
	y1 = std::min(y1, frame.height - 1);
	if (x0 > x1 || y0 > y1) return;

	Bands &bands = g_renderer.bands;
	const int width = x1 - x0 + 1;
	if (!bands.pixels || width > bands.width) {
		raster::Rasterizer rasterizer(frame, g_renderer.scratch, base, stride, x0, y0, x1, y1);
		rasterizer.run();
		return;
	}

	buildBandLists(frame, y0, y1, bands);
	const std::size_t rowBytes = static_cast<std::size_t>(width) * sizeof(Pixel);
	const bool keepUnder = frame.background.mode == raster::kBackgroundNone;
#if GEA3D_DMA2D_COPY
	if (bands.spare) SCB_CleanInvalidateDCache();
#endif
	int band = 0;
	for (int bandY = y0; bandY <= y1; bandY += bands.rows, band++) {
		const int bandLast = minInt(bandY + bands.rows - 1, y1);
		Pixel *buffer = (band & 1) != 0 && bands.spare ? bands.spare : bands.pixels;
		// The band buffer addressed in screen coordinates, stride = width.
		Pixel *bandBase = buffer - (static_cast<std::ptrdiff_t>(bandY) * width + x0);
		Pixel *target = base + static_cast<std::ptrdiff_t>(bandY) * stride + x0;
		const int rows = bandLast - bandY + 1;
		if (keepUnder) {
			for (int y = 0; y < rows; y++) std::memcpy(buffer + static_cast<std::ptrdiff_t>(y) * width, target + static_cast<std::ptrdiff_t>(y) * stride, rowBytes);
		}
		raster::Rasterizer rasterizer(frame, g_renderer.scratch, bandBase, width, x0, bandY, x1, bandLast, raster::Occlusion::kPainter);
		rasterizer.runPainter(bandList(bands, band));
#if GEA3D_DMA2D_COPY
		if (bands.spare) {
			startBandCopy(target, stride, buffer, width, rows);
			continue;
		}
#endif
		for (int y = 0; y < rows; y++) copyRow(target + static_cast<std::ptrdiff_t>(y) * stride, buffer + static_cast<std::ptrdiff_t>(y) * width, rowBytes);
	}
#if GEA3D_DMA2D_COPY
	if (bands.spare) {
		waitBandCopy();
		SCB_CleanInvalidateDCache();
	}
#endif
}

// Display::flushRectsRasterized chunk: pixels hold rows [oy, oy + h) of a
// full-width band, so the chunk width is its stride.
void rasterChunk(Pixel *pixels, int width, int height, int ox, int oy, void *user)
{
	const raster::Frame &frame = *static_cast<const raster::Frame *>(user);
	Pixel *base = pixels - (static_cast<std::ptrdiff_t>(oy) * width + ox);
	rasterizeInto(frame, base, width, ox, oy, ox + width - 1, oy + height - 1);
}

int displayWidth(int fallback)
{
	auto *canvas = pd::Display::canvas();
	const int width = canvas ? canvas->width() : 0;
	return width > 0 ? width : fallback;
}

bool presentDirect(raster::Frame &frame)
{
	if (frame.width <= 0 || frame.height <= 0) return false;

	const pd::DisplayFlushRect rect{0, 0, displayWidth(frame.width) - 1, frame.height - 1};
	pd::Display::flushRectsRasterized(&rect, 1, rasterChunk, &frame, true);
	pd::Display::rebindCanvasToFramebuffer();
	return true;
}

#if GEA3D_PRESENT_COMMANDS
std::vector<gea::framework::graphics::TriangleEntry> g_entries;

bool presentCommands(raster::Frame &frame)
{
	g_entries.clear();
	for (int i = frame.count - 1; i >= 0; i--) {
		const Prim &prim = frame.prims[frame.order[static_cast<std::size_t>(i)]];
		if (prim.type != raster::kTriangle) continue;
		gea::framework::graphics::TriangleEntry entry{};
		entry.x0 = static_cast<std::int16_t>(prim.x[0] >> 4);
		entry.y0 = static_cast<std::int16_t>(prim.y[0] >> 4);
		entry.x1 = static_cast<std::int16_t>(prim.x[1] >> 4);
		entry.y1 = static_cast<std::int16_t>(prim.y[1] >> 4);
		entry.x2 = static_cast<std::int16_t>(prim.x[2] >> 4);
		entry.y2 = static_cast<std::int16_t>(prim.y[2] >> 4);
		entry.color = prim.color;
		entry.rowY0 = prim.rowY0;
		entry.rowY1 = prim.rowY1;
		g_entries.push_back(entry);
	}
	pd::DisplayPresentCommand commands[2];
	int count = 0;
	const raster::Background &bg = frame.background;
	if (bg.mode != raster::kBackgroundNone) {
		commands[count] = {};
		commands[count].type = pd::DisplayPresentCommandType::FillRectRgb565;
		commands[count].fillRectRgb565 = {0, 0, frame.width, frame.height, raster::detail::packColor(bg.top[0], bg.top[1], bg.top[2]), 255};
		count++;
	}
	commands[count] = {};
	commands[count].type = pd::DisplayPresentCommandType::FillTrianglesRgb565;
	commands[count].fillTrianglesRgb565.entries = g_entries.data();
	commands[count].fillTrianglesRgb565.count = static_cast<int>(g_entries.size());
	commands[count].fillTrianglesRgb565.alpha = 255;
	count++;
	return pd::Display::present(commands, count);
}
#endif

#if GEA3D_UNDERLAY && !GEA3D_PRESENT_COMMANDS
// Engine callback: paints the last frame into the display canvas, clipped to
// the canvas's current clip (a dirty region or one flush band).
void paintUnderlay(int x0, int y0, int x1, int y1, void *user)
{
	auto *canvas = pd::Display::canvas();
	if (!canvas || !canvas->pixels()) return;

	int cx0 = 0, cy0 = 0, cx1 = -1, cy1 = -1;
	canvas->currentClip(&cx0, &cy0, &cx1, &cy1);
	x0 = std::max({x0, cx0, 0});
	y0 = std::max({y0, cy0, 0});
	x1 = std::min({x1, cx1, canvas->width() - 1});
	y1 = std::min({y1, cy1, canvas->height() - 1});
	if (x0 > x1 || y0 > y1) return;

	const int stride = canvas->strideBytes() / static_cast<int>(sizeof(Pixel));
	rasterizeInto(*static_cast<const raster::Frame *>(user), canvas->pixels(), stride, x0, y0, x1, y1);
}

bool presentUnderlay(raster::Frame &frame)
{
	if (!gea::framework::display::Underlay::composited()) return false;

	auto *canvas = pd::Display::canvas();
	if (!canvas || frame.width <= 0 || frame.height <= 0) return false;

	gea::framework::display::Underlay::set(paintUnderlay, &frame, 0, 0, frame.width - 1, frame.height - 1);
	canvas->markDirty(0, 0, frame.width - 1, frame.height - 1);
	return true;
}
#endif

bool presentSurface(raster::Frame &frame)
{
	const int slot = g_surfaces.find(g_renderer.surface);
	if (slot < 0) return false;

	const int width = g_surfaces.width[slot];
	const int height = g_surfaces.height[slot];
	rasterizeInto(frame, g_surfaces.pixels[slot], width, 0, 0, std::min(width, frame.width) - 1, std::min(height, frame.height) - 1);
	return true;
}

#if GEA3D_ASYNC_PRESENT
TaskHandle_t g_presentTask = nullptr;
SemaphoreHandle_t g_presentWork = nullptr;
SemaphoreHandle_t g_presentDone = nullptr;
raster::Frame *g_presentFrame = nullptr;
volatile bool g_presentOk = true;

void presentTaskMain(void *)
{
	for (;;) {
		xSemaphoreTake(g_presentWork, portMAX_DELAY);
		g_presentOk = presentDirect(*g_presentFrame);
		xSemaphoreGive(g_presentDone);
	}
}

bool ensurePresentTask()
{
	if (g_presentTask) return true;
	g_presentWork = xSemaphoreCreateBinary();
	g_presentDone = xSemaphoreCreateBinary();
	if (!g_presentWork || !g_presentDone) return false;
	// "Done" starts available: the first frame must not wait.
	xSemaphoreGive(g_presentDone);
	const BaseType_t ok = xTaskCreatePinnedToCore(presentTaskMain, "gea3d_present", 12288, nullptr, 20, &g_presentTask, 0);
	return ok == pdPASS;
}

// Blocks until the previous asynchronous present released its frame.
void waitForPresent()
{
	if (!g_presentTask) return;
	xSemaphoreTake(g_presentDone, portMAX_DELAY);
	xSemaphoreGive(g_presentDone);
}
#else
void waitForPresent() {}
#endif

// Hands the built frame to its destination. Returns false when rejected.
bool present()
{
	Renderer &r = g_renderer;
	raster::Frame &frame = r.frame();
	r.stats.primitives = frame.count;

	if (r.surface >= 0) {
		waitForPresent();
		return presentSurface(frame);
	}
#if GEA3D_PRESENT_COMMANDS
	return presentCommands(frame);
#else
#if GEA3D_UNDERLAY
	waitForPresent();
	if (presentUnderlay(frame)) {
		r.stats.composited = 1;
		return true;
	}
	gea::framework::display::Underlay::clear();
#endif
#if GEA3D_ASYNC_PRESENT
	if (ensurePresentTask()) {
		// Wait for frame N-1, hand frame N to core 0 and build N+1 in the
		// other slot while it streams.
		xSemaphoreTake(g_presentDone, portMAX_DELAY);
		const bool previousOk = g_presentOk;
		g_presentFrame = &frame;
		xSemaphoreGive(g_presentWork);
		r.building ^= 1;
		return previousOk;
	}
#endif
	return presentDirect(frame);
#endif
}

void setBackground(raster::Background &background, double bgColor)
{
	if (bgColor < 0) {
		background.mode = raster::kBackgroundNone;
		return;
	}
	background.mode = raster::kBackgroundFlat;
	hexToRgb(bgColor, background.top);
	std::memcpy(background.bottom, background.top, sizeof(background.top));
}

}  // namespace

// ================================================================ buffers

double createBuffer()
{
	return allocate(g_buffers);
}

void deleteBuffer(double buffer)
{
	Buffer *slot = lookup(g_buffers, buffer);
	if (slot) *slot = Buffer();
}

void bufferDataF32Raw(double buffer, const float *data, std::size_t count)
{
	Buffer *slot = lookup(g_buffers, buffer);
	if (!slot) return;
	slot->f32.assign(data, data + count);
	slot->version++;
}

void bufferDataU32Raw(double buffer, const std::uint32_t *data, std::size_t count)
{
	Buffer *slot = lookup(g_buffers, buffer);
	if (!slot) return;
	slot->u32.assign(data, data + count);
	slot->version++;
}

// ================================================================ geometry / material

double createGeometry()
{
	return allocate(g_geometries);
}

void deleteGeometry(double geometry)
{
	Geometry *slot = lookup(g_geometries, geometry);
	if (slot) *slot = Geometry();
}

void geometryBuffers(double geometry, double position, double index, double color, double normal)
{
	Geometry *slot = lookup(g_geometries, geometry);
	if (!slot) return;
	slot->position = static_cast<int>(position);
	slot->index = static_cast<int>(index);
	slot->color = static_cast<int>(color);
	slot->normal = static_cast<int>(normal);
	slot->radius = -1.0f;
}

double createMaterial()
{
	return allocate(g_materials);
}

void deleteMaterial(double material)
{
	Material *slot = lookup(g_materials, material);
	if (slot) *slot = Material();
}

void materialSet(double material, double shade, double r, double g, double b, double opacity, double side, double flags, double size)
{
	Material *slot = lookup(g_materials, material);
	if (!slot) return;
	slot->shade = static_cast<int>(shade);
	const double rgb[3] = {r, g, b};
	for (int k = 0; k < 3; k++) slot->color[k] = clampFloat(static_cast<float>(rgb[k] * 255.0), 0.0f, 255.0f);
	slot->opacity = static_cast<float>(opacity);
	slot->side = static_cast<int>(side);
	slot->flags = static_cast<int>(flags);
	slot->size = static_cast<float>(size);
}

// ================================================================ nodes

double createNode(double kind)
{
	const int id = allocate(g_nodes);
	g_nodes[static_cast<std::size_t>(id)].kind = static_cast<int>(kind);
	return id;
}

void deleteNode(double node)
{
	Node *slot = lookup(g_nodes, node);
	if (!slot) return;

	const int id = static_cast<int>(node);
	unlink(id);
	while (slot->firstChild >= 0) unlink(slot->firstChild);
	*slot = Node();
}

void nodeParent(double node, double parent)
{
	if (!lookup(g_nodes, node)) return;

	const int id = static_cast<int>(node);
	const int parentId = static_cast<int>(parent);
	const bool attach = lookup(g_nodes, parent) != nullptr && !isAncestor(id, parentId);
	if (g_nodes[static_cast<std::size_t>(id)].parent == (attach ? parentId : -1)) return;

	unlink(id);
	if (attach) link(id, parentId);
}

void nodeTransform(double node, double px, double py, double pz, double qx, double qy, double qz, double qw, double sx, double sy, double sz)
{
	Node *slot = lookup(g_nodes, node);
	if (!slot) return;
	const float position[3] = {static_cast<float>(px), static_cast<float>(py), static_cast<float>(pz)};
	const float quaternion[4] = {static_cast<float>(qx), static_cast<float>(qy), static_cast<float>(qz), static_cast<float>(qw)};
	const float scale[3] = {static_cast<float>(sx), static_cast<float>(sy), static_cast<float>(sz)};
	// An unchanged transform keeps the cached world matrices.
	const bool same = !slot->useMatrix && std::memcmp(slot->position, position, sizeof(position)) == 0 &&
	                  std::memcmp(slot->quaternion, quaternion, sizeof(quaternion)) == 0 && std::memcmp(slot->scale, scale, sizeof(scale)) == 0;
	if (same) return;

	std::memcpy(slot->position, position, sizeof(position));
	std::memcpy(slot->quaternion, quaternion, sizeof(quaternion));
	std::memcpy(slot->scale, scale, sizeof(scale));
	slot->useMatrix = false;
	slot->localDirty = true;
}

void nodeMatrixRaw(double node, const float *elements, std::size_t count)
{
	Node *slot = lookup(g_nodes, node);
	if (!slot || count < 16) return;
	// An unchanged matrix keeps the cached world matrices.
	if (slot->useMatrix && std::memcmp(slot->local, elements, sizeof(slot->local)) == 0) return;

	std::memcpy(slot->local, elements, sizeof(slot->local));
	slot->useMatrix = true;
	slot->localDirty = true;
}

void nodeFlags(double node, double visible, double frustumCulled, double renderOrder)
{
	Node *slot = lookup(g_nodes, node);
	if (!slot) return;
	slot->visible = visible != 0;
	slot->frustumCulled = frustumCulled != 0;
	slot->renderOrder = static_cast<int>(renderOrder);
}

void nodeGeometry(double node, double geometry, double material)
{
	Node *slot = lookup(g_nodes, node);
	if (!slot) return;
	slot->geometry = static_cast<int>(geometry);
	slot->material = static_cast<int>(material);
}

void nodeInstances(double node, double matrices, double colors, double count)
{
	Node *slot = lookup(g_nodes, node);
	if (!slot) return;
	slot->instanced = true;
	slot->instanceMatrices = static_cast<int>(matrices);
	slot->instanceColors = static_cast<int>(colors);
	slot->instanceCount = std::max(0, static_cast<int>(count));
}

void nodeLight(double node, double kind, double r, double g, double b, double intensity, double groundR, double groundG, double groundB, double distance, double decay)
{
	Node *slot = lookup(g_nodes, node);
	if (!slot) return;
	slot->lightKind = static_cast<int>(kind);
	slot->lightColor[0] = static_cast<float>(r);
	slot->lightColor[1] = static_cast<float>(g);
	slot->lightColor[2] = static_cast<float>(b);
	slot->intensity = static_cast<float>(intensity);
	slot->groundColor[0] = static_cast<float>(groundR);
	slot->groundColor[1] = static_cast<float>(groundG);
	slot->groundColor[2] = static_cast<float>(groundB);
	slot->distance = static_cast<float>(distance);
	slot->decay = static_cast<float>(decay);
}

void nodeLightTarget(double node, double x, double y, double z)
{
	Node *slot = lookup(g_nodes, node);
	if (!slot) return;
	slot->target[0] = static_cast<float>(x);
	slot->target[1] = static_cast<float>(y);
	slot->target[2] = static_cast<float>(z);
}

void cameraProjectionRaw(double node, const float *elements, std::size_t count, double near, double far)
{
	Node *slot = lookup(g_nodes, node);
	if (!slot || count < 16) return;
	std::memcpy(slot->projection, elements, sizeof(slot->projection));
	slot->near = static_cast<float>(near);
	slot->far = static_cast<float>(far);
}

void sceneEnvironment(double scene, double background, double top, double bottom, double fog, double fogColor, double fogNear, double fogFar, double fogDensity)
{
	Node *slot = lookup(g_nodes, scene);
	if (!slot) return;

	Environment &env = slot->environment;
	const int mode = static_cast<int>(background);
	env.background.mode = static_cast<std::uint8_t>(std::clamp(mode, 0, 2));
	hexToRgb(top, env.background.top);
	if (mode == raster::kBackgroundGradient) hexToRgb(bottom, env.background.bottom);
	else std::memcpy(env.background.bottom, env.background.top, sizeof(env.background.top));
	env.fog = static_cast<int>(fog);
	std::uint8_t rgb[3];
	hexToRgb(fogColor, rgb);
	for (int k = 0; k < 3; k++) env.fogColor[k] = rgb[k];
	env.fogNear = static_cast<float>(fogNear);
	env.fogFar = static_cast<float>(fogFar);
	env.fogDensity = static_cast<float>(fogDensity);
}

// ================================================================ frame

void renderOptions(double depthBuffer, double capacity)
{
	g_renderer.depthBuffer = depthBuffer != 0;
	if (capacity >= 64) g_renderer.capacity = static_cast<int>(capacity);
}

void renderTarget(double surface)
{
	g_renderer.surface = static_cast<int>(surface);
}

double createSurface(double width, double height)
{
	const int w = static_cast<int>(width);
	const int h = static_cast<int>(height);
	if (w <= 0 || h <= 0 || g_surfaces.count >= 8) return -1;

	auto *pixels = new (std::nothrow) Pixel[static_cast<std::size_t>(w) * static_cast<std::size_t>(h)]();
	if (!pixels) return -1;
	const int id = gea::framework::graphics::ImageStore::instance().registerBuffer(pixels, w, h);
	if (id < 0) {
		delete[] pixels;
		return -1;
	}
	const int slot = g_surfaces.count++;
	g_surfaces.ids[slot] = id;
	g_surfaces.pixels[slot] = pixels;
	g_surfaces.width[slot] = w;
	g_surfaces.height[slot] = h;
	return id;
}

double drawScene(double scene, double camera, double width, double height)
{
	Node *sceneNode = lookup(g_nodes, scene);
	Node *cameraNode = lookup(g_nodes, camera);
	if (!sceneNode || !cameraNode) return 0;

	Renderer &r = g_renderer;
	beginFrameState(static_cast<int>(width), static_cast<int>(height));
	r.legacyActive = false;
	raster::Frame &frame = r.frame();
	r.environment = sceneNode->environment;
	frame.background = r.environment.background;

	collect(static_cast<int>(scene));

	ensureWorld(static_cast<int>(camera));
	math::invertAffine(r.view.view, cameraNode->world);
	setProjection(cameraNode->projection, cameraNode->near, cameraNode->far);

	for (const int id : r.lightNodes) addLight(g_nodes[static_cast<std::size_t>(id)]);
	for (const int id : r.drawables) drawNode(g_nodes[static_cast<std::size_t>(id)]);

	sortFrame();
	const int count = frame.count;
	return present() ? count : -1;
}

double stat(double which)
{
	const Stats &s = g_renderer.stats;
	switch (static_cast<int>(which)) {
	case 0: return s.submitted;
	case 1: return s.nearRejected;
	case 2: return s.culled;
	case 3: return s.clipDropped;
	case 4: return s.dropped;
	case 5: return s.meshesDrawn;
	case 6: return s.meshesCulled;
	case 7: return s.primitives;
	case 8: return s.composited;
	default: return 0;
	}
}

// ================================================================ raycast

double raycast(double node, double ox, double oy, double oz, double dx, double dy, double dz, double near, double far)
{
	Node *slot = lookup(g_nodes, node);
	if (!slot || slot->kind != kNodeMesh) return 0;
	Geometry *geometry = lookup(g_geometries, slot->geometry);
	Material *material = lookup(g_materials, slot->material);
	const Buffer *positions = geometry ? bufferAt(geometry->position) : nullptr;
	if (!positions || !material) return 0;

	float direction[3] = {static_cast<float>(dx), static_cast<float>(dy), static_cast<float>(dz)};
	if (!math::normalize3(direction)) return 0;
	ensureWorld(static_cast<int>(node));

	const Buffer *indexBuffer = bufferAt(geometry->index);
	const std::uint32_t *indices = indexBuffer && !indexBuffer->u32.empty() ? indexBuffer->u32.data() : nullptr;
	const std::size_t vertexCount = positions->f32.size() / 3;
	const std::size_t triangles = (indices ? indexBuffer->u32.size() : vertexCount) / 3;
	const float *p = positions->f32.data();
	const Buffer *instances = slot->instanced ? bufferAt(slot->instanceMatrices) : nullptr;
	const int instanceCount = slot->instanced ? (instances ? std::min(slot->instanceCount, static_cast<int>(instances->f32.size() / 16)) : 0) : 1;

	float best = static_cast<float>(far);
	bool hit = false;
	for (int instance = 0; instance < instanceCount; instance++) {
		float world[16];
		if (instances) math::multiplyAffine(world, slot->world, &instances->f32[static_cast<std::size_t>(instance) * 16]);
		else std::memcpy(world, slot->world, sizeof(world));
		float inverse[16];
		if (!math::invertAffine(inverse, world)) continue;

		float o[3], d[3];
		math::transformPoint(inverse, static_cast<float>(ox), static_cast<float>(oy), static_cast<float>(oz), o);
		math::transformDirection(inverse, direction[0], direction[1], direction[2], d);

		for (std::size_t t = 0; t < triangles; t++) {
			std::uint32_t id[3];
			for (int k = 0; k < 3; k++) id[k] = indices ? indices[t * 3 + k] : static_cast<std::uint32_t>(t * 3 + k);
			if (id[0] >= vertexCount || id[1] >= vertexCount || id[2] >= vertexCount) continue;

			const float *a = &p[id[0] * 3], *b = &p[id[1] * 3], *c = &p[id[2] * 3];
			const float e1[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
			const float e2[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
			const float h[3] = {d[1] * e2[2] - d[2] * e2[1], d[2] * e2[0] - d[0] * e2[2], d[0] * e2[1] - d[1] * e2[0]};
			const float det = e1[0] * h[0] + e1[1] * h[1] + e1[2] * h[2];
			if (det > -1e-9f && det < 1e-9f) continue;
			if ((material->side == 0 && det < 0.0f) || (material->side == 1 && det > 0.0f)) continue;

			const float inv = 1.0f / det;
			const float s[3] = {o[0] - a[0], o[1] - a[1], o[2] - a[2]};
			const float u = (s[0] * h[0] + s[1] * h[1] + s[2] * h[2]) * inv;
			if (u < 0.0f || u > 1.0f) continue;
			const float q[3] = {s[1] * e1[2] - s[2] * e1[1], s[2] * e1[0] - s[0] * e1[2], s[0] * e1[1] - s[1] * e1[0]};
			const float v = (d[0] * q[0] + d[1] * q[1] + d[2] * q[2]) * inv;
			if (v < 0.0f || u + v > 1.0f) continue;
			const float t0 = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) * inv;
			if (t0 < static_cast<float>(near) || t0 >= best) continue;

			best = t0;
			hit = true;
			g_rayHit.distance = t0;
			g_rayHit.point[0] = static_cast<float>(ox) + direction[0] * t0;
			g_rayHit.point[1] = static_cast<float>(oy) + direction[1] * t0;
			g_rayHit.point[2] = static_cast<float>(oz) + direction[2] * t0;
			g_rayHit.face = static_cast<int>(t);
			g_rayHit.instance = instances ? instance : -1;
		}
	}
	return hit ? 1 : 0;
}

double rayResult(double which)
{
	switch (static_cast<int>(which)) {
	case 0: return g_rayHit.distance;
	case 1: return g_rayHit.point[0];
	case 2: return g_rayHit.point[1];
	case 3: return g_rayHit.point[2];
	case 4: return g_rayHit.face;
	case 5: return g_rayHit.instance;
	default: return 0;
	}
}

// ================================================================ immediate API

void beginFrameRaw(const float *projection, std::size_t count, double near, double bgColor, double viewportW, double viewportH)
{
	Renderer &r = g_renderer;
	beginFrameState(static_cast<int>(viewportW), static_cast<int>(viewportH));
	r.legacyActive = true;
	r.environment = Environment();
	setBackground(r.frame().background, bgColor);
	math::identity(r.view.view);
	float matrix[16];
	for (int i = 0; i < 16; i++) matrix[i] = i < static_cast<int>(count) ? projection[i] : 0.0f;
	setProjection(matrix, static_cast<float>(near), 1e30f);
}

void ambientLight(double r, double g, double b)
{
	Lights &lights = g_renderer.lights;
	lights.ambient[0] += static_cast<float>(r);
	lights.ambient[1] += static_cast<float>(g);
	lights.ambient[2] += static_cast<float>(b);
}

void directionalLight(double x, double y, double z, double r, double g, double b)
{
	Lights &lights = g_renderer.lights;
	float direction[3] = {static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)};
	if (lights.directionalCount >= kMaxLights || !math::normalize3(direction)) return;
	const int i = lights.directionalCount++;
	std::memcpy(lights.direction[i], direction, sizeof(direction));
	lights.directionalColor[i][0] = static_cast<float>(r);
	lights.directionalColor[i][1] = static_cast<float>(g);
	lights.directionalColor[i][2] = static_cast<float>(b);
}

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
                     double doubleSide)
{
	if (!g_renderer.legacyActive || modelViewCount < 16) return;
	const Buffer *positions = bufferAt(static_cast<int>(positionBuffer));
	if (!positions) return;

	Surface surface;
	surface.shade = static_cast<int>(shadeMode);
	surface.color[0] = static_cast<float>(baseR);
	surface.color[1] = static_cast<float>(baseG);
	surface.color[2] = static_cast<float>(baseB);
	surface.side = doubleSide != 0 ? 2 : 0;
	surface.vertexColors = colorBuffer >= 0;
	surface.fog = false;

	Geometry geometry;
	geometry.position = static_cast<int>(positionBuffer);
	geometry.index = static_cast<int>(indexBuffer);
	geometry.color = static_cast<int>(colorBuffer);

	const std::size_t vertexCount = positions->f32.size() / 3;
	const Buffer *indices = bufferAt(geometry.index);
	const std::size_t available = indices ? indices->u32.size() : vertexCount;
	const std::size_t indexCount = std::min(available, static_cast<std::size_t>(std::max(0.0, triangleCount)) * 3);
	const Buffer *colors = bufferAt(geometry.color);
	const float white[3] = {1, 1, 1};
	transformVertices(modelView, positions->f32.data(), vertexCount, colors && colors->f32.size() >= vertexCount * 3 ? colors->f32.data() : nullptr, nullptr, surface, white, false, !shadesPerFace(surface, false));
	emitTriangles(surface, indices ? vertexCount : std::min(vertexCount, indexCount), indices && !indices->u32.empty() ? indices->u32.data() : nullptr, indexCount, false);
}

double endFrame()
{
	if (!g_renderer.legacyActive) return 0;
	g_renderer.legacyActive = false;
	sortFrame();
	const int count = g_renderer.frame().count;
	return present() ? count : -1;
}

}  // namespace gea::host::gea3d
