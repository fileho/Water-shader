/*
 * Source code for the NPGR019 lab practices. Copyright Martin Kahoun 2021.
 * Licensed under the zlib license, see LICENSE.txt in the root directory.
 */

#pragma once

#include <ShaderCompiler.h>

// Shader programs
namespace ShaderProgram
{
  enum
  {
    Default, Instancing, PointRendering, Tonemapping, Water, NumShaderPrograms
  };
}

// Shader programs handle
extern GLuint shaderProgram[ShaderProgram::NumShaderPrograms];

// Helper function for creating and compiling the shaders
bool compileShaders();

// ============================================================================

// Vertex shader types
namespace VertexShader
{
  enum
  {
    Default, Instancing, Point, ScreenQuad, Water, NumVertexShaders
  };
}

// Vertex shader sources
static const char* vsSource[] = {
// ----------------------------------------------------------------------------
// Default vertex shader
// ----------------------------------------------------------------------------
R"(
#version 330 core

// The following is not not needed since GLSL version #430
#extension GL_ARB_explicit_uniform_location : require

// Uniform blocks, i.e., constants
layout (std140) uniform TransformBlock
{
  // Transposed worldToView matrix - stored compactly as an array of 3 x vec4
  mat3x4 worldToView;
  mat4x4 projection;
};

// Model to world transformation separately, takes 4 slots!
layout (location = 0) uniform mat4x3 modelToWorld;

// Vertex attribute block, i.e., input
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec3 tangent;
layout (location = 3) in vec2 texCoord;

// Vertex output
out VertexData
{
  vec2 texCoord;
  vec3 tangent;
  vec3 bitangent;
  vec3 normal;
  vec4 worldPos;
} vOut;

void main()
{
  // Pass texture coordinates to the fragment shader
  vOut.texCoord = texCoord.st;

  // Construct the normal transformation matrix
  mat3 normalTransform = transpose(inverse(mat3(modelToWorld)));

  // Create the tangent space matrix and pass it to the fragment shader
  vOut.normal = normalize(normalTransform * normal);
  vOut.tangent = normalize(mat3(modelToWorld) * tangent);
  vOut.bitangent = cross(vOut.tangent, vOut.normal);

  // Transform vertex position
  vOut.worldPos = vec4(modelToWorld * vec4(position.xyz, 1.0f), 1.0f);

  // We must multiply from the left because of transposed worldToView
  vec4 viewPos = vec4(vOut.worldPos * worldToView, 1.0f);

  gl_Position = projection * viewPos;
}
)",
// ----------------------------------------------------------------------------
// Instancing vertex shader using instancing buffer via uniform block objects
// ----------------------------------------------------------------------------
R"(
#version 330 core

// The following is not not needed since GLSL version #430
#extension GL_ARB_explicit_uniform_location : require

// Uniform blocks, i.e., constants
layout (std140) uniform TransformBlock
{
  // Transposed worldToView matrix - stored compactly as an array of 3 x vec4
  mat3x4 worldToView;
  mat4x4 projection;
};

// Vertex attribute block, i.e., input
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec3 tangent;
layout (location = 3) in vec2 texCoord;
	
layout (location = 0) uniform vec4 clippingPlane;

// Must match the structure on the CPU side
struct InstanceData
{
  // Transposed worldToView matrix - stored compactly as an array of 3 x vec4
  mat3x4 modelToWorld;
};

// Uniform buffer used for instances
layout (std140) uniform InstanceBuffer
{
  // We are limited to 4096 vec4 registers in total, hence the maximum number of instances
  // being 1024 meaning we could fit another vec4 worth of data
  InstanceData instanceBuffer[1024];
};

// Vertex output
out VertexData
{
  vec2 texCoord;
  vec3 tangent;
  vec3 bitangent;
  vec3 normal;
  vec4 worldPos;
} vOut;

void main()
{
  // Retrieve the model to world matrix from the instance buffer
  mat3x4 modelToWorld = instanceBuffer[gl_InstanceID].modelToWorld;
	
  // Transform vertex position, note we multiply from the left because of transposed modelToWorld
  vOut.worldPos = vec4(vec4(position.xyz, 1.0f) * modelToWorld, 1.0f);

  // Clipping plane for reflexions
  gl_ClipDistance[0] = dot(clippingPlane, vOut.worldPos);
	
  // Pass texture coordinates to the fragment shader
  vOut.texCoord = texCoord.st;

  // Construct the normal transformation matrix
  mat3 normalTransform = transpose(inverse(mat3(modelToWorld)));

  // Create the tangent space matrix and pass it to the fragment shader
  // Note: we must multiply from the left because of transposed modelToWorld
  vOut.normal = normalize(normal * normalTransform);
  vOut.tangent = normalize(tangent * mat3(modelToWorld));
  vOut.bitangent = cross(vOut.tangent, vOut.normal);

  vec4 viewPos = vec4(vOut.worldPos * worldToView, 1.0f);

  gl_Position = projection * viewPos;
}
)",
// ----------------------------------------------------------------------------
// Vertex shader for point rendering
// ----------------------------------------------------------------------------
R"(
#version 330 core

// Uniform blocks, i.e., constants
layout (std140) uniform TransformBlock
{
  // Transposed worldToView matrix - stored compactly as an array of 3 x vec4
  mat3x4 worldToView;
  mat4x4 projection;
};

uniform vec3 position;

void main()
{
  // We must multiply from the left because of transposed worldToView
  vec4 viewPos = vec4(vec4(position, 1.0f) * worldToView, 1.0f);
  gl_Position = projection * viewPos;
}
)",
// ----------------------------------------------------------------------------
// Fullscreen quad vertex shader
// ----------------------------------------------------------------------------
R"(
#version 330 core

// Fullscreen quad
vec3 position[6] = vec3[6](vec3(-1.0f, -1.0f, 0.0f),
                           vec3( 1.0f, -1.0f, 0.0f),
                           vec3( 1.0f,  1.0f, 0.0f),
                           vec3( 1.0f,  1.0f, 0.0f),
                           vec3(-1.0f,  1.0f, 0.0f),
                           vec3(-1.0f, -1.0f, 0.0f));

// Quad UV coordinates
out vec2 UV;

void main()
{
  UV = position[gl_VertexID].xy * 0.5f + 0.5f;
  gl_Position = vec4(position[gl_VertexID].xyz, 1.0f);
}
)",

// ----------------------------------------------------------------------------
// Vertex shader for Water rendering
// ----------------------------------------------------------------------------
R"(
#version 430 core

// Vertex attribute block, i.e., input
layout (location = 0) in vec3 position;
	
void main()
{
  gl_Position = vec4(position, 1.0f);
}
)",
""};

// ============================================================================

// Fragment shader types
namespace FragmentShader
{
  enum
  {
    Default, SingleColor, Tonemapping, Water, NumFragmentShaders
  };
}

// Fragment shader sources
static const char* fsSource[] = {
// ----------------------------------------------------------------------------
// Default fragment shader source
// ----------------------------------------------------------------------------
R"(
#version 330 core

// The following is not not needed since GLSL version #430
#extension GL_ARB_explicit_uniform_location : require

// The following is not not needed since GLSL version #420
#extension GL_ARB_shading_language_420pack : require

// Texture sampler
layout (binding = 0) uniform sampler2D Diffuse;
layout (binding = 1) uniform sampler2D Normal;
layout (binding = 2) uniform sampler2D Specular;
layout (binding = 3) uniform sampler2D Occlusion;

// Note: explicit location because AMD APU drivers screw up position when linking against
// the default vertex shader with mat4x3 modelToWorld at location 0 occupying 4 slots

// Light position/direction
layout (location = 4) uniform vec3 lightPosWS;
// View position in world space coordinates
layout (location = 5) uniform vec4 viewPosWS;

// Fragment shader inputs
in VertexData
{
  vec2 texCoord;
  vec3 tangent;
  vec3 bitangent;
  vec3 normal;
  vec4 worldPos;
} vIn;

// Fragment shader outputs
layout (location = 0) out vec4 color;

void main()
{
  // Normally you'd pass this as another uniform
  vec3 lightColor = vec3(100.0f, 100.0f, 100.0f);

  // Sample textures
  vec3 albedo = texture(Diffuse, vIn.texCoord.st).rgb;
  vec3 noSample = texture(Normal, vIn.texCoord.st).rgb;
  float specSample = texture(Specular, vIn.texCoord.st).r;
  float occlusion = texture(Occlusion, vIn.texCoord.st).r;

  // Calculate world-space normal
  mat3 STN = {vIn.tangent, vIn.bitangent, vIn.normal};
  vec3 normal = STN * (noSample * 2.0f - 1.0f);

  // Calculate the lighting direction and distance
  vec3 lightDir = lightPosWS.xyz - vIn.worldPos.xyz;
  float lengthSq = dot(lightDir, lightDir);
  float length = sqrt(lengthSq);
  lightDir /= length;

  // Calculate the view and reflection/halfway direction
  vec3 viewDir = normalize(viewPosWS.xyz - vIn.worldPos.xyz);
  // Cheaper approximation of reflected direction = reflect(-lightDir, normal)
  vec3 halfDir = normalize(viewDir + lightDir);

  // Calculate diffuse and specular coefficients
  float NdotL = max(0.0f, dot(normal, lightDir));
  float NdotH = max(0.0f, dot(normal, halfDir));

  // Calculate horizon fading factor
  float horizon = clamp(1.0f + dot(vIn.normal, lightDir), 0.0f, 1.0f);
  horizon *= horizon;
  horizon *= horizon;
  horizon *= horizon;
  horizon *= horizon;

  // Calculate the Phong model terms: ambient, diffuse, specular
  vec3 ambient = vec3(0.25f, 0.25f, 0.25f) * 5.0f * occlusion;
  vec3 diffuse = horizon * NdotL * lightColor / lengthSq;
  vec3 specular = horizon * specSample * lightColor * pow(NdotH, 64.0f) / lengthSq; // Defines shininess

  // Spotlight cone
  vec3 spotDir = normalize(lightPosWS.xyz);
  float theta = dot(lightDir, spotDir);
  float outer = 0.7f;
  float inner = 0.5f;
  float epsilon = outer - inner;
  float attenuation = clamp((theta - outer) / epsilon, 0.0f, 1.0f);
  diffuse *= attenuation;
  specular *= attenuation;

  // Calculate the final color
  vec3 finalColor = albedo * (ambient + diffuse) + specular;
  color = vec4(finalColor, 1.0f);
}
)",
// ----------------------------------------------------------------------------
// Single color pixel shader
// ----------------------------------------------------------------------------
R"(
#version 330 core

// Input color
uniform vec3 color;

// Output color
out vec4 oColor;

void main()
{
  oColor = vec4(color.rgb, 1.0f);
}
)",
// ----------------------------------------------------------------------------
// Tonemapping fragment shader source
// ----------------------------------------------------------------------------
R"(
#version 330 core

// The following is not not needed since GLSL version #430
#extension GL_ARB_explicit_uniform_location : require

// The following is not not needed since GLSL version #420
#extension GL_ARB_shading_language_420pack : require

// Our HDR buffer texture
layout (binding = 0) uniform sampler2DMS HDR;

// Number of used MSAA samples
layout (location = 0) uniform float MSAA_LEVEL;

// Quad UV coordinates
in vec2 UV;

// Output
out vec4 color;

vec3 ApplyTonemapping(vec3 hdr)
{
  // Reinhard global operator
  vec3 result = hdr / (hdr + vec3(1.0f));

  return result;
}

void main()
{
  // Query the size of the texture and calculate texel coordinates
  ivec2 texSize = textureSize(HDR);
  ivec2 texel = ivec2(UV * texSize);

  // Accumulate color for all MSAA samples
  vec3 finalColor = vec3(0.0f);
  for (int i = 0; i < int(MSAA_LEVEL); ++i)
  {
     // Fetch a single sample from a single texel (no interpolation)
     vec3 s = texelFetch(HDR, texel, i).rgb;
     finalColor += ApplyTonemapping(s);
  }

  color = vec4(finalColor.rgb / MSAA_LEVEL, 1.0f);
}
)",
// ----------------------------------------------------------------------------
// Water fragment shader
// ----------------------------------------------------------------------------
R"(
#version 430 core

// Output color
layout (binding = 0) uniform sampler2D refraction;
layout (binding = 1) uniform sampler2D reflexion;
layout (binding = 2) uniform sampler2D depth;
layout (binding = 3) uniform sampler2D bump;
	
layout (location = 1) uniform float time;
layout (location = 2) uniform vec2 resolution;
layout (location = 3) uniform vec4 viewPosWS;
layout (location = 4) uniform vec4 wave[4];

layout (std140) uniform TransformBlock
{
  // Transposed worldToView matrix - stored compactly as an array of 3 x vec4
  mat3x4 worldToView;
  mat4x4 projection;
};

in VertexData
{
  vec2 uv;
  vec4 worldPos;
} vIn;

out vec4 oColor;

vec3 waveDx(vec4 w) {
  return vec3(1, 0, w.x * w.a * cos(w.x * vIn.uv.x + w.y * vIn.uv.y + w.z * time));
}
vec3 waveDy(vec4 w) {
  return vec3(0, 1, w.y * w.a * cos(w.x * vIn.uv.x + w.y * vIn.uv.y + w.z * time));
}

float remapDepth(float d) {
  //return min(d * 6.0f + 0.2f, 1.0f);
  return 2 / (1 + pow(20, -(d * 10 + 0.05f))) - 1; // use sigmoid like function remapped to [-1, 1] (0 at 0)
}

float smoothEdges() {
  float u = vIn.uv.x;
  float v = vIn.uv.y;
  u = min(u, 1 - u);
  v = min(v, 1 - v);
  float m = min(u, v);
  return min(1.0f, 10.0f * m * m);
}
	

void main()
{
  // Calculate Bitangent with unrolled loop
  vec3 B = waveDx(wave[0]);
  B += waveDx(wave[1]);
  B += waveDx(wave[2]);
  B += waveDx(wave[3]);
  
  // Calculate Tangent
  vec3 T = waveDy(wave[0]);
  T += waveDy(wave[1]);
  T += waveDy(wave[2]);
  T += waveDy(wave[3]);
  
  // Tangent space Normal
  vec3 N = cross(B, T);
  N = normalize(N);
  

  vec2 coords = vec2(gl_FragCoord.x / resolution.x, gl_FragCoord.y / resolution.y);
  float d = texture(depth, coords).r; // sample depth map
  // Calculate depth of the water - refraction map depth - surface depth
  float z = gl_FragCoord.z;
  float n = 0.5f;
  float f = 150.1f;
  float linearZ = (2.0 * n) / (f + n - z*(f-n));
  float linearD = (2.0 * n) / (f + n - d*(f-n));
  linearD -= linearZ;
  d = remapDepth(linearD);
  
  vec2 bumpCoords = sin(vIn.uv + vec2(time * 0.02f, time * 0.01f)) * 0.5f + 0.5f; // remap to [0, 1]
  vec2 bm = texture(bump, bumpCoords).rg * 2.0f - 1.0f; // remap to [-1,1]
  bm *= 0.5f; // scale down bump map offsets

  // Calculate view space normals adjusted with bump map details
  vec3 worldN = normalize(vec3(N.x + bm.x, N.z, N.y + bm.y));
  vec3 viewN = vec4(worldN, 0.0f) * worldToView;
	
  // Calculate sample offsets and scale it by depth for smooth edges
  vec3 view = viewPosWS.xyz - vIn.worldPos.xyz;
  float viewDistance = max(length(view), 10.0f);
  vec3 viewDir = normalize(view);

  vec2 offset = viewN.xz / (viewDistance * 3.0f + 1.0f); // scale down normals based on view distance

  vec3 refrac = texture(refraction, coords + offset).rgb;
  vec3 reflex = texture(reflexion, vec2(coords.x + offset.x, 1 - (coords.y + offset.y))).rgb; 


  // Reflexion coefficient calculated using fresnel equations (n1 = 1, n2 = 1.333)
  // Use normal pointing directly up, otherwise it add a lot of noise
  float angle = max(dot(viewDir, vec3(0, 1, 0)),0);
  const float n2 = 1.333;
  float rhs = sqrt(1 - (1 / n2 * pow(sin(acos(angle)), 2)));
  float rs = pow((angle - n2 * rhs) / (angle + n2 * rhs), 2);
  float rp = pow((rhs - n2 * angle) / (rhs + n2 * angle), 2);
  float refCoef = rs + rp;

  // Schlick's approximation
  // float R0 = 0.02037;
  // float refCoef = R0 + (1 - R0) * pow(1 - max(dot(viewDir, vec3(0,1,0)), 0), 5);

  // specular highlights
  vec3 lightDir = vec3(0, 10, 5);
  lightDir = normalize(lightDir);
  vec3 halfDir = normalize(viewDir + lightDir);
  float NdotH = max(0, dot(worldN, halfDir));
  vec3 specular = 2 * vec3(1, 1, 1) * pow(NdotH, 128.0f) * smoothEdges();
  reflex += specular;

	
  // This is more "realistic" absorption, works better for dirty water - lakes, sea; bad for clean swimming pool
  // vec3 absorb = clamp(1 - d * 2.5f * vec3(0.8f, 0.6f, 0.4f), 0, 1);
  // refrac *= absorb;
  // oColor = mix(vec4(refrac, 1.0), vec4(reflex,1), refCoef);
  // return;
	
  vec4 absorption = vec4(0.0f, 0.5f, 0.9f, 1.0f);
  oColor = mix(mix(vec4(refrac, 1.0f), absorption, d), vec4(reflex,1), refCoef);	
}
)",
	
""};

static const char* tsSource[] =
{
R"(
// tessellation control shader
#version 430 core

layout(vertices = 4) out;

void main()
{
    // ----------------------------------------------------------------------
    // pass attributes through
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;

    // ----------------------------------------------------------------------
    // invocation zero controls tessellation levels for the entire patch
    if (gl_InvocationID == 0)
    {
        gl_TessLevelOuter[0] = 16;
        gl_TessLevelOuter[1] = 16;
        gl_TessLevelOuter[2] = 16;
        gl_TessLevelOuter[3] = 16;

        gl_TessLevelInner[0] = 8;
        gl_TessLevelInner[1] = 8;
    }
}
)",
R"(
// tessellation evaluation shader
#version 430 core
	
layout (quads, fractional_odd_spacing, ccw) in;
layout (std140) uniform TransformBlock
{
  // Transposed worldToView matrix - stored compactly as an array of 3 x vec4
  mat3x4 worldToView;
  mat4x4 projection;
};
layout (location = 0) uniform mat3x4 modelToWorld;
layout (location = 1) uniform float time;
layout (location = 4) uniform vec4 wave[4];

out VertexData
{
    vec2 uv;
    vec4 worldPos;
} vOut;


float h(vec4 w) {
  return w.a * sin(vOut.uv.x * w.x + vOut.uv.y * w.y + w.z * time);
}


void main()
{
    // get patch coordinate
    float u = gl_TessCoord.x;
    float v = gl_TessCoord.y;

    // ----------------------------------------------------------------------
    // retrieve control point position coordinates
    vec4 p00 = gl_in[0].gl_Position;
    vec4 p01 = gl_in[1].gl_Position;
    vec4 p10 = gl_in[2].gl_Position;
    vec4 p11 = gl_in[3].gl_Position;

    // bilinearly interpolate position coordinate across patch
    vec4 p0 = (p01 - p00) * v + p00;
    vec4 p1 = (p10 - p11) * v + p11;
    vec4 p = (p1 - p0) * u + p0;

	vOut.uv = vec2(u, v);


    // ----------------------------------------------------------------------
    // output patch point position in clip space
	vec4 worldPos = vec4(p * modelToWorld, 1.0f);

	// sin wave vertex displacement
    worldPos.y += h(wave[0]);
    worldPos.y += h(wave[1]);
    worldPos.y += h(wave[2]);
    worldPos.y += h(wave[3]);

	vOut.worldPos = worldPos;
	
	vec4 viewPos = vec4(worldPos * worldToView, 1.0f);
	gl_Position = projection * viewPos;
}
)"
};