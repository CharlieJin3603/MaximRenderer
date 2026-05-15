/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
The header that contains the types and enumeration constants that the Metal shaders and the C/Objective-C source share.
*/

#ifndef ShaderTypes_h
#define ShaderTypes_h

#include <simd/simd.h>

#define GEOMETRY_MASK_TRIANGLE      1
#define GEOMETRY_MASK_SPHERE        2
#define GEOMETRY_MASK_LIGHT         4
#define GEOMETRY_MASK_UNLIT         8
#define GEOMETRY_MASK_BRUSHED_METAL 16

#define GEOMETRY_MASK_GEOMETRY (GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_SPHERE)

#define RAY_MASK_PRIMARY   (GEOMETRY_MASK_GEOMETRY | GEOMETRY_MASK_LIGHT | GEOMETRY_MASK_UNLIT | GEOMETRY_MASK_BRUSHED_METAL)
#define RAY_MASK_SHADOW    (GEOMETRY_MASK_GEOMETRY | GEOMETRY_MASK_BRUSHED_METAL)
#define RAY_MASK_SECONDARY (GEOMETRY_MASK_GEOMETRY | GEOMETRY_MASK_UNLIT | GEOMETRY_MASK_BRUSHED_METAL)

struct Camera
{
    vector_float3 position;
    vector_float3 right;
    vector_float3 up;
    vector_float3 forward;
};

struct AreaLight
{
    vector_float3 position;
    vector_float3 forward;
    vector_float3 right;
    vector_float3 up;
    vector_float3 color;
};

struct FrameData
{
    unsigned int width;
    unsigned int height;
    unsigned int frameIndex;
    Camera camera;
    unsigned int lightCount;
};

struct Sphere
{
    vector_float3 origin;
    float radius;
    vector_float3 color;
};

#endif
