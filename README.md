# Metal Ray Tracing with Motion Blur and MetalFX Upscaling

A GPU-accelerated path tracer built on Metal ray tracing, with motion blur, physically-based materials, and MetalFX upscaling.

## Overview

This project was initially based on Apple's sample code associated with WWDC22 session [10105: Maximize your Metal ray tracing performance](https://developer.apple.com/wwdc22/10105/) and WWDC21 session [10149: Enhance your app with Metal ray tracing](https://developer.apple.com/wwdc21/10149/). It has since been extended with several additional rendering features.

### Features

**Core ray tracing (from original Apple sample)**
- Path tracing with up to 3 bounce indirect lighting, using a Halton low-discrepancy sequence for sampling
- Instance motion blur via `MTLAccelerationStructureMotionInstanceDescriptor` with keyframed transforms
- Primitive motion blur via `MTLAccelerationStructureMotionTriangleGeometryDescriptor` with per-vertex keyframes, on devices that support it
- Cornell box scene with two animated Ninja characters (one using instance motion, one using primitive motion)
- Temporal accumulation for progressive refinement

**MetalFX upscaling**
- Renders at 50% of the display resolution and upscales to full resolution using MetalFX
- Supports both spatial scaling (`MTLFXSpatialScaler`) and temporal scaling (`MTLFXTemporalScaler`), selecting automatically based on device support
- Temporal scaling uses per-pixel motion vectors and linear depth written by the ray tracing kernel, derived from projecting the primary hit point through the previous frame's camera
- Option to disable MetalFX entirely to compare output quality

**Anisotropic GGX brushed metal material**
- Full PBR BRDF implemented in Metal shading language for a brushed metal torus in the scene
- Anisotropic GGX normal distribution function (Burley 2012)
- Height-correlated Smith masking-shadowing function (Heitz 2014)
- Schlick Fresnel approximation
- Importance sampling of the anisotropic GGX distribution for the indirect bounce
- Brushing tangent direction encoded per-vertex in the color channel, oriented along the torus ring

**Camera controls (macOS)**
- Mouse drag to orbit around the scene
- Scroll wheel to zoom in and out
- W/A/S/D keys to pan the orbit target

## Requirements

* macOS 13 or later
* iOS 16 or later
* Xcode 14 or later
* A GPU that supports Metal ray tracing (`supportsRaytracing`)
* MetalFX upscaling requires a compatible GPU (spatial scaling via `MTLFXSpatialScalerDescriptor.supportsDevice:`, temporal scaling via `MTLFXTemporalScalerDescriptor.supportsDevice:`)
