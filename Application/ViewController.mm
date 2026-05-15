/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
The implementation of the cross-platform view controller.
*/
#import "ViewController.h"
#import "Renderer.h"

#import <simd/simd.h>

using namespace simd;

@implementation ViewController
{
    MTKView *_view;

    Renderer *_renderer;
    Scene *_scene;

    float _azimuth;       // horizontal orbit angle (radians)
    float _elevation;     // vertical orbit angle (radians)
    float _cameraRadius;  // distance from orbitTarget
    vector_float3 _orbitTarget;
}

- (void)viewDidLoad
{
    [super viewDidLoad];

    _view = (MTKView *)self.view;

#if TARGET_OS_IPHONE
    _view.device = MTLCreateSystemDefaultDevice();
#else
    NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();

    id<MTLDevice> selectedDevice;

    for(id<MTLDevice> device in devices)
    {
        if(device.supportsRaytracing)
        {
            if(!selectedDevice || !device.isLowPower)
            {
                selectedDevice = device;
            }
        }
    }
    _view.device = selectedDevice;

    NSLog(@"Selected Device: %@", _view.device.name);
#endif

    // The device must support Metal and ray tracing.
    NSAssert(_view.device && _view.device.supportsRaytracing,
             @"Ray tracing isn't supported on this device");
    
#if defined(MTL_SUPPORT_PRIMITIVE_MOTION_QUERY) && MTL_SUPPORT_PRIMITIVE_MOTION_QUERY
    BOOL usePrimitiveMotion = _view.device.supportsPrimitiveMotionBlur;
#else
    BOOL usePrimitiveMotion = false;
#endif

#if TARGET_OS_IPHONE
    _view.backgroundColor = UIColor.blackColor;
#endif
    _view.colorPixelFormat = MTLPixelFormatRGBA16Float;

    Scene *scene = [Scene newMotionBlurSceneWithDevice:_view.device
                                    usePrimitiveMotion:usePrimitiveMotion];

    _scene = scene;

    // Derive initial orbit parameters from the scene's camera.
    vector_float3 offset = _scene.cameraPosition - _scene.cameraTarget;
    _orbitTarget  = _scene.cameraTarget;
    _cameraRadius = simd_length(offset);
    _elevation    = asinf(offset.y / _cameraRadius);
    _azimuth      = atan2f(offset.x, offset.z);

    _renderer = [[Renderer alloc] initWithDevice:_view.device
                                           scene:scene
                              usePrimitiveMotion:usePrimitiveMotion];

    [_renderer mtkView:_view drawableSizeWillChange:_view.bounds.size];

    _view.delegate = _renderer;
}

- (void)updateCamera
{
    float x = _cameraRadius * cosf(_elevation) * sinf(_azimuth);
    float y = _cameraRadius * sinf(_elevation);
    float z = _cameraRadius * cosf(_elevation) * cosf(_azimuth);
    _scene.cameraPosition = (vector_float3){ _orbitTarget.x + x, _orbitTarget.y + y, _orbitTarget.z + z };
    _scene.cameraTarget   = _orbitTarget;
    _scene.cameraUp       = (vector_float3){ 0.0f, 1.0f, 0.0f };
    [_renderer resetAccumulation];
}

#if !TARGET_OS_IPHONE

- (void)viewDidAppear
{
    [super viewDidAppear];
    [self.view.window makeFirstResponder:self];
}

- (BOOL)acceptsFirstResponder
{
    return YES;
}

- (void)mouseDragged:(NSEvent *)event
{
    _azimuth   += event.deltaX * 0.005f;
    _elevation -= event.deltaY * 0.005f;
    _elevation  = fmaxf(-(float)M_PI_2 + 0.01f, fminf((float)M_PI_2 - 0.01f, _elevation));
    [self updateCamera];
}

- (void)scrollWheel:(NSEvent *)event
{
    _cameraRadius -= event.scrollingDeltaY * 0.01f;
    _cameraRadius  = fmaxf(0.1f, _cameraRadius);
    [self updateCamera];
}

- (void)keyDown:(NSEvent *)event
{
    NSString *chars = event.charactersIgnoringModifiers;
    if (chars.length == 0) { [super keyDown:event]; return; }

    float panSpeed = _cameraRadius * 0.02f;
    // Negate so forward points from the camera toward the scene (look direction).
    vector_float3 forward = simd_normalize((vector_float3){ -sinf(_azimuth), 0.0f, -cosf(_azimuth) });
    vector_float3 right   = simd_normalize((vector_float3){ cosf(_azimuth), 0.0f, -sinf(_azimuth) });

    unichar c = [chars characterAtIndex:0];
    if      (c == 'w' || c == 'W') { _orbitTarget += forward * panSpeed; }
    else if (c == 's' || c == 'S') { _orbitTarget -= forward * panSpeed; }
    else if (c == 'a' || c == 'A') { _orbitTarget -= right   * panSpeed; }
    else if (c == 'd' || c == 'D') { _orbitTarget += right   * panSpeed; }
    else { [super keyDown:event]; return; }

    [self updateCamera];
}

#endif

@end
