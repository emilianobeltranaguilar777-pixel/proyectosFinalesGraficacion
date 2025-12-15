# AR Filter - Neon Mask (Snapchat-style)

## Overview

This is a **standalone AR filter module** that creates a Snapchat-style face filter using:
- **MediaPipe FaceMesh** for face detection and landmark tracking
- **OpenGL + GLFW** for rendering geometric primitives
- **Real-time animations** that respond to facial expressions

## Features

- **Neon Mask Effect**: Geometric shapes overlaid on the face with animated neon colors
- **Expression Detection**: Filter intensity increases when you open your mouth
- **Head Tracking**: Filter elements follow head tilt and rotation
- **Smooth Animations**: Pulsing colors, animated decorations
- **High Performance**: Targets >25 FPS on Ubuntu Linux

## Module Structure

```
ar_filter/
├── __init__.py         # Module entry point
├── README.md           # This file
├── face_tracker.py     # MediaPipe FaceMesh wrapper
├── metrics.py          # Pure math functions (testable)
├── primitives.py       # Geometry generators (testable)
├── gl_app.py           # Main OpenGL application
└── shaders/
    ├── basic.vert      # Vertex shader
    └── basic.frag      # Fragment shader
```

## Architecture

### face_tracker.py
- Encapsulates MediaPipe FaceMesh
- Returns ONLY normalized landmarks and bounding box
- No rendering, no OpenGL

### metrics.py
- Pure mathematical functions
- `face_width()`, `mouth_openness()`, `head_tilt()`, etc.
- Fully testable without hardware

### primitives.py
- Generates vertex data for shapes
- Circles, horns, stars, zigzag lines
- Returns numpy arrays, NO OpenGL calls

### gl_app.py
- Complete standalone OpenGL application
- Initializes GLFW window
- Compiles and uses shaders
- Main render loop
- ESC key exits cleanly

## Usage

### From Python
```python
from ar_filter import run_ar_filter
run_ar_filter()
```

### From Command Line
```bash
cd ar_filter
python gl_app.py
```

### From Main Project (Integrated)
Press **[F]** key in the main menu to launch the AR filter.

## Dependencies

- `mediapipe` - Face detection
- `PyOpenGL` - OpenGL bindings
- `glfw` - Window management
- `numpy` - Array operations
- `opencv-python` - Camera capture and image processing

## Controls

| Key | Action |
|-----|--------|
| **ESC** | Exit filter |
| **Open mouth** | Increase effect intensity |
| **Tilt head** | Filter follows rotation |

## Filter Effects

1. **Forehead Decorations**: Animated circles above eyebrows
2. **Curved Horns**: Geometric horns that follow head tilt
3. **Eye Accents**: Zigzag lines extending from eyes
4. **Nose Bridge**: Neon lines along nose
5. **Stars**: Appear when mouth is open
6. **Cheek Circles**: Pulsing decorations on cheeks

## Color Animation

The filter cycles through neon colors:
- Cyan
- Magenta
- Yellow
- Green
- Blue
- Pink
- Orange

## Performance Notes

- Targets 30 FPS camera capture
- Optimized for >25 FPS rendering
- No textures, no FBO, no blur effects
- Simple shaders for maximum compatibility

## Testing

Run tests from project root:
```bash
pytest tests/test_ar_metrics.py -v
pytest tests/test_ar_primitives.py -v
pytest tests/test_ar_filter_smoke.py -v
```

## Isolation Guarantee

This module is **completely independent**:
- Does NOT import `main.py`
- Does NOT use `Gesture3D.py`
- Does NOT use `ColorPainter.py`
- Does NOT use `NeonMenu`
- Does NOT share camera with main project
- Exiting returns everything intact

## Academic Notes

This filter demonstrates:
1. **Computer Vision**: Real-time face landmark detection with MediaPipe
2. **Computer Graphics**: OpenGL primitive rendering with shaders
3. **Animation**: Time-based and expression-based animations
4. **Software Architecture**: Clean separation of concerns (tracking, math, geometry, rendering)

## Troubleshooting

### No face detected
- Ensure good lighting
- Face the camera directly
- Check that MediaPipe is installed

### Low FPS
- Close other applications
- Reduce window size
- Check GPU drivers

### OpenGL errors
- Ensure PyOpenGL and glfw are installed
- Verify OpenGL 3.3+ support
- Try: `pip install PyOpenGL PyOpenGL_accelerate glfw`
