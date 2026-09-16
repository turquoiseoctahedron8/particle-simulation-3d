# 3D Particle Simulation

A physics-based 3D particle simulation built with Python, Pygame, and OpenGL.

## Features

- **100 interactive 3D particles** with physics simulation
- **Gravity and damping** for realistic motion
- **Particle-particle collision detection** with elastic responses
- **Boundary collisions** with bounce behavior
- **Real-time 3D visualization** using OpenGL
- **Interactive camera controls** for exploring the simulation

## Requirements

- Python 3.8+
- numpy
- pygame
- PyOpenGL

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python particle_simulation.py
```

## Controls

- **SPACE** - Pause/Resume simulation
- **R** - Reset (clear all particles and restart)
- **LEFT/RIGHT ARROW** - Rotate view
- **UP/DOWN ARROW** - Zoom in/out
- **ESC** - Exit

## How It Works

### Physics System
- Each particle has mass, position, velocity, and acceleration
- Forces (gravity) are applied each frame
- Velocity Verlet integration is used for accurate physics
- Air damping simulates friction

### Collision Handling
- Particle-particle collisions detected via distance checks
- Elastic collision response exchanges momentum
- Boundary collisions bounce particles with damping
- Particles stay within a bounded region

### Visualization
- Particles rendered as 3D spheres
- Reference grid for spatial orientation
- Real-time rendering at 60 FPS
- Smooth camera rotation and zoom

## Future Enhancements

- Spatial partitioning for better collision performance
- Different particle forces (magnetism, repulsion)
- Particle trails/history visualization
- Custom particle spawning
- Different collision materials with varying friction

---

Created to demonstrate 3D physics simulation capabilities!
