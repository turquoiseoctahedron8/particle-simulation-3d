import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math

class Particle:
    """Represents a 3D particle with position, velocity, and mass."""
    
    def __init__(self, position, velocity, mass=1.0):
        self.position = np.array(position, dtype=np.float32)
        self.velocity = np.array(velocity, dtype=np.float32)
        self.acceleration = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.mass = mass
        self.radius = 0.05
    
    def apply_force(self, force):
        """Apply force to particle: F = ma => a = F/m"""
        self.acceleration += force / self.mass
    
    def update(self, dt):
        """Update velocity and position using Verlet integration."""
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt
        self.acceleration = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
    def draw(self):
        """Draw particle as a sphere."""
        glPushMatrix()
        glTranslatef(*self.position)
        quad = GLUquadric()
        glColor3f(1.0, 0.5, 0.0)
        gluSphere(quad, self.radius, 16, 16)
        glPopMatrix()


class ParticleSystem:
    """Manages a system of particles with physics simulation."""
    
    def __init__(self, num_particles=50):
        self.particles = []
        self.gravity = np.array([0.0, -9.8, 0.0], dtype=np.float32)
        self.damping = 0.99
        self.bounds = 5.0
        self.collision_enabled = True
        
        # Initialize particles with random positions and velocities
        for _ in range(num_particles):
            pos = np.random.uniform(-2, 2, 3).astype(np.float32)
            vel = np.random.uniform(-1, 1, 3).astype(np.float32)
            mass = np.random.uniform(0.5, 2.0)
            self.particles.append(Particle(pos, vel, mass))
    
    def apply_forces(self):
        """Apply gravity to all particles."""
        for particle in self.particles:
            particle.apply_force(self.gravity * particle.mass)
    
    def handle_collisions(self):
        """Handle particle-particle collisions."""
        if not self.collision_enabled:
            return
        
        for i, p1 in enumerate(self.particles):
            for p2 in self.particles[i+1:]:
                diff = p2.position - p1.position
                distance = np.linalg.norm(diff)
                min_distance = p1.radius + p2.radius
                
                if distance < min_distance and distance > 0:
                    # Elastic collision
                    normal = diff / distance
                    relative_vel = p1.velocity - p2.velocity
                    vel_along_normal = np.dot(relative_vel, normal)
                    
                    if vel_along_normal > 0:  # Objects moving towards each other
                        # Separate particles
                        overlap = min_distance - distance
                        p1.position -= normal * (overlap / 2)
                        p2.position += normal * (overlap / 2)
                        
                        # Exchange velocities along collision normal
                        p1.velocity -= vel_along_normal * normal
                        p2.velocity += vel_along_normal * normal
    
    def handle_boundaries(self):
        """Keep particles within bounds with bounce."""
        for particle in self.particles:
            for axis in range(3):
                if particle.position[axis] > self.bounds:
                    particle.position[axis] = self.bounds
                    particle.velocity[axis] *= -0.8  # Damped bounce
                elif particle.position[axis] < -self.bounds:
                    particle.position[axis] = -self.bounds
                    particle.velocity[axis] *= -0.8
    
    def update(self, dt):
        """Update all particles in the system."""
        self.apply_forces()
        self.handle_collisions()
        
        for particle in self.particles:
            particle.velocity *= self.damping
            particle.update(dt)
        
        self.handle_boundaries()
    
    def draw(self):
        """Draw all particles."""
        for particle in self.particles:
            particle.draw()
    
    def add_particle(self, position, velocity):
        """Add a new particle to the system."""
        self.particles.append(Particle(position, velocity))
    
    def clear(self):
        """Remove all particles."""
        self.particles.clear()


def draw_grid(size=10, step=1):
    """Draw a reference grid."""
    glColor3f(0.5, 0.5, 0.5)
    glBegin(GL_LINES)
    for i in range(-size, size+1, step):
        glVertex3f(i, -3, -size)
        glVertex3f(i, -3, size)
        glVertex3f(-size, -3, i)
        glVertex3f(size, -3, i)
    glEnd()


def main():
    """Main simulation loop."""
    pygame.init()
    display = (1200, 800)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("3D Particle Simulation")
    
    # OpenGL settings
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    
    # Set up perspective
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0, -1, -12)
    
    # Set up lighting
    glLight(GL_LIGHT0, GL_POSITION, (5, 5, 5, 0))
    glLight(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1))
    glLight(GL_LIGHT0, GL_DIFFUSE, (1, 1, 1, 1))
    
    # Create particle system
    system = ParticleSystem(num_particles=100)
    
    clock = pygame.time.Clock()
    running = True
    paused = False
    rotation = 0
    
    print("Controls:")
    print("  SPACE - Pause/Resume")
    print("  R - Reset simulation")
    print("  LEFT/RIGHT - Rotate view")
    print("  UP/DOWN - Zoom in/out")
    print("  ESC - Exit")
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    system.clear()
                    system = ParticleSystem(num_particles=100)
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            rotation -= 2
        if keys[pygame.K_RIGHT]:
            rotation += 2
        if keys[pygame.K_UP]:
            glTranslatef(0, 0, 0.2)
        if keys[pygame.K_DOWN]:
            glTranslatef(0, 0, -0.2)
        
        # Update simulation
        if not paused:
            system.update(1/60)
        
        # Render
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0, -1, -12)
        glRotatef(rotation, 0, 1, 0)
        
        draw_grid()
        system.draw()
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()


if __name__ == "__main__":
    main()
