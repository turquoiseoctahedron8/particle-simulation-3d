import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import HTML
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

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
    
    def get_position(self):
        """Return current position."""
        return self.position.copy()


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
    
    def get_positions(self):
        """Get all particle positions for visualization."""
        positions = np.array([p.get_position() for p in self.particles])
        return positions
    
    def reset(self):
        """Reset all particles."""
        self.particles.clear()
        for _ in range(len(self.particles)):
            pos = np.random.uniform(-2, 2, 3).astype(np.float32)
            vel = np.random.uniform(-1, 1, 3).astype(np.float32)
            mass = np.random.uniform(0.5, 2.0)
            self.particles.append(Particle(pos, vel, mass))


def run_simulation_colab(num_particles=100, num_frames=300, save_animation=False):
    """
    Run particle simulation with matplotlib visualization.
    
    Args:
        num_particles: Number of particles to simulate
        num_frames: Number of frames to simulate
        save_animation: If True, saves animation as mp4 (requires ffmpeg)
    """
    
    # Create particle system
    system = ParticleSystem(num_particles=num_particles)
    dt = 1/60  # 60 FPS
    
    # Create figure and axis
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set up plot limits
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_zlim(-6, 6)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Particle Simulation (Matplotlib)')
    
    # Create scatter plot
    scatter = ax.scatter([], [], [], c='red', marker='o', s=50, alpha=0.6)
    
    # Store positions history for plotting
    positions_list = []
    
    def update_frame(frame):
        """Update function for animation."""
        # Simulate one step
        system.update(dt)
        
        # Get positions
        positions = system.get_positions()
        positions_list.append(positions.copy())
        
        # Update scatter plot
        scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
        
        # Rotate view for better visualization
        ax.view_init(elev=20, azim=frame % 360)
        
        return [scatter]
    
    print(f"Creating animation with {num_frames} frames...")
    
    # Create animation
    anim = FuncAnimation(
        fig, 
        update_frame, 
        frames=num_frames,
        interval=50,  # 50ms per frame
        blit=True,
        repeat=True
    )
    
    # For Colab, display as HTML video
    try:
        return HTML(anim.to_jshtml())
    except:
        # Fallback: save as gif if HTML doesn't work
        print("Saving animation as GIF...")
        anim.save('particle_simulation.gif', writer='pillow', fps=20)
        plt.close()
        from IPython.display import Image
        return Image('particle_simulation.gif')


def run_simulation_static(num_particles=100, num_steps=500):
    """
    Run simulation and create a static visualization showing particle positions over time.
    Better for limited environments.
    """
    
    system = ParticleSystem(num_particles=num_particles)
    dt = 1/60
    
    # Store positions at different time steps
    time_steps = [0, num_steps//4, num_steps//2, 3*num_steps//4, num_steps]
    stored_positions = {}
    
    for step in range(num_steps + 1):
        system.update(dt)
        if step in time_steps:
            stored_positions[step] = system.get_positions()
    
    # Create 2x3 subplot showing evolution
    fig = plt.figure(figsize=(15, 10))
    
    for idx, step in enumerate(time_steps[:5]):  # Show 5 snapshots
        ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
        positions = stored_positions[step]
        
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                  c='red', marker='o', s=30, alpha=0.6)
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_zlim(-6, 6)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Time Step: {step}')
    
    plt.tight_layout()
    plt.show()
    
    return fig


# Example usage for Colab:
if __name__ == "__main__":
    print("3D Particle Simulation - Matplotlib Version")
    print("=" * 50)
    
    # For Jupyter/Colab notebooks:
    # animation = run_simulation_colab(num_particles=100, num_frames=300)
    # display(animation)
    
    # For static visualization:
    fig = run_simulation_static(num_particles=100, num_steps=300)
