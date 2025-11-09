//! Simple example demonstrating PSO optimization on the sphere function
//!
//! Run this example with:
//! ```
//! cargo run --example simple_optimization
//! ```

use particle_swarm_opt::{Hyperparameters, Swarm};

fn main() {
    // Define a simple fitness function: sphere function (sum of squares)
    // The optimal value is 0 at position [0, 0, 0, ...]
    let fitness_fn = |position: &[f64]| -> f64 {
        position.iter().map(|x| x * x).sum()
    };

    // Create hyperparameters for a 5-dimensional optimization problem
    let hyperparams = Hyperparameters::new(
        30,      // swarm_size: 30 particles
        5,       // dimensions: 5D search space
        -10.0,   // lower_bound for all dimensions
        10.0,    // upper_bound for all dimensions
    );

    println!("Initializing PSO with {} particles in {} dimensions",
             hyperparams.swarm_size,
             hyperparams.dimensions());
    println!("Search space: [{}, {}]^{}",
             hyperparams.lower_bounds[0],
             hyperparams.upper_bounds[0],
             hyperparams.dimensions());
    println!();

    // Initialize the swarm
    let swarm = Swarm::new(hyperparams, fitness_fn);

    println!("Swarm initialized successfully!");
    println!("Number of particles: {}", swarm.size());
    println!("Global best fitness: {:.6}", swarm.global_best_fitness);
    println!("Global best position: {:?}", swarm.global_best_position);
    println!();

    // Display some particle information
    println!("Sample particle positions:");
    for (i, particle) in swarm.particles.iter().take(5).enumerate() {
        println!("  Particle {}: position={:?}, fitness={:.6}",
                 i,
                 particle.position,
                 particle.fitness);
    }
    println!();

    println!("Phase 1 complete: Core engine foundation established!");
}
