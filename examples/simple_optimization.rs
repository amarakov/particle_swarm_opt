//! Simple example demonstrating PSO optimization on the sphere function
//!
//! Run this example with:
//! ```
//! cargo run --example simple_optimization
//! ```

use particle_swarm_opt::{Hyperparameters, Swarm};

fn main() {
    println!("=== Particle Swarm Optimization Demo ===\n");

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

    println!("Configuration:");
    println!("  Particles: {}", hyperparams.swarm_size);
    println!("  Dimensions: {}", hyperparams.dimensions());
    println!("  Search space: [{}, {}]^{}",
             hyperparams.lower_bounds[0],
             hyperparams.upper_bounds[0],
             hyperparams.dimensions());
    println!("  Inertia weight: {}", hyperparams.inertia_weight);
    println!("  Cognitive coefficient: {}", hyperparams.cognitive_coeff);
    println!("  Social coefficient: {}", hyperparams.social_coeff);
    println!();

    // Initialize the swarm
    println!("Initializing swarm...");
    let mut swarm = Swarm::new(hyperparams, &fitness_fn);

    println!("Initial state:");
    println!("  Global best fitness: {:.6}", swarm.global_best_fitness);
    println!("  Global best position: [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
             swarm.global_best_position[0],
             swarm.global_best_position[1],
             swarm.global_best_position[2],
             swarm.global_best_position[3],
             swarm.global_best_position[4]);
    println!();

    // Run optimization
    println!("Starting optimization for 100 iterations...");
    let max_iterations = 100;
    let history = swarm.optimize(&fitness_fn, max_iterations);

    println!();
    println!("Optimization complete!");
    println!();

    // Display results
    println!("Final results:");
    println!("  Iterations: {}", history.len());
    println!("  Global best fitness: {:.10}", swarm.global_best_fitness);
    println!("  Global best position: [{:.6}, {:.6}, {:.6}, {:.6}, {:.6}]",
             swarm.global_best_position[0],
             swarm.global_best_position[1],
             swarm.global_best_position[2],
             swarm.global_best_position[3],
             swarm.global_best_position[4]);
    println!();

    // Show convergence progress
    println!("Convergence progress (every 10 iterations):");
    for (i, iter_data) in history.iterations.iter().enumerate() {
        if i % 10 == 0 || i == history.len() - 1 {
            println!("  Iteration {:3}: fitness = {:.10}",
                     iter_data.iteration,
                     iter_data.global_best_fitness);
        }
    }
    println!();

    // Calculate improvement
    let initial_fitness = history.iterations[0].global_best_fitness;
    let final_fitness = swarm.global_best_fitness;
    let improvement = ((initial_fitness - final_fitness) / initial_fitness) * 100.0;

    println!("Performance:");
    println!("  Initial fitness: {:.6}", initial_fitness);
    println!("  Final fitness: {:.10}", final_fitness);
    println!("  Improvement: {:.2}%", improvement);
    println!();

    println!("Phase 2 complete: Optimization loop implemented!");
}
