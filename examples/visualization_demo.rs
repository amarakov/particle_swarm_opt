//! Visualization Demo for PSO
//!
//! This example demonstrates all visualization capabilities:
//! - Convergence curves showing fitness improvement
//! - 2D landscape contour plots with particle positions
//! - Animation frame generation for swarm movement
//!
//! Run this example with:
//! ```bash
//! cargo run --example visualization_demo
//! ```

use particle_swarm_opt::visualization::generate_all_plots;
use particle_swarm_opt::{Hyperparameters, Swarm};

/// Rastrigin function - a common benchmark with many local minima
///
/// Global minimum is at (0, 0) with value 0
/// Formula: f(x) = A*n + Σ[x_i^2 - A*cos(2π*x_i)]
/// where A = 10 and n is the number of dimensions
fn rastrigin(position: &[f64]) -> f64 {
    let a = 10.0;
    let n = position.len() as f64;

    a * n + position.iter()
        .map(|x| x * x - a * (2.0 * std::f64::consts::PI * x).cos())
        .sum::<f64>()
}

/// Sphere function - simple convex function
///
/// Global minimum is at (0, 0, ...) with value 0
/// Formula: f(x) = Σ[x_i^2]
fn sphere(position: &[f64]) -> f64 {
    position.iter().map(|x| x * x).sum()
}

/// Rosenbrock function - narrow valley, difficult to optimize
///
/// Global minimum is at (1, 1) with value 0
/// Formula: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
fn rosenbrock(position: &[f64]) -> f64 {
    let x = position[0];
    let y = position[1];
    (1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2)
}

fn main() {
    println!("=== PSO Visualization Demo ===\n");

    // Example 1: Rastrigin function (challenging with many local minima)
    println!("Example 1: Rastrigin Function");
    println!("------------------------------");
    optimize_and_visualize(
        rastrigin,
        "Rastrigin",
        (-5.12, 5.12),
        100,
        50,
        "output/rastrigin",
        false, // Don't generate animation for first example (too many frames)
    );

    // Example 2: Sphere function (simple, convex)
    println!("\nExample 2: Sphere Function");
    println!("--------------------------");
    optimize_and_visualize(
        sphere,
        "Sphere",
        (-10.0, 10.0),
        50,
        30,
        "output/sphere",
        true, // Generate animation for this one
    );

    // Example 3: Rosenbrock function (narrow valley)
    println!("\nExample 3: Rosenbrock Function");
    println!("------------------------------");
    optimize_and_visualize(
        rosenbrock,
        "Rosenbrock",
        (-2.0, 2.0),
        100,
        50,
        "output/rosenbrock",
        false,
    );

    println!("\n=== All Visualizations Complete ===");
    println!("\nGenerated files:");
    println!("  - Convergence plots: output/*/convergence.png");
    println!("  - Landscape plots: output/*/landscape_*.png");
    println!("  - Animation frames: output/sphere/animation_frames/");
    println!("\nTo create an animation GIF from the sphere example:");
    println!("  ffmpeg -framerate 10 -pattern_type glob -i 'output/sphere/animation_frames/*.png' -vf \"scale=800:-1\" sphere_animation.gif");
}

/// Helper function to optimize and generate all visualizations
fn optimize_and_visualize<F>(
    fitness_fn: F,
    _name: &str,
    bounds: (f64, f64),
    iterations: usize,
    swarm_size: usize,
    output_dir: &str,
    generate_animation: bool,
) where
    F: Fn(&[f64]) -> f64 + Sync,
{
    // Setup
    let hyperparams = Hyperparameters::new(swarm_size, 2, bounds.0, bounds.1);

    println!("  Configuration:");
    println!("    Particles: {}", swarm_size);
    println!("    Dimensions: 2D");
    println!("    Bounds: [{}, {}]", bounds.0, bounds.1);
    println!("    Iterations: {}", iterations);

    // Initialize and optimize
    println!("  Initializing swarm...");
    let mut swarm = Swarm::new(hyperparams, &fitness_fn);
    let initial_fitness = swarm.global_best_fitness;

    println!("  Running optimization...");
    let history = swarm.optimize(&fitness_fn, iterations);

    let final_fitness = swarm.global_best_fitness;
    let improvement = ((initial_fitness - final_fitness) / initial_fitness) * 100.0;

    println!("  Results:");
    println!("    Initial fitness: {:.6}", initial_fitness);
    println!("    Final fitness: {:.6}", final_fitness);
    println!("    Improvement: {:.2}%", improvement);
    println!("    Best position: [{:.4}, {:.4}]",
             swarm.global_best_position[0],
             swarm.global_best_position[1]);

    // Generate visualizations
    println!("  Generating visualizations...");
    let plot_bounds = (bounds.0, bounds.1, bounds.0, bounds.1);

    match generate_all_plots(
        &fitness_fn,
        &history,
        output_dir,
        Some(plot_bounds),
        generate_animation,
        None,
    ) {
        Ok(_) => println!("  ✓ Visualizations saved to {}/", output_dir),
        Err(e) => eprintln!("  Error generating plots: {}", e),
    }
}
