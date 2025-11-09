//! Simple Visualization Example
//!
//! A minimal example showing how to create basic visualizations
//! after running a PSO optimization.
//!
//! Run with:
//! ```bash
//! cargo run --example simple_visualization
//! ```

use particle_swarm_opt::visualization::{plot_convergence, plot_2d_landscape};
use particle_swarm_opt::{Hyperparameters, Swarm};

fn main() {
    println!("=== Simple PSO Visualization Example ===\n");

    // Define a simple 2D fitness function (sphere function)
    let fitness_fn = |position: &[f64]| -> f64 {
        position.iter().map(|x| x * x).sum()
    };

    // Create hyperparameters
    let hyperparams = Hyperparameters::new(
        30,     // 30 particles
        2,      // 2D problem
        -10.0,  // lower bound
        10.0,   // upper bound
    );

    // Initialize swarm and run optimization
    println!("Running optimization...");
    let mut swarm = Swarm::new(hyperparams, &fitness_fn);
    let history = swarm.optimize(&fitness_fn, 50);

    println!("Optimization complete!");
    println!("  Final best fitness: {:.6}", swarm.global_best_fitness);
    println!("  Best position: [{:.4}, {:.4}]\n",
             swarm.global_best_position[0],
             swarm.global_best_position[1]);

    // Create output directory
    std::fs::create_dir_all("output").expect("Failed to create output directory");

    // 1. Plot convergence curve
    println!("Generating convergence plot...");
    plot_convergence(
        &history,
        "output/simple_convergence.png",
        Some("Sphere Function Optimization"),
        None,
    )
    .expect("Failed to create convergence plot");
    println!("  ✓ Saved to output/simple_convergence.png");

    // 2. Plot initial landscape
    println!("Generating initial landscape plot...");
    let bounds = (-10.0, 10.0, -10.0, 10.0);
    plot_2d_landscape(
        &fitness_fn,
        &history,
        0, // iteration 0
        "output/simple_landscape_initial.png",
        bounds,
        Some("Initial Swarm State"),
        None,
    )
    .expect("Failed to create initial landscape plot");
    println!("  ✓ Saved to output/simple_landscape_initial.png");

    // 3. Plot final landscape
    println!("Generating final landscape plot...");
    plot_2d_landscape(
        &fitness_fn,
        &history,
        history.len() - 1, // last iteration
        "output/simple_landscape_final.png",
        bounds,
        Some("Final Swarm State"),
        None,
    )
    .expect("Failed to create final landscape plot");
    println!("  ✓ Saved to output/simple_landscape_final.png");

    println!("\n=== Visualization Complete ===");
    println!("Check the 'output' directory for generated plots!");
}
