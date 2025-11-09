//! Advanced PSO Features Example
//!
//! This example demonstrates Phase 5 advanced features:
//! 1. Adaptive Inertia Weight (linearly decreasing)
//! 2. Local Best (lBest) Topology with ring neighborhood
//! 3. Pluggable strategies for initialization and boundary handling
//!
//! Run with: cargo run --example advanced_features

use particle_swarm_opt::{
    strategies::*,
    Hyperparameters, Swarm,
};

/// Rastrigin function - a challenging multimodal benchmark function
///
/// Global minimum at f(0, 0, ..., 0) = 0
/// Many local minima make this ideal for testing robustness
fn rastrigin(position: &[f64]) -> f64 {
    let a = 10.0;
    let n = position.len() as f64;

    a * n + position.iter()
        .map(|&x| x.powi(2) - a * (2.0 * std::f64::consts::PI * x).cos())
        .sum::<f64>()
}

/// Rosenbrock function - another challenging benchmark
///
/// Global minimum at f(1, 1, ..., 1) = 0
/// Narrow valley makes convergence difficult
fn rosenbrock(position: &[f64]) -> f64 {
    position.windows(2)
        .map(|w| {
            let x = w[0];
            let y = w[1];
            100.0 * (y - x.powi(2)).powi(2) + (1.0 - x).powi(2)
        })
        .sum()
}

fn main() {
    println!("=== Advanced PSO Features Demo ===\n");

    // Problem configuration
    let dimensions = 5;
    let swarm_size = 50;
    let max_iterations = 200;
    let lower_bound = -5.0;
    let upper_bound = 5.0;

    println!("Problem: Rastrigin function ({}D)", dimensions);
    println!("Swarm size: {}", swarm_size);
    println!("Max iterations: {}\n", max_iterations);

    // ========================================
    // Example 1: Standard PSO (baseline)
    // ========================================
    println!("--- Example 1: Standard PSO (Baseline) ---");
    println!("Configuration:");
    println!("  - Constant inertia: 0.7");
    println!("  - Global best topology");
    println!("  - Uniform initialization");
    println!("  - Reflect boundary handling");

    let hyperparams_standard = Hyperparameters::new(
        swarm_size,
        dimensions,
        lower_bound,
        upper_bound,
    );

    let mut swarm_standard = Swarm::new(hyperparams_standard, &rastrigin);
    let history_standard = swarm_standard.optimize(&rastrigin, max_iterations);

    println!("\nResults:");
    println!("  Best fitness: {:.6}", swarm_standard.global_best_fitness);
    println!("  Best position: {:?}", swarm_standard.global_best_position.iter()
        .map(|x| format!("{:.4}", x))
        .collect::<Vec<_>>());

    // ========================================
    // Example 2: Advanced PSO with all Phase 5 features
    // ========================================
    println!("\n--- Example 2: Advanced PSO (Phase 5 Features) ---");
    println!("Configuration:");
    println!("  - Linearly decreasing inertia: 0.9 → 0.4");
    println!("  - Local best topology (ring with neighborhood size 2)");
    println!("  - Uniform initialization");
    println!("  - Reflect boundary handling");

    let hyperparams_advanced = Hyperparameters::advanced(
        swarm_size,
        dimensions,
        lower_bound,
        upper_bound,
    );

    let mut swarm_advanced = Swarm::new(hyperparams_advanced, &rastrigin);
    let history_advanced = swarm_advanced.optimize(&rastrigin, max_iterations);

    println!("\nResults:");
    println!("  Best fitness: {:.6}", swarm_advanced.global_best_fitness);
    println!("  Best position: {:?}", swarm_advanced.global_best_position.iter()
        .map(|x| format!("{:.4}", x))
        .collect::<Vec<_>>());

    // Compare improvement
    let improvement = ((swarm_standard.global_best_fitness - swarm_advanced.global_best_fitness)
        / swarm_standard.global_best_fitness) * 100.0;
    println!("\nImprovement over baseline: {:.2}%", improvement);

    // ========================================
    // Example 3: Custom strategy configuration
    // ========================================
    println!("\n--- Example 3: Custom Strategies ---");
    println!("Configuration:");
    println!("  - Linearly decreasing inertia: 0.95 → 0.3");
    println!("  - Local best topology (ring with neighborhood size 3)");
    println!("  - Chaotic initialization (mu=4.0)");
    println!("  - Absorb boundary handling");

    let lower_bounds = vec![lower_bound; dimensions];
    let upper_bounds = vec![upper_bound; dimensions];

    let hyperparams_custom = Hyperparameters::with_strategies(
        swarm_size,
        lower_bounds,
        upper_bounds,
        Box::new(ChaoticInitialization::new()),
        Box::new(AbsorbBoundary),
        Box::new(LinearlyDecreasingInertia::with_range(0.95, 0.3)),
        Box::new(LocalBest::with_size(3)),
    );

    let mut swarm_custom = Swarm::new(hyperparams_custom, &rastrigin);
    let history_custom = swarm_custom.optimize(&rastrigin, max_iterations);

    println!("\nResults:");
    println!("  Best fitness: {:.6}", swarm_custom.global_best_fitness);
    println!("  Best position: {:?}", swarm_custom.global_best_position.iter()
        .map(|x| format!("{:.4}", x))
        .collect::<Vec<_>>());

    // ========================================
    // Example 4: Testing on different function (Rosenbrock)
    // ========================================
    println!("\n--- Example 4: Rosenbrock Function with Advanced Features ---");
    println!("Configuration:");
    println!("  - Linearly decreasing inertia: 0.9 → 0.4");
    println!("  - Local best topology (ring with neighborhood size 2)");

    let hyperparams_rosenbrock = Hyperparameters::advanced(
        swarm_size,
        dimensions,
        -2.0,  // Tighter bounds for Rosenbrock
        2.0,
    );

    let mut swarm_rosenbrock = Swarm::new(hyperparams_rosenbrock, &rosenbrock);
    let _history_rosenbrock = swarm_rosenbrock.optimize(&rosenbrock, max_iterations);

    println!("\nResults:");
    println!("  Best fitness: {:.6}", swarm_rosenbrock.global_best_fitness);
    println!("  Best position: {:?}", swarm_rosenbrock.global_best_position.iter()
        .map(|x| format!("{:.4}", x))
        .collect::<Vec<_>>());
    println!("  (Optimal: all ones)");

    // ========================================
    // Example 5: Different boundary strategies
    // ========================================
    println!("\n--- Example 5: Comparing Boundary Strategies ---");

    // Test with Random boundary
    let hyperparams_random = Hyperparameters::with_strategies(
        swarm_size,
        vec![lower_bound; dimensions],
        vec![upper_bound; dimensions],
        Box::new(UniformInitialization),
        Box::new(RandomBoundary),
        Box::new(LinearlyDecreasingInertia::new()),
        Box::new(LocalBest::new()),
    );

    let mut swarm_random = Swarm::new(hyperparams_random, &rastrigin);
    swarm_random.optimize(&rastrigin, max_iterations);

    println!("Random boundary strategy:");
    println!("  Best fitness: {:.6}", swarm_random.global_best_fitness);

    // ========================================
    // Summary and visualization info
    // ========================================
    println!("\n=== Summary ===");
    println!("All Phase 5 features have been successfully demonstrated:");
    println!("  ✓ Adaptive Inertia Weight (linearly decreasing)");
    println!("  ✓ lBest Topology (ring neighborhood)");
    println!("  ✓ Pluggable Initialization Strategies (uniform, chaotic)");
    println!("  ✓ Pluggable Boundary Strategies (reflect, absorb, random)");
    println!("\nAdvanced features typically show improvement on complex problems");
    println!("with multiple local optima, as they maintain better diversity.");

    println!("\nConvergence history saved - {} iterations logged", history_advanced.len());
    println!("Use visualization examples to plot results!");
}
