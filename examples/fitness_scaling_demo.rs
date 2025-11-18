//! Fitness Scaling Demonstration
//!
//! This example demonstrates how to use fitness scaling to improve optimization
//! performance on flat fitness landscapes, like those encountered with COSM MAXCUT
//! problems where fitness values are clustered in a narrow range.
//!
//! # Problem
//!
//! When fitness values are very similar (e.g., -5755.05 to -5761.20), the fitness
//! landscape appears flat to the optimizer. This makes it difficult for PSO to
//! distinguish between good and bad solutions and find productive search directions.
//!
//! # Solution
//!
//! Fitness scaling transforms the fitness values to amplify differences while
//! preserving their relative ordering. This makes the landscape more informative
//! to the optimizer.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example fitness_scaling_demo
//! ```

use particle_swarm_opt::fitness_scaling::{FitnessScaler, ScalingStrategy};
use particle_swarm_opt::{Hyperparameters, Swarm};

/// Simulates a COSM-like MAXCUT objective function with flat landscape
///
/// This function returns values clustered in a narrow range around -5755 to -5761,
/// similar to the real COSM MAXCUT problem. The differences between solutions are
/// very small (< 0.11%), making optimization challenging.
fn cosm_like_maxcut(params: &[f64]) -> f64 {
    // Base value (simulating a MAXCUT result around -5755)
    let base = -5755.0;

    // Add small variations based on parameters (typically 0 to 6 units)
    // This simulates the small differences seen in real COSM MAXCUT results
    let variation = params.iter()
        .enumerate()
        .map(|(i, &x)| {
            // Each parameter contributes a small amount to fitness
            let normalized = (x - 50.0) / 100.0; // Assuming bounds [0, 100]
            -normalized.powi(2) * (i + 1) as f64 * 0.3
        })
        .sum::<f64>();

    base + variation
}

fn main() {
    println!("======================================================================");
    println!("           FITNESS SCALING DEMONSTRATION FOR FLAT LANDSCAPES");
    println!("======================================================================\n");

    // Problem setup (matching the user's COSM configuration)
    let swarm_size = 40;
    let dimensions = 5;
    let max_iterations = 50;

    // Parameter bounds (example values, adjust based on your COSM parameters)
    let lower_bounds = vec![0.0, 0.0, 0.0, 0.0, 0.0];
    let upper_bounds = vec![1.0, 0.1, 50.0, 100.0, 200.0];

    println!("Configuration:");
    println!("  Swarm size:      {}", swarm_size);
    println!("  Dimensions:      {}", dimensions);
    println!("  Iterations:      {}\n", max_iterations);

    // ========================================================================
    // Run 1: WITHOUT fitness scaling (baseline)
    // ========================================================================
    println!("----------------------------------------------------------------------");
    println!("RUN 1: WITHOUT Fitness Scaling (Baseline)");
    println!("----------------------------------------------------------------------\n");

    let hyperparams_baseline = Hyperparameters::with_bounds(
        swarm_size,
        lower_bounds.clone(),
        upper_bounds.clone(),
    );

    let mut swarm_baseline = Swarm::new(hyperparams_baseline, &cosm_like_maxcut);
    let initial_fitness_baseline = swarm_baseline.global_best_raw_fitness;

    let history_baseline = swarm_baseline.optimize(&cosm_like_maxcut, max_iterations);
    let final_fitness_baseline = swarm_baseline.global_best_raw_fitness;
    let improvement_baseline = ((final_fitness_baseline - initial_fitness_baseline) / initial_fitness_baseline.abs()) * 100.0;

    println!("Results:");
    println!("  Initial fitness: {:.10}", initial_fitness_baseline);
    println!("  Final fitness:   {:.10}", final_fitness_baseline);
    println!("  Improvement:     {:.4}%\n", improvement_baseline);

    // Show convergence progress
    println!("Convergence Progress (sampled):");
    println!("{:>10} | {:>18} | {:>15}", "Iteration", "Best Fitness", "Improvement");
    println!("{}", "-".repeat(50));
    for (i, iter_data) in history_baseline.iterations.iter().enumerate() {
        if i % 10 == 0 || i == history_baseline.iterations.len() - 1 {
            let iter_improvement = ((iter_data.global_best_fitness - initial_fitness_baseline) / initial_fitness_baseline.abs()) * 100.0;
            println!("{:>10} | {:>18.10} | {:>14.4}%",
                i,
                iter_data.global_best_fitness,
                iter_improvement
            );
        }
    }
    println!();

    // ========================================================================
    // Run 2: WITH Exponential Scaling (beta = 0.01)
    // ========================================================================
    println!("----------------------------------------------------------------------");
    println!("RUN 2: WITH Exponential Fitness Scaling (beta = 0.01)");
    println!("----------------------------------------------------------------------\n");
    println!("Strategy: Exponential scaling amplifies small differences");
    println!("Formula:  scaled = exp(beta * (raw_fitness - min))");
    println!("Effect:   Differences are amplified exponentially\n");

    let hyperparams_exp = Hyperparameters::with_bounds(
        swarm_size,
        lower_bounds.clone(),
        upper_bounds.clone(),
    ).with_fitness_scaling(
        FitnessScaler::new(ScalingStrategy::Exponential { beta: 0.01 })
    );

    let mut swarm_exp = Swarm::new(hyperparams_exp, &cosm_like_maxcut);
    let initial_fitness_exp = swarm_exp.global_best_raw_fitness;

    let history_exp = swarm_exp.optimize(&cosm_like_maxcut, max_iterations);
    let final_fitness_exp = swarm_exp.global_best_raw_fitness;
    let improvement_exp = ((final_fitness_exp - initial_fitness_exp) / initial_fitness_exp.abs()) * 100.0;

    println!("Results:");
    println!("  Initial fitness: {:.10}", initial_fitness_exp);
    println!("  Final fitness:   {:.10}", final_fitness_exp);
    println!("  Improvement:     {:.4}%\n", improvement_exp);

    // Show convergence progress
    println!("Convergence Progress (sampled):");
    println!("{:>10} | {:>18} | {:>15}", "Iteration", "Best Fitness", "Improvement");
    println!("{}", "-".repeat(50));
    for (i, iter_data) in history_exp.iterations.iter().enumerate() {
        if i % 10 == 0 || i == history_exp.iterations.len() - 1 {
            let iter_improvement = ((iter_data.global_best_fitness - initial_fitness_exp) / initial_fitness_exp.abs()) * 100.0;
            println!("{:>10} | {:>18.10} | {:>14.4}%",
                i,
                iter_data.global_best_fitness,
                iter_improvement
            );
        }
    }
    println!();

    // ========================================================================
    // Run 3: WITH MinMax Scaling
    // ========================================================================
    println!("----------------------------------------------------------------------");
    println!("RUN 3: WITH MinMax Fitness Scaling");
    println!("----------------------------------------------------------------------\n");
    println!("Strategy: Min-Max scaling normalizes fitness to [0, 1] range");
    println!("Formula:  scaled = (raw_fitness - min) / (max - min)");
    println!("Effect:   Creates uniform distribution in [0, 1]\n");

    let hyperparams_minmax = Hyperparameters::with_bounds(
        swarm_size,
        lower_bounds.clone(),
        upper_bounds.clone(),
    ).with_fitness_scaling(
        FitnessScaler::new(ScalingStrategy::MinMax { target_max: 1.0 })
    );

    let mut swarm_minmax = Swarm::new(hyperparams_minmax, &cosm_like_maxcut);
    let initial_fitness_minmax = swarm_minmax.global_best_raw_fitness;

    let history_minmax = swarm_minmax.optimize(&cosm_like_maxcut, max_iterations);
    let final_fitness_minmax = swarm_minmax.global_best_raw_fitness;
    let improvement_minmax = ((final_fitness_minmax - initial_fitness_minmax) / initial_fitness_minmax.abs()) * 100.0;

    println!("Results:");
    println!("  Initial fitness: {:.10}", initial_fitness_minmax);
    println!("  Final fitness:   {:.10}", final_fitness_minmax);
    println!("  Improvement:     {:.4}%\n", improvement_minmax);

    // ========================================================================
    // Run 4: WITH Power Law Scaling (power = 2.0)
    // ========================================================================
    println!("----------------------------------------------------------------------");
    println!("RUN 4: WITH Power Law Fitness Scaling (power = 2.0)");
    println!("----------------------------------------------------------------------\n");
    println!("Strategy: Power law applies exponent to normalized differences");
    println!("Formula:  scaled = ((raw_fitness - min) / (max - min))^power");
    println!("Effect:   Power > 1 amplifies small differences, < 1 compresses\n");

    let hyperparams_power = Hyperparameters::with_bounds(
        swarm_size,
        lower_bounds.clone(),
        upper_bounds.clone(),
    ).with_fitness_scaling(
        FitnessScaler::new(ScalingStrategy::PowerLaw { power: 2.0 })
    );

    let mut swarm_power = Swarm::new(hyperparams_power, &cosm_like_maxcut);
    let initial_fitness_power = swarm_power.global_best_raw_fitness;

    let history_power = swarm_power.optimize(&cosm_like_maxcut, max_iterations);
    let final_fitness_power = swarm_power.global_best_raw_fitness;
    let improvement_power = ((final_fitness_power - initial_fitness_power) / initial_fitness_power.abs()) * 100.0;

    println!("Results:");
    println!("  Initial fitness: {:.10}", initial_fitness_power);
    println!("  Final fitness:   {:.10}", final_fitness_power);
    println!("  Improvement:     {:.4}%\n", improvement_power);

    // ========================================================================
    // Summary and Recommendations
    // ========================================================================
    println!("======================================================================");
    println!("                        SUMMARY & RECOMMENDATIONS");
    println!("======================================================================\n");

    println!("Final Best Fitness by Strategy:");
    println!("  No Scaling:       {:.10} (improvement: {:.4}%)", final_fitness_baseline, improvement_baseline);
    println!("  Exponential:      {:.10} (improvement: {:.4}%)", final_fitness_exp, improvement_exp);
    println!("  MinMax:           {:.10} (improvement: {:.4}%)", final_fitness_minmax, improvement_minmax);
    println!("  Power Law:        {:.10} (improvement: {:.4}%)\n", final_fitness_power, improvement_power);

    // Find best strategy
    let strategies = vec![
        ("No Scaling", final_fitness_baseline, improvement_baseline),
        ("Exponential", final_fitness_exp, improvement_exp),
        ("MinMax", final_fitness_minmax, improvement_minmax),
        ("Power Law", final_fitness_power, improvement_power),
    ];

    let best = strategies.iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();

    println!("Best Strategy: {} (fitness: {:.10})\n", best.0, best.1);

    println!("Recommendations for COSM MAXCUT optimization:");
    println!("  1. Exponential Scaling (beta = 0.001 - 0.1):");
    println!("     - Best for very flat landscapes with tiny differences");
    println!("     - Tune beta: larger values = more amplification");
    println!("     - Start with beta = 0.01 and adjust based on results\n");

    println!("  2. Power Law Scaling (power = 1.5 - 3.0):");
    println!("     - Good for moderately flat landscapes");
    println!("     - power > 1 amplifies differences");
    println!("     - More controlled than exponential scaling\n");

    println!("  3. MinMax Scaling:");
    println!("     - Simple normalization to [0, 1]");
    println!("     - Good baseline to try first");
    println!("     - Less aggressive than exponential or power law\n");

    println!("  4. Rank-Based Scaling:");
    println!("     - Use when fitness distribution is highly irregular");
    println!("     - Converts values to ranks (1, 2, 3, ...)");
    println!("     - Most robust but loses magnitude information\n");

    println!("Usage in your code:");
    println!("  use particle_swarm_opt::fitness_scaling::{{FitnessScaler, ScalingStrategy}};");
    println!();
    println!("  let hyperparams = Hyperparameters::with_bounds(...)");
    println!("      .with_fitness_scaling(");
    println!("          FitnessScaler::new(ScalingStrategy::Exponential {{ beta: 0.01 }})");
    println!("      );");
    println!();
    println!("======================================================================");
}
