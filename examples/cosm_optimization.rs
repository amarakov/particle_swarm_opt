//! COSM Parameter Optimization Example
//!
//! This example demonstrates how to use the PSO library to optimize
//! COSM (Cosmological Simulation) parameters.
//!
//! Run this example with:
//! ```
//! cargo run --example cosm_optimization
//! ```

use particle_swarm_opt::{
    cosm::{CosmConfig, create_cosm_objective},
    results::generate_full_report,
    Hyperparameters, Swarm,
};

fn main() {
    println!("\n{}", "=".repeat(70));
    println!("           COSM PARAMETER OPTIMIZATION WITH PSO");
    println!("{}", "=".repeat(70));

    // Define the COSM parameters we want to optimize
    // These are example cosmological parameters
    let parameter_names = vec![
        "omega_matter".to_string(),    // Matter density parameter
        "omega_lambda".to_string(),    // Dark energy density parameter
        "hubble_constant".to_string(), // Hubble constant (reduced)
        "sigma_8".to_string(),         // Amplitude of matter fluctuations
        "n_s".to_string(),             // Spectral index
    ];

    // Define physically reasonable bounds for each parameter
    let lower_bounds = vec![
        0.20,  // omega_matter: 20% to 40%
        0.60,  // omega_lambda: 60% to 80%
        0.60,  // hubble_constant: 0.6 to 0.8 (h = H0/100)
        0.70,  // sigma_8: 0.7 to 0.9
        0.90,  // n_s: 0.9 to 1.1
    ];

    let upper_bounds = vec![
        0.40,  // omega_matter
        0.80,  // omega_lambda
        0.80,  // hubble_constant
        0.90,  // sigma_8
        1.10,  // n_s
    ];

    // Known baseline values (e.g., from Planck 2018)
    let baseline = vec![
        0.315, // omega_matter
        0.685, // omega_lambda
        0.674, // hubble_constant (h)
        0.811, // sigma_8
        0.965, // n_s
    ];

    // Create COSM configuration
    let cosm_config = CosmConfig::new(
        parameter_names.clone(),
        lower_bounds.clone(),
        upper_bounds.clone(),
    )
    .with_baseline(baseline.clone());

    println!("\nCOSM Configuration:");
    println!("  Parameters to optimize: {}", cosm_config.num_parameters());
    for (i, name) in parameter_names.iter().enumerate() {
        println!(
            "    {:<20} [{:.4}, {:.4}] (baseline: {:.4})",
            name, lower_bounds[i], upper_bounds[i], baseline[i]
        );
    }
    println!();

    // Create the objective function with validation
    let objective_fn = create_cosm_objective(&cosm_config);

    // Configure PSO hyperparameters
    let swarm_size = 40;
    let max_iterations = 100;

    let hyperparams = Hyperparameters::with_bounds(
        swarm_size,
        lower_bounds.clone(),
        upper_bounds.clone(),
    );

    println!("PSO Configuration:");
    println!("  Swarm size:           {}", hyperparams.swarm_size);
    println!("  Dimensions:           {}", hyperparams.dimensions());
    println!("  Max iterations:       {}", max_iterations);
    println!("  Inertia weight:       {}", hyperparams.inertia_weight);
    println!("  Cognitive coeff:      {}", hyperparams.cognitive_coeff);
    println!("  Social coeff:         {}", hyperparams.social_coeff);
    println!();

    // Initialize the swarm
    println!("Initializing swarm with {} particles...", swarm_size);
    let mut swarm = Swarm::new(hyperparams, &objective_fn);

    println!("Initial state:");
    println!("  Global best fitness:  {:.10}", swarm.global_best_fitness);
    println!("  Global best position:");
    for (name, value) in parameter_names
        .iter()
        .zip(swarm.global_best_position.iter())
    {
        println!("    {:<20} = {:.6}", name, value);
    }
    println!();

    // Run optimization
    println!("Starting optimization...");
    println!("{}", "-".repeat(70));

    let history = swarm.optimize(&objective_fn, max_iterations);

    println!("{}", "-".repeat(70));
    println!("Optimization complete!");
    println!();

    // Display final results
    println!("Final Results:");
    println!("  Iterations performed: {}", history.len());
    println!("  Final best fitness:   {:.10}", swarm.global_best_fitness);
    println!();

    println!("Optimized COSM Parameters:");
    for (name, value) in parameter_names.iter().zip(swarm.global_best_position.iter()) {
        println!("  {:<20} = {:.6}", name, value);
    }
    println!();

    // Compare with baseline
    println!("Comparison with Baseline:");
    for (name, (optimized, base)) in parameter_names
        .iter()
        .zip(swarm.global_best_position.iter().zip(baseline.iter()))
    {
        let diff = optimized - base;
        let pct = (diff / base) * 100.0;
        println!(
            "  {:<20} optimized={:.6}, baseline={:.6}, diff={:+.6} ({:+.2}%)",
            name, optimized, base, diff, pct
        );
    }
    println!();

    // Show convergence progress (sample every few iterations)
    println!("Convergence Progress (sample):");
    let step = if history.len() > 10 { history.len() / 10 } else { 1 };
    for (i, iter_data) in history.iterations.iter().enumerate() {
        if i % step == 0 || i == history.len() - 1 {
            println!(
                "  Iteration {:3}: fitness = {:.10}",
                iter_data.iteration, iter_data.global_best_fitness
            );
        }
    }
    println!();

    // Calculate improvement
    let initial_fitness = history.iterations[0].global_best_fitness;
    let final_fitness = swarm.global_best_fitness;
    let improvement = if initial_fitness != 0.0 {
        ((initial_fitness - final_fitness) / initial_fitness) * 100.0
    } else {
        0.0
    };

    println!("Performance Summary:");
    println!("  Initial fitness:      {:.10}", initial_fitness);
    println!("  Final fitness:        {:.10}", final_fitness);
    println!("  Improvement:          {:.2}%", improvement);
    println!();

    // Save results to files
    println!("Saving results...");
    let output_dir = "output";
    match generate_full_report(
        &swarm,
        &history,
        output_dir,
        "cosm_optimization",
        Some(parameter_names),
    ) {
        Ok(_) => println!("✓ All results saved successfully!"),
        Err(e) => eprintln!("✗ Error saving results: {}", e),
    }

    println!();
    println!("{}", "=".repeat(70));
    println!("               OPTIMIZATION COMPLETE");
    println!("{}", "=".repeat(70));
    println!();
    println!("NOTE: The objective function currently uses a placeholder (Rosenbrock).");
    println!("      Replace 'cosm_objective_function' in src/cosm.rs with your actual");
    println!("      COSM benchmark to optimize real cosmological parameters.");
    println!();
}
