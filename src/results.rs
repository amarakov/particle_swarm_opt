//! Results and Reporting Module
//!
//! This module provides functionality for saving, loading, and displaying
//! PSO optimization results in various formats (JSON, CSV, console output).

use crate::{OptimizationHistory, Swarm};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io;
use std::path::Path;

/// Summary of optimization results
///
/// Contains the final state of the optimization and key metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResults {
    /// Final global best position found
    pub best_position: Vec<f64>,
    /// Final global best fitness value
    pub best_fitness: f64,
    /// Number of iterations performed
    pub iterations: usize,
    /// Number of particles used
    pub swarm_size: usize,
    /// Number of dimensions in the search space
    pub dimensions: usize,
    /// Initial fitness (before optimization)
    pub initial_fitness: f64,
    /// Improvement percentage
    pub improvement_percent: f64,
    /// Parameter names (optional)
    pub parameter_names: Option<Vec<String>>,
}

impl OptimizationResults {
    /// Creates optimization results from a swarm and history
    ///
    /// # Arguments
    ///
    /// * `swarm` - The swarm after optimization
    /// * `history` - The optimization history
    /// * `parameter_names` - Optional parameter names
    pub fn from_swarm(
        swarm: &Swarm,
        history: &OptimizationHistory,
        parameter_names: Option<Vec<String>>,
    ) -> Self {
        let initial_fitness = if !history.is_empty() {
            history.iterations[0].global_best_fitness
        } else {
            swarm.global_best_fitness
        };

        let improvement = if initial_fitness != 0.0 {
            ((initial_fitness - swarm.global_best_fitness) / initial_fitness.abs()) * 100.0
        } else {
            0.0
        };

        Self {
            best_position: swarm.global_best_position.clone(),
            best_fitness: swarm.global_best_fitness,
            iterations: history.len(),
            swarm_size: swarm.size(),
            dimensions: swarm.dimensions(),
            initial_fitness,
            improvement_percent: improvement,
            parameter_names,
        }
    }

    /// Saves the results to a JSON file
    ///
    /// # Arguments
    ///
    /// * `path` - Path to save the JSON file
    ///
    /// # Returns
    ///
    /// Result indicating success or failure
    pub fn save_json<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let file = File::create(path)?;
        serde_json::to_writer_pretty(file, self)?;
        Ok(())
    }

    /// Loads results from a JSON file
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the JSON file
    ///
    /// # Returns
    ///
    /// The loaded OptimizationResults or an error
    pub fn load_json<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::open(path)?;
        let results = serde_json::from_reader(file)?;
        Ok(results)
    }

    /// Prints a formatted summary to the console
    pub fn print_summary(&self) {
        println!("\n{}", "=".repeat(70));
        println!("               OPTIMIZATION RESULTS SUMMARY");
        println!("{}", "=".repeat(70));

        println!("\nConfiguration:");
        println!("  Swarm size:      {}", self.swarm_size);
        println!("  Dimensions:      {}", self.dimensions);
        println!("  Iterations:      {}", self.iterations);

        println!("\nPerformance:");
        println!("  Initial fitness: {:.10}", self.initial_fitness);
        println!("  Final fitness:   {:.10}", self.best_fitness);
        println!("  Improvement:     {:.2}%", self.improvement_percent);

        println!("\nBest Solution:");
        if let Some(ref names) = self.parameter_names {
            for (name, value) in names.iter().zip(self.best_position.iter()) {
                println!("  {:<20} = {:.10}", name, value);
            }
        } else {
            for (i, value) in self.best_position.iter().enumerate() {
                println!("  Parameter {:2}:      {:.10}", i, value);
            }
        }

        println!("\n{}", "=".repeat(70));
    }

    /// Prints a compact one-line summary
    pub fn print_compact(&self) {
        println!(
            "Final: fitness={:.6}, improvement={:.2}%, iterations={}",
            self.best_fitness, self.improvement_percent, self.iterations
        );
    }
}

/// Saves optimization history to a JSON file
///
/// # Arguments
///
/// * `history` - The optimization history to save
/// * `path` - Path to save the JSON file
///
/// # Returns
///
/// Result indicating success or failure
pub fn save_history_json<P: AsRef<Path>>(
    history: &OptimizationHistory,
    path: P,
) -> io::Result<()> {
    let file = File::create(path)?;
    serde_json::to_writer_pretty(file, history)?;
    Ok(())
}

/// Loads optimization history from a JSON file
///
/// # Arguments
///
/// * `path` - Path to the JSON file
///
/// # Returns
///
/// The loaded OptimizationHistory or an error
pub fn load_history_json<P: AsRef<Path>>(path: P) -> io::Result<OptimizationHistory> {
    let file = File::open(path)?;
    let history = serde_json::from_reader(file)?;
    Ok(history)
}

/// Saves convergence data to a CSV file
///
/// Creates a CSV with columns: iteration, global_best_fitness
///
/// # Arguments
///
/// * `history` - The optimization history
/// * `path` - Path to save the CSV file
///
/// # Returns
///
/// Result indicating success or failure
pub fn save_convergence_csv<P: AsRef<Path>>(
    history: &OptimizationHistory,
    path: P,
) -> io::Result<()> {
    let mut writer = csv::Writer::from_path(path)?;

    // Write header
    writer.write_record(&["iteration", "global_best_fitness"])?;

    // Write data
    for iter_data in &history.iterations {
        writer.write_record(&[
            iter_data.iteration.to_string(),
            iter_data.global_best_fitness.to_string(),
        ])?;
    }

    writer.flush()?;
    Ok(())
}

/// Saves detailed particle data to a CSV file
///
/// Creates a CSV with particle positions and fitness values for a specific iteration
///
/// # Arguments
///
/// * `history` - The optimization history
/// * `iteration` - Which iteration to export (0-based)
/// * `path` - Path to save the CSV file
///
/// # Returns
///
/// Result indicating success or failure
pub fn save_particles_csv<P: AsRef<Path>>(
    history: &OptimizationHistory,
    iteration: usize,
    path: P,
) -> io::Result<()> {
    if iteration >= history.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "Iteration {} out of bounds (max: {})",
                iteration,
                history.len() - 1
            ),
        ));
    }

    let iter_data = &history.iterations[iteration];
    let dimensions = iter_data.particle_positions[0].len();

    let mut writer = csv::Writer::from_path(path)?;

    // Write header
    let mut header = vec!["particle_id".to_string()];
    for i in 0..dimensions {
        header.push(format!("dim_{}", i));
    }
    header.push("fitness".to_string());
    writer.write_record(&header)?;

    // Write data
    for (particle_id, (position, fitness)) in iter_data
        .particle_positions
        .iter()
        .zip(iter_data.particle_fitnesses.iter())
        .enumerate()
    {
        let mut record = vec![particle_id.to_string()];
        for value in position {
            record.push(value.to_string());
        }
        record.push(fitness.to_string());
        writer.write_record(&record)?;
    }

    writer.flush()?;
    Ok(())
}

/// Prints convergence progress to the console
///
/// # Arguments
///
/// * `history` - The optimization history
/// * `step` - Show every nth iteration (e.g., step=10 shows iterations 0, 10, 20, ...)
pub fn print_convergence(history: &OptimizationHistory, step: usize) {
    println!("\nConvergence Progress:");
    println!("{:-<50}", "");
    println!("{:>10} | {:>15} | {:>15}", "Iteration", "Best Fitness", "Improvement");
    println!("{:-<50}", "");

    let mut prev_fitness = f64::INFINITY;

    for (i, iter_data) in history.iterations.iter().enumerate() {
        if i % step == 0 || i == history.len() - 1 {
            let improvement = if prev_fitness.is_finite() && prev_fitness != 0.0 {
                ((prev_fitness - iter_data.global_best_fitness) / prev_fitness.abs()) * 100.0
            } else {
                0.0
            };

            println!(
                "{:>10} | {:>15.10} | {:>14.2}%",
                iter_data.iteration, iter_data.global_best_fitness, improvement
            );

            prev_fitness = iter_data.global_best_fitness;
        }
    }

    println!("{:-<50}", "");
}

/// Complete report that saves all data and prints summaries
///
/// This is a convenience function that:
/// - Saves results to JSON
/// - Saves full history to JSON
/// - Saves convergence data to CSV
/// - Prints console summary
///
/// # Arguments
///
/// * `swarm` - The swarm after optimization
/// * `history` - The optimization history
/// * `output_dir` - Directory to save output files
/// * `prefix` - Prefix for output filenames
/// * `parameter_names` - Optional parameter names
///
/// # Returns
///
/// Result indicating success or failure
pub fn generate_full_report<P: AsRef<Path>>(
    swarm: &Swarm,
    history: &OptimizationHistory,
    output_dir: P,
    prefix: &str,
    parameter_names: Option<Vec<String>>,
) -> io::Result<()> {
    let output_dir = output_dir.as_ref();

    // Create output directory if it doesn't exist
    std::fs::create_dir_all(output_dir)?;

    // Create results summary
    let results = OptimizationResults::from_swarm(swarm, history, parameter_names);

    // Save results to JSON
    let results_path = output_dir.join(format!("{}_results.json", prefix));
    results.save_json(&results_path)?;
    println!("\n✓ Results saved to: {}", results_path.display());

    // Save full history to JSON
    let history_path = output_dir.join(format!("{}_history.json", prefix));
    save_history_json(history, &history_path)?;
    println!("✓ History saved to: {}", history_path.display());

    // Save convergence data to CSV
    let convergence_path = output_dir.join(format!("{}_convergence.csv", prefix));
    save_convergence_csv(history, &convergence_path)?;
    println!("✓ Convergence data saved to: {}", convergence_path.display());

    // Print summary to console
    results.print_summary();

    // Print convergence progress
    let step = if history.len() > 20 {
        history.len() / 20
    } else {
        1
    };
    print_convergence(history, step);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Hyperparameters, Swarm};
    use std::fs;

    #[test]
    fn test_optimization_results_creation() {
        let fitness_fn = |position: &[f64]| -> f64 { position.iter().map(|x| x * x).sum() };

        let hyperparams = Hyperparameters::new(10, 2, -5.0, 5.0);
        let mut swarm = Swarm::new(hyperparams, &fitness_fn);
        let history = swarm.optimize(&fitness_fn, 10);

        let results = OptimizationResults::from_swarm(
            &swarm,
            &history,
            Some(vec!["x".to_string(), "y".to_string()]),
        );

        assert_eq!(results.dimensions, 2);
        assert_eq!(results.swarm_size, 10);
        assert_eq!(results.iterations, 10);
        assert!(results.parameter_names.is_some());
    }

    #[test]
    fn test_save_and_load_results_json() {
        let fitness_fn = |position: &[f64]| -> f64 { position.iter().map(|x| x * x).sum() };

        let hyperparams = Hyperparameters::new(10, 2, -5.0, 5.0);
        let mut swarm = Swarm::new(hyperparams, &fitness_fn);
        let history = swarm.optimize(&fitness_fn, 5);

        let results = OptimizationResults::from_swarm(&swarm, &history, None);

        let temp_path = "/tmp/test_results.json";
        results.save_json(temp_path).unwrap();

        let loaded = OptimizationResults::load_json(temp_path).unwrap();

        assert_eq!(results.best_fitness, loaded.best_fitness);
        assert_eq!(results.dimensions, loaded.dimensions);

        // Cleanup
        let _ = fs::remove_file(temp_path);
    }

    #[test]
    fn test_save_and_load_history_json() {
        let fitness_fn = |position: &[f64]| -> f64 { position.iter().map(|x| x * x).sum() };

        let hyperparams = Hyperparameters::new(10, 2, -5.0, 5.0);
        let mut swarm = Swarm::new(hyperparams, &fitness_fn);
        let history = swarm.optimize(&fitness_fn, 5);

        let temp_path = "/tmp/test_history.json";
        save_history_json(&history, temp_path).unwrap();

        let loaded = load_history_json(temp_path).unwrap();

        assert_eq!(history.len(), loaded.len());

        // Cleanup
        let _ = fs::remove_file(temp_path);
    }

    #[test]
    fn test_save_convergence_csv() {
        let fitness_fn = |position: &[f64]| -> f64 { position.iter().map(|x| x * x).sum() };

        let hyperparams = Hyperparameters::new(10, 2, -5.0, 5.0);
        let mut swarm = Swarm::new(hyperparams, &fitness_fn);
        let history = swarm.optimize(&fitness_fn, 5);

        let temp_path = "/tmp/test_convergence.csv";
        save_convergence_csv(&history, temp_path).unwrap();

        // Verify file exists and has content
        let content = fs::read_to_string(temp_path).unwrap();
        assert!(content.contains("iteration"));
        assert!(content.contains("global_best_fitness"));

        // Cleanup
        let _ = fs::remove_file(temp_path);
    }

    #[test]
    fn test_save_particles_csv() {
        let fitness_fn = |position: &[f64]| -> f64 { position.iter().map(|x| x * x).sum() };

        let hyperparams = Hyperparameters::new(5, 2, -5.0, 5.0);
        let mut swarm = Swarm::new(hyperparams, &fitness_fn);
        let history = swarm.optimize(&fitness_fn, 3);

        let temp_path = "/tmp/test_particles.csv";
        save_particles_csv(&history, 0, temp_path).unwrap();

        // Verify file exists and has content
        let content = fs::read_to_string(temp_path).unwrap();
        assert!(content.contains("particle_id"));
        assert!(content.contains("dim_0"));
        assert!(content.contains("fitness"));

        // Cleanup
        let _ = fs::remove_file(temp_path);
    }

    #[test]
    fn test_save_particles_csv_invalid_iteration() {
        let fitness_fn = |position: &[f64]| -> f64 { position.iter().map(|x| x * x).sum() };

        let hyperparams = Hyperparameters::new(5, 2, -5.0, 5.0);
        let mut swarm = Swarm::new(hyperparams, &fitness_fn);
        let history = swarm.optimize(&fitness_fn, 3);

        let temp_path = "/tmp/test_particles_invalid.csv";
        let result = save_particles_csv(&history, 10, temp_path);

        assert!(result.is_err());
    }
}
