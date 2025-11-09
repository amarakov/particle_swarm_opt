//! COSM (Cosmological Simulation) Parameter Optimization Module
//!
//! This module provides the interface between the PSO engine and the COSM
//! parameter optimization task. It defines the objective function that
//! evaluates how well a given set of COSM parameters performs.

use serde::{Deserialize, Serialize};

/// Configuration for COSM parameter optimization
///
/// Defines which parameters to optimize and their valid ranges
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CosmConfig {
    /// Names of the parameters being optimized
    pub parameter_names: Vec<String>,
    /// Lower bounds for each parameter
    pub lower_bounds: Vec<f64>,
    /// Upper bounds for each parameter
    pub upper_bounds: Vec<f64>,
    /// Optional: baseline/default values for reference
    pub baseline_values: Option<Vec<f64>>,
}

impl CosmConfig {
    /// Creates a new COSM configuration
    ///
    /// # Arguments
    ///
    /// * `parameter_names` - Names of the parameters to optimize
    /// * `lower_bounds` - Lower bounds for each parameter
    /// * `upper_bounds` - Upper bounds for each parameter
    ///
    /// # Panics
    ///
    /// Panics if the lengths of parameter_names, lower_bounds, and upper_bounds don't match
    pub fn new(
        parameter_names: Vec<String>,
        lower_bounds: Vec<f64>,
        upper_bounds: Vec<f64>,
    ) -> Self {
        assert_eq!(
            parameter_names.len(),
            lower_bounds.len(),
            "Parameter names and lower bounds must have the same length"
        );
        assert_eq!(
            parameter_names.len(),
            upper_bounds.len(),
            "Parameter names and upper bounds must have the same length"
        );

        Self {
            parameter_names,
            lower_bounds,
            upper_bounds,
            baseline_values: None,
        }
    }

    /// Sets baseline values for comparison
    pub fn with_baseline(mut self, baseline: Vec<f64>) -> Self {
        assert_eq!(
            baseline.len(),
            self.parameter_names.len(),
            "Baseline values must match the number of parameters"
        );
        self.baseline_values = Some(baseline);
        self
    }

    /// Returns the number of parameters being optimized
    pub fn num_parameters(&self) -> usize {
        self.parameter_names.len()
    }
}

/// Evaluates the fitness of a set of COSM parameters
///
/// This is the objective function that will be passed to the PSO optimizer.
/// It accepts a slice of f64 values representing the COSM parameters and
/// returns a single f64 fitness value (lower is better).
///
/// # Arguments
///
/// * `parameters` - Slice of parameter values to evaluate
///
/// # Returns
///
/// A fitness score where lower values indicate better performance
///
/// # Note
///
/// This is a placeholder function. The actual COSM benchmark implementation
/// should be inserted here by the user. The function signature must remain
/// the same: `&[f64] -> f64`
///
/// # Example
///
/// ```
/// use particle_swarm_opt::cosm::cosm_objective_function;
///
/// // Evaluate a set of COSM parameters
/// let params = vec![0.3, 0.7, 0.05, 68.0, 0.96];
/// let fitness = cosm_objective_function(&params);
/// println!("Fitness: {}", fitness);
/// ```
pub fn cosm_objective_function(parameters: &[f64]) -> f64 {
    // PLACEHOLDER: Insert actual COSM benchmark call here
    //
    // This function should:
    // 1. Take the parameters slice
    // 2. Run the COSM simulation/benchmark with these parameters
    // 3. Return a fitness value (e.g., error metric, chi-squared, likelihood, etc.)
    //
    // Example structure:
    // let result = run_cosm_simulation(parameters);
    // let fitness = calculate_error_metric(result);
    // fitness
    //
    // For now, we return a simple test function (Rosenbrock function)
    // This should be replaced with actual COSM evaluation

    if parameters.is_empty() {
        return f64::INFINITY;
    }

    // Rosenbrock function as placeholder (minimum at all 1's)
    let mut sum = 0.0;
    for i in 0..parameters.len() - 1 {
        let x = parameters[i];
        let x_next = parameters[i + 1];
        sum += 100.0 * (x_next - x * x).powi(2) + (1.0 - x).powi(2);
    }

    sum
}

/// Creates a COSM objective function closure that captures additional context
///
/// This function allows you to create a closure that includes additional
/// configuration or state needed for the COSM evaluation.
///
/// # Arguments
///
/// * `config` - COSM configuration
///
/// # Returns
///
/// A closure that can be passed to the PSO optimizer
///
/// # Example
///
/// ```
/// use particle_swarm_opt::cosm::{CosmConfig, create_cosm_objective};
///
/// let config = CosmConfig::new(
///     vec!["omega_m".to_string(), "omega_lambda".to_string()],
///     vec![0.0, 0.0],
///     vec![1.0, 1.0],
/// );
///
/// let objective = create_cosm_objective(&config);
/// let fitness = objective(&[0.3, 0.7]);
/// ```
pub fn create_cosm_objective<'a>(
    config: &'a CosmConfig,
) -> impl Fn(&[f64]) -> f64 + Sync + 'a {
    move |parameters: &[f64]| -> f64 {
        // Validate parameter count
        if parameters.len() != config.num_parameters() {
            eprintln!(
                "Warning: Expected {} parameters, got {}",
                config.num_parameters(),
                parameters.len()
            );
            return f64::INFINITY;
        }

        // Validate parameter bounds (extra safety check)
        for (i, &param) in parameters.iter().enumerate() {
            if param < config.lower_bounds[i] || param > config.upper_bounds[i] {
                eprintln!(
                    "Warning: Parameter {} ({}) out of bounds [{}, {}]",
                    config.parameter_names[i],
                    param,
                    config.lower_bounds[i],
                    config.upper_bounds[i]
                );
                return f64::INFINITY;
            }
        }

        // Call the actual COSM objective function
        cosm_objective_function(parameters)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosm_config_creation() {
        let config = CosmConfig::new(
            vec!["omega_m".to_string(), "omega_lambda".to_string(), "h".to_string()],
            vec![0.0, 0.0, 0.5],
            vec![1.0, 1.0, 1.0],
        );

        assert_eq!(config.num_parameters(), 3);
        assert_eq!(config.parameter_names[0], "omega_m");
        assert!(config.baseline_values.is_none());
    }

    #[test]
    fn test_cosm_config_with_baseline() {
        let config = CosmConfig::new(
            vec!["param1".to_string(), "param2".to_string()],
            vec![0.0, 0.0],
            vec![10.0, 10.0],
        )
        .with_baseline(vec![5.0, 5.0]);

        assert!(config.baseline_values.is_some());
        assert_eq!(config.baseline_values.unwrap(), vec![5.0, 5.0]);
    }

    #[test]
    #[should_panic(expected = "Parameter names and lower bounds must have the same length")]
    fn test_cosm_config_mismatched_lengths() {
        let _config = CosmConfig::new(
            vec!["param1".to_string()],
            vec![0.0, 0.0],
            vec![10.0],
        );
    }

    #[test]
    fn test_cosm_objective_function_placeholder() {
        // Test that the placeholder function works
        let params = vec![1.0, 1.0, 1.0];
        let fitness = cosm_objective_function(&params);

        // For Rosenbrock function, minimum is at all 1's
        assert!(fitness >= 0.0);
        assert!(fitness < 0.001); // Should be very close to 0 at [1,1,1]
    }

    #[test]
    fn test_cosm_objective_function_empty() {
        let params = vec![];
        let fitness = cosm_objective_function(&params);
        assert!(fitness.is_infinite());
    }

    #[test]
    fn test_create_cosm_objective() {
        let config = CosmConfig::new(
            vec!["param1".to_string(), "param2".to_string()],
            vec![-5.0, -5.0],
            vec![5.0, 5.0],
        );

        let objective = create_cosm_objective(&config);
        let fitness = objective(&[1.0, 1.0]);

        assert!(fitness.is_finite());
        assert!(fitness >= 0.0);
    }

    #[test]
    fn test_create_cosm_objective_wrong_dimension() {
        let config = CosmConfig::new(
            vec!["param1".to_string(), "param2".to_string()],
            vec![-5.0, -5.0],
            vec![5.0, 5.0],
        );

        let objective = create_cosm_objective(&config);
        let fitness = objective(&[1.0]); // Wrong number of parameters

        assert!(fitness.is_infinite());
    }

    #[test]
    fn test_create_cosm_objective_out_of_bounds() {
        let config = CosmConfig::new(
            vec!["param1".to_string(), "param2".to_string()],
            vec![-5.0, -5.0],
            vec![5.0, 5.0],
        );

        let objective = create_cosm_objective(&config);
        let fitness = objective(&[10.0, 1.0]); // First parameter out of bounds

        assert!(fitness.is_infinite());
    }
}
