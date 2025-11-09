//! Particle Swarm Optimization Library
//!
//! This library provides a type-safe implementation of Particle Swarm Optimization (PSO)
//! for optimizing COSM parameters using Rust's type system.

use rand::Rng;

/// Hyperparameters for the Particle Swarm Optimization algorithm
#[derive(Debug, Clone)]
pub struct Hyperparameters {
    /// Number of particles in the swarm
    pub swarm_size: usize,
    /// Inertia weight - controls the influence of previous velocity
    pub inertia_weight: f64,
    /// Cognitive coefficient - controls attraction to particle's best position
    pub cognitive_coeff: f64,
    /// Social coefficient - controls attraction to global best position
    pub social_coeff: f64,
    /// Lower bounds for each dimension
    pub lower_bounds: Vec<f64>,
    /// Upper bounds for each dimension
    pub upper_bounds: Vec<f64>,
}

impl Hyperparameters {
    /// Creates a new Hyperparameters struct with default PSO values
    ///
    /// # Arguments
    ///
    /// * `swarm_size` - Number of particles in the swarm
    /// * `dimensions` - Number of dimensions in the search space
    /// * `lower_bound` - Lower bound for all dimensions
    /// * `upper_bound` - Upper bound for all dimensions
    ///
    /// # Returns
    ///
    /// A Hyperparameters struct with standard PSO coefficients
    pub fn new(swarm_size: usize, dimensions: usize, lower_bound: f64, upper_bound: f64) -> Self {
        Self {
            swarm_size,
            inertia_weight: 0.7,      // Standard value
            cognitive_coeff: 1.5,     // Standard value
            social_coeff: 1.5,        // Standard value
            lower_bounds: vec![lower_bound; dimensions],
            upper_bounds: vec![upper_bound; dimensions],
        }
    }

    /// Creates a new Hyperparameters struct with custom bounds per dimension
    ///
    /// # Arguments
    ///
    /// * `swarm_size` - Number of particles in the swarm
    /// * `lower_bounds` - Lower bounds for each dimension
    /// * `upper_bounds` - Upper bounds for each dimension
    ///
    /// # Returns
    ///
    /// A Hyperparameters struct with standard PSO coefficients and custom bounds
    pub fn with_bounds(swarm_size: usize, lower_bounds: Vec<f64>, upper_bounds: Vec<f64>) -> Self {
        assert_eq!(
            lower_bounds.len(),
            upper_bounds.len(),
            "Lower and upper bounds must have the same length"
        );
        Self {
            swarm_size,
            inertia_weight: 0.7,
            cognitive_coeff: 1.5,
            social_coeff: 1.5,
            lower_bounds,
            upper_bounds,
        }
    }

    /// Returns the number of dimensions in the search space
    pub fn dimensions(&self) -> usize {
        self.lower_bounds.len()
    }
}

/// Represents a single particle in the swarm
///
/// Each particle represents a potential solution (a set of COSM parameters)
/// and maintains its current state as well as its historical best position.
#[derive(Debug, Clone)]
pub struct Particle {
    /// Current position in the search space
    pub position: Vec<f64>,
    /// Current velocity vector
    pub velocity: Vec<f64>,
    /// Fitness value at current position
    pub fitness: f64,
    /// Best position this particle has found
    pub best_position: Vec<f64>,
    /// Fitness value at the particle's best position
    pub best_fitness: f64,
}

impl Particle {
    /// Creates a new particle with random position within bounds
    ///
    /// # Arguments
    ///
    /// * `hyperparams` - Reference to hyperparameters containing bounds
    /// * `rng` - Random number generator
    ///
    /// # Returns
    ///
    /// A new Particle with random position and zero velocity
    fn new<R: Rng>(hyperparams: &Hyperparameters, rng: &mut R) -> Self {
        let dimensions = hyperparams.dimensions();

        // Initialize position randomly within bounds
        let position: Vec<f64> = (0..dimensions)
            .map(|i| {
                rng.gen_range(hyperparams.lower_bounds[i]..=hyperparams.upper_bounds[i])
            })
            .collect();

        // Initialize velocity to zero (safe and common practice)
        let velocity = vec![0.0; dimensions];

        // Initialize fitness to worst possible value (will be updated)
        let fitness = f64::INFINITY;

        // Clone position for best_position (will be updated after first evaluation)
        let best_position = position.clone();
        let best_fitness = f64::INFINITY;

        Self {
            position,
            velocity,
            fitness,
            best_position,
            best_fitness,
        }
    }

    /// Updates the particle's personal best if current position is better
    fn update_personal_best(&mut self) {
        if self.fitness < self.best_fitness {
            self.best_fitness = self.fitness;
            self.best_position = self.position.clone();
        }
    }
}

/// Manages the collection of particles and orchestrates the optimization process
#[derive(Debug)]
pub struct Swarm {
    /// Collection of all particles in the swarm
    pub particles: Vec<Particle>,
    /// Best position found by any particle in the swarm
    pub global_best_position: Vec<f64>,
    /// Fitness value at the global best position
    pub global_best_fitness: f64,
    /// Hyperparameters controlling the optimization
    pub hyperparameters: Hyperparameters,
}

impl Swarm {
    /// Creates and initializes a new swarm
    ///
    /// This method:
    /// 1. Creates the specified number of particles
    /// 2. Assigns random starting positions within defined boundaries
    /// 3. Initializes velocities to zero
    /// 4. Evaluates initial fitness for all particles
    /// 5. Sets personal bests for each particle
    /// 6. Determines the initial global best position
    ///
    /// # Arguments
    ///
    /// * `hyperparams` - Hyperparameters for the PSO algorithm
    /// * `fitness_fn` - Function to evaluate the fitness of a position
    ///
    /// # Returns
    ///
    /// An initialized Swarm ready for optimization
    ///
    /// # Example
    ///
    /// ```
    /// use particle_swarm_opt::{Swarm, Hyperparameters};
    ///
    /// // Define a simple fitness function (sphere function)
    /// let fitness_fn = |position: &[f64]| {
    ///     position.iter().map(|x| x * x).sum()
    /// };
    ///
    /// // Create hyperparameters for 2D optimization
    /// let hyperparams = Hyperparameters::new(30, 2, -10.0, 10.0);
    ///
    /// // Initialize the swarm
    /// let swarm = Swarm::new(hyperparams, fitness_fn);
    /// ```
    pub fn new<F>(hyperparams: Hyperparameters, fitness_fn: F) -> Self
    where
        F: Fn(&[f64]) -> f64,
    {
        let mut rng = rand::thread_rng();

        // Create particles with random positions
        let mut particles: Vec<Particle> = (0..hyperparams.swarm_size)
            .map(|_| Particle::new(&hyperparams, &mut rng))
            .collect();

        // Evaluate initial fitness for all particles
        for particle in &mut particles {
            particle.fitness = fitness_fn(&particle.position);
            particle.best_fitness = particle.fitness;
            particle.best_position = particle.position.clone();
        }

        // Find the global best position
        let mut global_best_fitness = f64::INFINITY;
        let mut global_best_position = vec![0.0; hyperparams.dimensions()];

        for particle in &particles {
            if particle.fitness < global_best_fitness {
                global_best_fitness = particle.fitness;
                global_best_position = particle.position.clone();
            }
        }

        Self {
            particles,
            global_best_position,
            global_best_fitness,
            hyperparameters: hyperparams,
        }
    }

    /// Returns the number of dimensions in the search space
    pub fn dimensions(&self) -> usize {
        self.hyperparameters.dimensions()
    }

    /// Returns the number of particles in the swarm
    pub fn size(&self) -> usize {
        self.particles.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hyperparameters_new() {
        let hyperparams = Hyperparameters::new(30, 5, -10.0, 10.0);

        assert_eq!(hyperparams.swarm_size, 30);
        assert_eq!(hyperparams.dimensions(), 5);
        assert_eq!(hyperparams.lower_bounds, vec![-10.0; 5]);
        assert_eq!(hyperparams.upper_bounds, vec![10.0; 5]);
        assert_eq!(hyperparams.inertia_weight, 0.7);
        assert_eq!(hyperparams.cognitive_coeff, 1.5);
        assert_eq!(hyperparams.social_coeff, 1.5);
    }

    #[test]
    fn test_hyperparameters_with_bounds() {
        let lower = vec![-5.0, -10.0, -15.0];
        let upper = vec![5.0, 10.0, 15.0];
        let hyperparams = Hyperparameters::with_bounds(20, lower.clone(), upper.clone());

        assert_eq!(hyperparams.swarm_size, 20);
        assert_eq!(hyperparams.dimensions(), 3);
        assert_eq!(hyperparams.lower_bounds, lower);
        assert_eq!(hyperparams.upper_bounds, upper);
    }

    #[test]
    #[should_panic(expected = "Lower and upper bounds must have the same length")]
    fn test_hyperparameters_mismatched_bounds() {
        let lower = vec![-5.0, -10.0];
        let upper = vec![5.0, 10.0, 15.0];
        let _hyperparams = Hyperparameters::with_bounds(20, lower, upper);
    }

    #[test]
    fn test_particle_initialization() {
        let hyperparams = Hyperparameters::new(10, 3, -5.0, 5.0);
        let mut rng = rand::thread_rng();
        let particle = Particle::new(&hyperparams, &mut rng);

        assert_eq!(particle.position.len(), 3);
        assert_eq!(particle.velocity.len(), 3);
        assert_eq!(particle.best_position.len(), 3);

        // Check that position is within bounds
        for i in 0..3 {
            assert!(particle.position[i] >= -5.0);
            assert!(particle.position[i] <= 5.0);
        }

        // Check that velocity is initialized to zero
        for vel in &particle.velocity {
            assert_eq!(*vel, 0.0);
        }
    }

    #[test]
    fn test_particle_update_personal_best() {
        let hyperparams = Hyperparameters::new(10, 2, -5.0, 5.0);
        let mut rng = rand::thread_rng();
        let mut particle = Particle::new(&hyperparams, &mut rng);

        // Set initial fitness
        particle.fitness = 10.0;
        particle.best_fitness = 10.0;

        // Improve fitness
        particle.fitness = 5.0;
        particle.position = vec![1.0, 2.0];
        particle.update_personal_best();

        assert_eq!(particle.best_fitness, 5.0);
        assert_eq!(particle.best_position, vec![1.0, 2.0]);

        // Worse fitness should not update
        particle.fitness = 8.0;
        particle.position = vec![3.0, 4.0];
        particle.update_personal_best();

        assert_eq!(particle.best_fitness, 5.0);
        assert_eq!(particle.best_position, vec![1.0, 2.0]);
    }

    #[test]
    fn test_swarm_initialization() {
        // Simple sphere function: sum of squares
        let fitness_fn = |position: &[f64]| {
            position.iter().map(|x| x * x).sum()
        };

        let hyperparams = Hyperparameters::new(20, 3, -10.0, 10.0);
        let swarm = Swarm::new(hyperparams, fitness_fn);

        assert_eq!(swarm.size(), 20);
        assert_eq!(swarm.dimensions(), 3);

        // Check that all particles are initialized
        for particle in &swarm.particles {
            assert_eq!(particle.position.len(), 3);
            assert_eq!(particle.velocity.len(), 3);
            assert!(particle.fitness.is_finite());
            assert_eq!(particle.fitness, particle.best_fitness);
        }

        // Check that global best is set to the best particle
        assert!(swarm.global_best_fitness.is_finite());
        assert_eq!(swarm.global_best_position.len(), 3);

        // Verify global best is actually the minimum
        let min_fitness = swarm.particles.iter()
            .map(|p| p.fitness)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        assert_eq!(swarm.global_best_fitness, min_fitness);
    }

    #[test]
    fn test_swarm_with_custom_bounds() {
        let fitness_fn = |position: &[f64]| {
            position.iter().map(|x| x * x).sum()
        };

        let lower = vec![-1.0, -2.0, -3.0];
        let upper = vec![1.0, 2.0, 3.0];
        let hyperparams = Hyperparameters::with_bounds(15, lower, upper);
        let swarm = Swarm::new(hyperparams, fitness_fn);

        assert_eq!(swarm.size(), 15);
        assert_eq!(swarm.dimensions(), 3);

        // Verify all particles are within custom bounds
        for particle in &swarm.particles {
            assert!(particle.position[0] >= -1.0 && particle.position[0] <= 1.0);
            assert!(particle.position[1] >= -2.0 && particle.position[1] <= 2.0);
            assert!(particle.position[2] >= -3.0 && particle.position[2] <= 3.0);
        }
    }
}
