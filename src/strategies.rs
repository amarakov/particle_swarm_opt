//! Strategy traits and implementations for PSO
//!
//! This module provides a pluggable architecture for various PSO strategies:
//! - Initialization strategies (uniform, chaotic)
//! - Boundary handling strategies (reflect, absorb, random)
//! - Topology strategies (global best, local best)
//! - Inertia weight strategies (constant, linearly decreasing)

use rand::RngCore;

/// Trait for particle initialization strategies
///
/// Different initialization strategies can help improve diversity and exploration
pub trait InitializationStrategy: Send + Sync {
    /// Initializes a particle's position within the given bounds
    ///
    /// # Arguments
    ///
    /// * `dimensions` - Number of dimensions
    /// * `lower_bounds` - Lower bounds for each dimension
    /// * `upper_bounds` - Upper bounds for each dimension
    /// * `rng` - Random number generator
    ///
    /// # Returns
    ///
    /// A vector representing the initial position
    fn initialize_position(
        &self,
        dimensions: usize,
        lower_bounds: &[f64],
        upper_bounds: &[f64],
        rng: &mut dyn RngCore,
    ) -> Vec<f64>;

    /// Clone the strategy into a Box
    fn clone_box(&self) -> Box<dyn InitializationStrategy>;
}

impl Clone for Box<dyn InitializationStrategy> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Uniform random initialization (standard approach)
///
/// Particles are initialized with uniform random distribution within bounds
#[derive(Debug, Clone)]
pub struct UniformInitialization;

impl InitializationStrategy for UniformInitialization {
    fn initialize_position(
        &self,
        dimensions: usize,
        lower_bounds: &[f64],
        upper_bounds: &[f64],
        rng: &mut dyn RngCore,
    ) -> Vec<f64> {
        use rand::Rng; // For gen_range method
        (0..dimensions)
            .map(|i| rng.gen_range(lower_bounds[i]..=upper_bounds[i]))
            .collect()
    }

    fn clone_box(&self) -> Box<dyn InitializationStrategy> {
        Box::new(self.clone())
    }
}

/// Chaotic initialization using logistic map
///
/// Uses chaotic sequences to generate more diverse initial positions,
/// which can help avoid premature convergence
#[derive(Debug, Clone)]
pub struct ChaoticInitialization {
    /// Chaotic parameter (typically 4.0 for full chaos)
    pub mu: f64,
}

impl Default for ChaoticInitialization {
    fn default() -> Self {
        Self { mu: 4.0 }
    }
}

impl ChaoticInitialization {
    /// Creates a new chaotic initialization with default parameter
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new chaotic initialization with custom parameter
    ///
    /// # Arguments
    ///
    /// * `mu` - Chaotic parameter (typically between 3.57 and 4.0)
    pub fn with_mu(mu: f64) -> Self {
        Self { mu }
    }
}

impl InitializationStrategy for ChaoticInitialization {
    fn initialize_position(
        &self,
        dimensions: usize,
        lower_bounds: &[f64],
        upper_bounds: &[f64],
        rng: &mut dyn RngCore,
    ) -> Vec<f64> {
        use rand::Rng; // For gen_range method
        let mut position = Vec::with_capacity(dimensions);

        // Start with a random seed for the chaotic sequence
        let mut x: f64 = rng.gen_range(0.1..0.9);

        for i in 0..dimensions {
            // Logistic map: x_{n+1} = mu * x_n * (1 - x_n)
            x = self.mu * x * (1.0 - x);

            // Map chaotic value [0, 1] to the bound range
            let value = lower_bounds[i] + x * (upper_bounds[i] - lower_bounds[i]);
            position.push(value);
        }

        position
    }

    fn clone_box(&self) -> Box<dyn InitializationStrategy> {
        Box::new(self.clone())
    }
}

/// Trait for boundary handling strategies
///
/// Different boundary handling strategies affect how particles behave
/// when they leave the search space
pub trait BoundaryStrategy: Send + Sync {
    /// Handles a particle that has moved outside bounds
    ///
    /// # Arguments
    ///
    /// * `position` - Current position (may be out of bounds)
    /// * `velocity` - Current velocity
    /// * `dimension` - The dimension index that violated bounds
    /// * `lower_bound` - Lower bound for this dimension
    /// * `upper_bound` - Upper bound for this dimension
    /// * `rng` - Random number generator
    ///
    /// # Returns
    ///
    /// Tuple of (new_position, new_velocity) for the given dimension
    fn handle_boundary(
        &self,
        position: f64,
        velocity: f64,
        dimension: usize,
        lower_bound: f64,
        upper_bound: f64,
        rng: &mut dyn RngCore,
    ) -> (f64, f64);

    /// Clone the strategy into a Box
    fn clone_box(&self) -> Box<dyn BoundaryStrategy>;
}

impl Clone for Box<dyn BoundaryStrategy> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Reflect boundary strategy
///
/// When a particle hits a boundary, it bounces back by reflecting
/// its position to the boundary and inverting its velocity
#[derive(Debug, Clone)]
pub struct ReflectBoundary;

impl BoundaryStrategy for ReflectBoundary {
    fn handle_boundary(
        &self,
        position: f64,
        velocity: f64,
        _dimension: usize,
        lower_bound: f64,
        upper_bound: f64,
        _rng: &mut dyn RngCore,
    ) -> (f64, f64) {
        if position < lower_bound {
            (lower_bound, -velocity)
        } else if position > upper_bound {
            (upper_bound, -velocity)
        } else {
            (position, velocity)
        }
    }

    fn clone_box(&self) -> Box<dyn BoundaryStrategy> {
        Box::new(self.clone())
    }
}

/// Absorb boundary strategy
///
/// When a particle hits a boundary, it stops at the boundary
/// and its velocity in that dimension is set to zero
#[derive(Debug, Clone)]
pub struct AbsorbBoundary;

impl BoundaryStrategy for AbsorbBoundary {
    fn handle_boundary(
        &self,
        position: f64,
        velocity: f64,
        _dimension: usize,
        lower_bound: f64,
        upper_bound: f64,
        _rng: &mut dyn RngCore,
    ) -> (f64, f64) {
        if position < lower_bound {
            (lower_bound, 0.0)
        } else if position > upper_bound {
            (upper_bound, 0.0)
        } else {
            (position, velocity)
        }
    }

    fn clone_box(&self) -> Box<dyn BoundaryStrategy> {
        Box::new(self.clone())
    }
}

/// Random boundary strategy
///
/// When a particle hits a boundary, it is randomly repositioned
/// within the valid bounds with zero velocity
#[derive(Debug, Clone)]
pub struct RandomBoundary;

impl BoundaryStrategy for RandomBoundary {
    fn handle_boundary(
        &self,
        position: f64,
        velocity: f64,
        _dimension: usize,
        lower_bound: f64,
        upper_bound: f64,
        rng: &mut dyn RngCore,
    ) -> (f64, f64) {
        use rand::Rng; // For gen_range method
        if position < lower_bound || position > upper_bound {
            (rng.gen_range(lower_bound..=upper_bound), 0.0)
        } else {
            (position, velocity)
        }
    }

    fn clone_box(&self) -> Box<dyn BoundaryStrategy> {
        Box::new(self.clone())
    }
}

/// Trait for inertia weight strategies
///
/// The inertia weight controls the balance between exploration
/// and exploitation during the optimization process
pub trait InertiaStrategy: Send + Sync {
    /// Calculates the inertia weight for the current iteration
    ///
    /// # Arguments
    ///
    /// * `current_iteration` - Current iteration number (0-indexed)
    /// * `max_iterations` - Total number of iterations
    ///
    /// # Returns
    ///
    /// The inertia weight for this iteration
    fn get_inertia(&self, current_iteration: usize, max_iterations: usize) -> f64;

    /// Clone the strategy into a Box
    fn clone_box(&self) -> Box<dyn InertiaStrategy>;
}

impl Clone for Box<dyn InertiaStrategy> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Constant inertia weight (standard approach)
#[derive(Debug, Clone)]
pub struct ConstantInertia {
    pub weight: f64,
}

impl ConstantInertia {
    /// Creates a constant inertia strategy with the given weight
    pub fn new(weight: f64) -> Self {
        Self { weight }
    }
}

impl Default for ConstantInertia {
    fn default() -> Self {
        Self { weight: 0.7 }
    }
}

impl InertiaStrategy for ConstantInertia {
    fn get_inertia(&self, _current_iteration: usize, _max_iterations: usize) -> f64 {
        self.weight
    }

    fn clone_box(&self) -> Box<dyn InertiaStrategy> {
        Box::new(self.clone())
    }
}

/// Linearly decreasing inertia weight
///
/// Starts with high inertia (exploration) and decreases to low inertia (exploitation).
/// This is a highly effective strategy for balancing exploration and exploitation.
///
/// Formula: w(t) = w_max - (w_max - w_min) * (t / T)
/// where t is current iteration and T is total iterations
#[derive(Debug, Clone)]
pub struct LinearlyDecreasingInertia {
    /// Initial (maximum) inertia weight
    pub w_max: f64,
    /// Final (minimum) inertia weight
    pub w_min: f64,
}

impl Default for LinearlyDecreasingInertia {
    fn default() -> Self {
        Self {
            w_max: 0.9,
            w_min: 0.4,
        }
    }
}

impl LinearlyDecreasingInertia {
    /// Creates a linearly decreasing inertia strategy with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a linearly decreasing inertia strategy with custom values
    ///
    /// # Arguments
    ///
    /// * `w_max` - Initial inertia weight (typically 0.9)
    /// * `w_min` - Final inertia weight (typically 0.4)
    pub fn with_range(w_max: f64, w_min: f64) -> Self {
        Self { w_max, w_min }
    }
}

impl InertiaStrategy for LinearlyDecreasingInertia {
    fn get_inertia(&self, current_iteration: usize, max_iterations: usize) -> f64 {
        if max_iterations <= 1 {
            return self.w_max;
        }

        let t = current_iteration as f64;
        let t_max = (max_iterations - 1) as f64;

        self.w_max - (self.w_max - self.w_min) * (t / t_max)
    }

    fn clone_box(&self) -> Box<dyn InertiaStrategy> {
        Box::new(self.clone())
    }
}

/// Trait for topology strategies
///
/// Topology defines how particles interact and share information
pub trait TopologyStrategy: Send + Sync {
    /// Gets the best position that should influence a given particle
    ///
    /// # Arguments
    ///
    /// * `particle_idx` - Index of the particle being updated
    /// * `particles_best_positions` - Best positions of all particles
    /// * `particles_best_fitnesses` - Best fitnesses of all particles
    ///
    /// # Returns
    ///
    /// The best position that should influence this particle
    fn get_best_position(
        &self,
        particle_idx: usize,
        particles_best_positions: &[Vec<f64>],
        particles_best_fitnesses: &[f64],
    ) -> Vec<f64>;

    /// Clone the strategy into a Box
    fn clone_box(&self) -> Box<dyn TopologyStrategy>;
}

impl Clone for Box<dyn TopologyStrategy> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Global best topology (standard PSO)
///
/// All particles are influenced by the single best position
/// found by any particle in the entire swarm
#[derive(Debug, Clone)]
pub struct GlobalBest;

impl TopologyStrategy for GlobalBest {
    fn get_best_position(
        &self,
        _particle_idx: usize,
        particles_best_positions: &[Vec<f64>],
        particles_best_fitnesses: &[f64],
    ) -> Vec<f64> {
        // Find the global best
        let mut best_idx = 0;
        let mut best_fitness = particles_best_fitnesses[0];

        for (idx, &fitness) in particles_best_fitnesses.iter().enumerate().skip(1) {
            if fitness < best_fitness {
                best_fitness = fitness;
                best_idx = idx;
            }
        }

        particles_best_positions[best_idx].clone()
    }

    fn clone_box(&self) -> Box<dyn TopologyStrategy> {
        Box::new(self.clone())
    }
}

/// Local best topology with ring neighborhood
///
/// Each particle is influenced only by the best position within its
/// local neighborhood (defined as k nearest neighbors in a ring).
/// This promotes diversity and can help avoid premature convergence.
#[derive(Debug, Clone)]
pub struct LocalBest {
    /// Size of the neighborhood (number of neighbors on each side)
    pub neighborhood_size: usize,
}

impl Default for LocalBest {
    fn default() -> Self {
        Self {
            neighborhood_size: 2, // 2 neighbors on each side (5 total including self)
        }
    }
}

impl LocalBest {
    /// Creates a local best topology with default neighborhood size
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a local best topology with custom neighborhood size
    ///
    /// # Arguments
    ///
    /// * `neighborhood_size` - Number of neighbors on each side in the ring
    pub fn with_size(neighborhood_size: usize) -> Self {
        Self { neighborhood_size }
    }
}

impl TopologyStrategy for LocalBest {
    fn get_best_position(
        &self,
        particle_idx: usize,
        particles_best_positions: &[Vec<f64>],
        particles_best_fitnesses: &[f64],
    ) -> Vec<f64> {
        let swarm_size = particles_best_positions.len();

        // Define the ring neighborhood
        let mut best_idx = particle_idx;
        let mut best_fitness = particles_best_fitnesses[particle_idx];

        // Check neighbors on both sides
        for offset in 1..=self.neighborhood_size {
            // Left neighbor
            let left_idx = if particle_idx >= offset {
                particle_idx - offset
            } else {
                swarm_size + particle_idx - offset
            };

            if particles_best_fitnesses[left_idx] < best_fitness {
                best_fitness = particles_best_fitnesses[left_idx];
                best_idx = left_idx;
            }

            // Right neighbor
            let right_idx = (particle_idx + offset) % swarm_size;

            if particles_best_fitnesses[right_idx] < best_fitness {
                best_fitness = particles_best_fitnesses[right_idx];
                best_idx = right_idx;
            }
        }

        particles_best_positions[best_idx].clone()
    }

    fn clone_box(&self) -> Box<dyn TopologyStrategy> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_initialization() {
        let strategy = UniformInitialization;
        let mut rng = rand::thread_rng();

        let lower = vec![-5.0, -10.0, -15.0];
        let upper = vec![5.0, 10.0, 15.0];

        let position = strategy.initialize_position(3, &lower, &upper, &mut rng);

        assert_eq!(position.len(), 3);
        assert!(position[0] >= -5.0 && position[0] <= 5.0);
        assert!(position[1] >= -10.0 && position[1] <= 10.0);
        assert!(position[2] >= -15.0 && position[2] <= 15.0);
    }

    #[test]
    fn test_chaotic_initialization() {
        let strategy = ChaoticInitialization::new();
        let mut rng = rand::thread_rng();

        let lower = vec![-5.0; 5];
        let upper = vec![5.0; 5];

        let position = strategy.initialize_position(5, &lower, &upper, &mut rng);

        assert_eq!(position.len(), 5);
        for &val in &position {
            assert!(val >= -5.0 && val <= 5.0);
        }
    }

    #[test]
    fn test_reflect_boundary() {
        let strategy = ReflectBoundary;
        let mut rng = rand::thread_rng();

        // Test lower boundary violation
        let (pos, vel) = strategy.handle_boundary(-6.0, 2.0, 0, -5.0, 5.0, &mut rng);
        assert_eq!(pos, -5.0);
        assert_eq!(vel, -2.0);

        // Test upper boundary violation
        let (pos, vel) = strategy.handle_boundary(6.0, 2.0, 0, -5.0, 5.0, &mut rng);
        assert_eq!(pos, 5.0);
        assert_eq!(vel, -2.0);

        // Test within bounds
        let (pos, vel) = strategy.handle_boundary(0.0, 2.0, 0, -5.0, 5.0, &mut rng);
        assert_eq!(pos, 0.0);
        assert_eq!(vel, 2.0);
    }

    #[test]
    fn test_absorb_boundary() {
        let strategy = AbsorbBoundary;
        let mut rng = rand::thread_rng();

        // Test lower boundary violation
        let (pos, vel) = strategy.handle_boundary(-6.0, 2.0, 0, -5.0, 5.0, &mut rng);
        assert_eq!(pos, -5.0);
        assert_eq!(vel, 0.0);

        // Test upper boundary violation
        let (pos, vel) = strategy.handle_boundary(6.0, 2.0, 0, -5.0, 5.0, &mut rng);
        assert_eq!(pos, 5.0);
        assert_eq!(vel, 0.0);
    }

    #[test]
    fn test_random_boundary() {
        let strategy = RandomBoundary;
        let mut rng = rand::thread_rng();

        // Test boundary violation
        let (pos, vel) = strategy.handle_boundary(-6.0, 2.0, 0, -5.0, 5.0, &mut rng);
        assert!(pos >= -5.0 && pos <= 5.0);
        assert_eq!(vel, 0.0);
    }

    #[test]
    fn test_constant_inertia() {
        let strategy = ConstantInertia::new(0.7);

        assert_eq!(strategy.get_inertia(0, 100), 0.7);
        assert_eq!(strategy.get_inertia(50, 100), 0.7);
        assert_eq!(strategy.get_inertia(99, 100), 0.7);
    }

    #[test]
    fn test_linearly_decreasing_inertia() {
        let strategy = LinearlyDecreasingInertia::with_range(0.9, 0.4);

        // At start
        assert_eq!(strategy.get_inertia(0, 100), 0.9);

        // At middle
        let mid_inertia = strategy.get_inertia(50, 100);
        assert!((mid_inertia - 0.65).abs() < 0.01);

        // At end
        let end_inertia = strategy.get_inertia(99, 100);
        assert!((end_inertia - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_global_best_topology() {
        let topology = GlobalBest;

        let positions = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ];
        let fitnesses = vec![10.0, 5.0, 15.0];

        // All particles should get the same best position (index 1)
        for i in 0..3 {
            let best = topology.get_best_position(i, &positions, &fitnesses);
            assert_eq!(best, vec![3.0, 4.0]);
        }
    }

    #[test]
    fn test_local_best_topology() {
        let topology = LocalBest::with_size(1); // 1 neighbor on each side

        let positions = vec![
            vec![1.0],  // fitness: 10.0
            vec![2.0],  // fitness: 5.0
            vec![3.0],  // fitness: 15.0
            vec![4.0],  // fitness: 8.0
            vec![5.0],  // fitness: 12.0
        ];
        let fitnesses = vec![10.0, 5.0, 15.0, 8.0, 12.0];

        // Particle 0: neighbors are 4, 0, 1 -> best is 1 (fitness 5.0)
        let best = topology.get_best_position(0, &positions, &fitnesses);
        assert_eq!(best, vec![2.0]);

        // Particle 2: neighbors are 1, 2, 3 -> best is 1 (fitness 5.0)
        let best = topology.get_best_position(2, &positions, &fitnesses);
        assert_eq!(best, vec![2.0]);

        // Particle 3: neighbors are 2, 3, 4 -> best is 3 (fitness 8.0)
        let best = topology.get_best_position(3, &positions, &fitnesses);
        assert_eq!(best, vec![4.0]);
    }
}
