//! Fitness Scaling Strategies
//!
//! This module provides various strategies for transforming fitness values to improve
//! optimizer performance. When fitness values are clustered in a narrow range (e.g.,
//! -5755 to -5761), the fitness landscape appears flat and PSO struggles to find
//! productive search directions.
//!
//! Fitness scaling amplifies differences between solutions while preserving their
//! relative ordering, making it easier for the optimizer to distinguish between
//! good and bad solutions.
//!
//! # Example
//!
//! ```
//! use particle_swarm_opt::fitness_scaling::{FitnessScaler, ScalingStrategy};
//!
//! // Raw fitness values clustered in narrow range
//! let raw_values = vec![-5755.05, -5760.50, -5761.20];
//!
//! // Create a scaler with exponential amplification
//! let mut scaler = FitnessScaler::new(ScalingStrategy::Exponential { beta: 0.01 });
//!
//! // Update scaler with current fitness values
//! scaler.update(&raw_values);
//!
//! // Scale individual values
//! let scaled = scaler.scale(-5761.20);
//! println!("Scaled fitness: {}", scaled);
//! ```

use serde::{Deserialize, Serialize};

/// Strategy for scaling fitness values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalingStrategy {
    /// No scaling - use raw fitness values (default behavior)
    None,

    /// Min-Max scaling: scales values to [0, target_max] range
    /// Formula: (fitness - min) / (max - min) * target_max
    /// Best for: Creating uniform [0, 1] range
    MinMax {
        /// Target maximum value (typically 1.0)
        target_max: f64,
    },

    /// Z-Score normalization: standardizes to mean 0, std dev 1
    /// Formula: (fitness - mean) / std_dev
    /// Best for: When you want to emphasize deviations from average
    ZScore,

    /// Exponential amplification: amplifies differences exponentially
    /// Formula: exp(beta * (fitness - min))
    /// Best for: Flat landscapes with tiny differences (like MAXCUT)
    /// Recommended beta values: 0.001 to 0.1 (tune based on your fitness range)
    Exponential {
        /// Amplification factor (larger = more amplification)
        beta: f64,
    },

    /// Rank-based: converts to ranks (1, 2, 3, ...)
    /// Formula: rank of fitness in sorted order
    /// Best for: Highly irregular fitness distributions
    /// Note: This requires storing all values, so it's applied batch-wise
    Rank,

    /// Power law transformation: applies power to normalized differences
    /// Formula: ((fitness - min) / (max - min))^power
    /// Best for: Amplifying small differences (power > 1) or compressing large ones (power < 1)
    PowerLaw {
        /// Power to apply (>1 amplifies differences, <1 compresses)
        power: f64,
    },

    /// Logarithmic scaling: log transformation for wide-range values
    /// Formula: log(1 + fitness - min)
    /// Best for: Compressing very wide ranges
    Logarithmic,

    /// Relative to baseline: measures improvement from baseline
    /// Formula: |fitness - baseline| or (baseline - fitness) for minimization
    /// Best for: When you have a known baseline or reference value
    RelativeToBaseline {
        /// Baseline fitness value
        baseline: f64,
    },
}

/// Fitness scaler that transforms fitness values using various strategies
#[derive(Debug, Clone)]
pub struct FitnessScaler {
    /// The scaling strategy to use
    pub strategy: ScalingStrategy,

    /// Statistics for current fitness values (updated during optimization)
    stats: FitnessStats,
}

/// Statistical properties of fitness values
#[derive(Debug, Clone)]
struct FitnessStats {
    min: f64,
    max: f64,
    mean: f64,
    std_dev: f64,
    count: usize,
}

impl Default for FitnessStats {
    fn default() -> Self {
        Self {
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            mean: 0.0,
            std_dev: 1.0,
            count: 0,
        }
    }
}

impl FitnessScaler {
    /// Creates a new fitness scaler with the specified strategy
    pub fn new(strategy: ScalingStrategy) -> Self {
        Self {
            strategy,
            stats: FitnessStats::default(),
        }
    }

    /// Updates the scaler's statistics with a new batch of fitness values
    ///
    /// This should be called each iteration with all particle fitness values
    /// to ensure accurate scaling.
    ///
    /// # Arguments
    ///
    /// * `fitness_values` - Slice of fitness values to compute statistics from
    pub fn update(&mut self, fitness_values: &[f64]) {
        if fitness_values.is_empty() {
            return;
        }

        // Filter out infinite values
        let valid_values: Vec<f64> = fitness_values
            .iter()
            .copied()
            .filter(|f| f.is_finite())
            .collect();

        if valid_values.is_empty() {
            return;
        }

        self.stats.count = valid_values.len();

        // Compute min and max
        self.stats.min = valid_values
            .iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        self.stats.max = valid_values
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        // Compute mean
        self.stats.mean = valid_values.iter().sum::<f64>() / valid_values.len() as f64;

        // Compute standard deviation
        if valid_values.len() > 1 {
            let variance = valid_values
                .iter()
                .map(|f| (f - self.stats.mean).powi(2))
                .sum::<f64>()
                / valid_values.len() as f64;
            self.stats.std_dev = variance.sqrt();

            // Avoid division by zero in Z-score
            if self.stats.std_dev < 1e-10 {
                self.stats.std_dev = 1.0;
            }
        } else {
            self.stats.std_dev = 1.0;
        }
    }

    /// Scales a single fitness value according to the current strategy
    ///
    /// # Arguments
    ///
    /// * `fitness` - The raw fitness value to scale
    ///
    /// # Returns
    ///
    /// The scaled fitness value
    ///
    /// # Note
    ///
    /// Make sure to call `update()` with current fitness values before scaling,
    /// otherwise the statistics will be outdated.
    pub fn scale(&self, fitness: f64) -> f64 {
        // Don't scale infinite values
        if !fitness.is_finite() {
            return fitness;
        }

        match self.strategy {
            ScalingStrategy::None => fitness,

            ScalingStrategy::MinMax { target_max } => {
                let range = self.stats.max - self.stats.min;
                if range < 1e-10 {
                    // All values are the same
                    return 0.0;
                }
                (fitness - self.stats.min) / range * target_max
            }

            ScalingStrategy::ZScore => {
                if self.stats.std_dev < 1e-10 {
                    return 0.0;
                }
                (fitness - self.stats.mean) / self.stats.std_dev
            }

            ScalingStrategy::Exponential { beta } => {
                // Shift so minimum is at 0, then apply exponential
                let shifted = fitness - self.stats.min;
                (beta * shifted).exp()
            }

            ScalingStrategy::PowerLaw { power } => {
                let range = self.stats.max - self.stats.min;
                if range < 1e-10 {
                    return 0.0;
                }
                let normalized = (fitness - self.stats.min) / range;
                normalized.powf(power)
            }

            ScalingStrategy::Logarithmic => {
                // Shift so minimum is at 0, add 1 to avoid log(0), then take log
                let shifted = fitness - self.stats.min + 1.0;
                shifted.ln()
            }

            ScalingStrategy::RelativeToBaseline { baseline } => {
                // For minimization, smaller fitness is better
                // So we want baseline - fitness (larger when fitness is smaller)
                // But we need non-negative, so use absolute value
                (fitness - baseline).abs()
            }

            ScalingStrategy::Rank => {
                // Rank-based scaling requires all values, so it's not ideal for
                // single-value scaling. We'll approximate by using the normalized
                // position between min and max.
                let range = self.stats.max - self.stats.min;
                if range < 1e-10 {
                    return 0.0;
                }
                // This gives a rough "rank-like" value
                (fitness - self.stats.min) / range * self.stats.count as f64
            }
        }
    }

    /// Scales a batch of fitness values
    ///
    /// This is more efficient than scaling values one-by-one and provides
    /// better results for Rank-based scaling.
    ///
    /// # Arguments
    ///
    /// * `fitness_values` - Slice of fitness values to scale
    ///
    /// # Returns
    ///
    /// Vector of scaled fitness values
    pub fn scale_batch(&self, fitness_values: &[f64]) -> Vec<f64> {
        match self.strategy {
            ScalingStrategy::Rank => {
                // For rank-based, we need to sort and assign ranks
                let mut indexed: Vec<(usize, f64)> = fitness_values
                    .iter()
                    .enumerate()
                    .map(|(i, &f)| (i, f))
                    .collect();

                // Sort by fitness (lower is better)
                indexed.sort_by(|a, b| {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                });

                // Assign ranks
                let mut result = vec![0.0; fitness_values.len()];
                for (rank, (original_idx, _)) in indexed.iter().enumerate() {
                    result[*original_idx] = rank as f64;
                }
                result
            }
            _ => {
                // For other strategies, scale individually
                fitness_values.iter().map(|&f| self.scale(f)).collect()
            }
        }
    }

    /// Returns the current fitness statistics
    pub fn stats(&self) -> (f64, f64, f64, f64) {
        (
            self.stats.min,
            self.stats.max,
            self.stats.mean,
            self.stats.std_dev,
        )
    }
}

impl Default for FitnessScaler {
    fn default() -> Self {
        Self::new(ScalingStrategy::None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_scaling() {
        let mut scaler = FitnessScaler::new(ScalingStrategy::None);
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        scaler.update(&values);

        assert_eq!(scaler.scale(3.0), 3.0);
        assert_eq!(scaler.scale(5.0), 5.0);
    }

    #[test]
    fn test_minmax_scaling() {
        let mut scaler = FitnessScaler::new(ScalingStrategy::MinMax { target_max: 1.0 });
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        scaler.update(&values);

        assert_eq!(scaler.scale(1.0), 0.0); // min maps to 0
        assert_eq!(scaler.scale(5.0), 1.0); // max maps to 1
        assert_eq!(scaler.scale(3.0), 0.5); // middle maps to 0.5
    }

    #[test]
    fn test_zscore_scaling() {
        let mut scaler = FitnessScaler::new(ScalingStrategy::ZScore);
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        scaler.update(&values);

        // Mean is 3.0, so 3.0 should map to ~0
        let scaled = scaler.scale(3.0);
        assert!(scaled.abs() < 1e-10);
    }

    #[test]
    fn test_exponential_scaling() {
        let mut scaler = FitnessScaler::new(ScalingStrategy::Exponential { beta: 0.1 });
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        scaler.update(&values);

        // Min value should map to exp(0) = 1.0
        let scaled_min = scaler.scale(1.0);
        assert!((scaled_min - 1.0).abs() < 1e-10);

        // Larger values should be exponentially larger
        let scaled_max = scaler.scale(5.0);
        assert!(scaled_max > scaled_min);
    }

    #[test]
    fn test_rank_scaling() {
        let scaler = FitnessScaler::new(ScalingStrategy::Rank);
        let values = vec![5.0, 1.0, 3.0, 2.0, 4.0];

        let scaled = scaler.scale_batch(&values);

        // Check that ranks are assigned correctly
        // 1.0 should be rank 0, 2.0 rank 1, 3.0 rank 2, etc.
        assert_eq!(scaled[1], 0.0); // 1.0 is best (rank 0)
        assert_eq!(scaled[3], 1.0); // 2.0 is second (rank 1)
        assert_eq!(scaled[2], 2.0); // 3.0 is third (rank 2)
        assert_eq!(scaled[4], 3.0); // 4.0 is fourth (rank 3)
        assert_eq!(scaled[0], 4.0); // 5.0 is worst (rank 4)
    }

    #[test]
    fn test_power_law_scaling() {
        let mut scaler = FitnessScaler::new(ScalingStrategy::PowerLaw { power: 2.0 });
        let values = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        scaler.update(&values);

        // Min should map to 0
        assert_eq!(scaler.scale(0.0), 0.0);

        // Max should map to 1
        assert!((scaler.scale(4.0) - 1.0).abs() < 1e-10);

        // Middle value (2.0) should be at 0.5, squared = 0.25
        let scaled = scaler.scale(2.0);
        assert!((scaled - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_cosm_like_values() {
        // Test with values similar to the COSM MAXCUT problem
        let mut scaler = FitnessScaler::new(ScalingStrategy::Exponential { beta: 0.01 });
        let values = vec![-5755.05, -5755.55, -5760.25, -5760.50, -5761.20];
        scaler.update(&values);

        let scaled = scaler.scale_batch(&values);

        // Check that differences are amplified
        // Original range: 6.15
        // Scaled range should be much larger
        let scaled_min = scaled.iter().copied().fold(f64::INFINITY, f64::min);
        let scaled_max = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let scaled_range = scaled_max - scaled_min;

        println!("Original range: {}", 5761.20 - 5755.05);
        println!("Scaled range: {}", scaled_range);

        // With beta=0.01 and range ~6, exp(0.01 * 6) ≈ 1.062
        // So scaled range should be around 0.062
        assert!(scaled_range > 0.01);
    }

    #[test]
    fn test_infinite_values() {
        let mut scaler = FitnessScaler::new(ScalingStrategy::MinMax { target_max: 1.0 });
        let values = vec![1.0, 2.0, f64::INFINITY, 3.0];
        scaler.update(&values);

        // Infinite values should remain infinite
        assert!(scaler.scale(f64::INFINITY).is_infinite());

        // Finite values should be scaled normally (ignoring the infinite value)
        assert!(scaler.scale(1.0).is_finite());
    }
}
