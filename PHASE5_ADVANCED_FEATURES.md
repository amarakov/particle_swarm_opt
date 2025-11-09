# Phase 5: Advanced Features for Robustness

This document describes the advanced features implemented in Phase 5 to enhance the PSO algorithm's robustness and prevent it from getting stuck in local optima.

## Overview

Phase 5 introduces three major enhancements:

1. **Adaptive Inertia Weight** - Dynamic adjustment of exploration vs. exploitation
2. **Local Best (lBest) Topology** - Neighborhood-based information sharing
3. **Pluggable Strategy System** - Modular, trait-based architecture for customization

## 1. Adaptive Inertia Weight

### Linearly Decreasing Inertia Weight

The inertia weight `w` controls the influence of a particle's previous velocity on its current movement. A high inertia weight encourages exploration (searching new areas), while a low inertia weight encourages exploitation (refining current solutions).

**Formula:**
```
w(t) = w_max - (w_max - w_min) × (t / T)
```

Where:
- `t` is the current iteration
- `T` is the total number of iterations
- `w_max` is the initial inertia weight (default: 0.9)
- `w_min` is the final inertia weight (default: 0.4)

### Benefits

- **Early iterations (high w):** Particles explore the search space widely
- **Late iterations (low w):** Particles converge to promising regions
- **Proven effectiveness:** This is one of the most effective PSO improvements in the literature

### Usage

```rust
use particle_swarm_opt::{Hyperparameters, Swarm};

// Use the advanced configuration with linearly decreasing inertia
let hyperparams = Hyperparameters::advanced(50, 5, -10.0, 10.0);
let mut swarm = Swarm::new(hyperparams, fitness_fn);
```

Or customize the range:

```rust
use particle_swarm_opt::strategies::LinearlyDecreasingInertia;

let inertia_strategy = Box::new(LinearlyDecreasingInertia::with_range(0.95, 0.3));
```

## 2. Local Best (lBest) Topology

### Ring Neighborhood Topology

Instead of all particles being influenced by the global best position (gBest), each particle is influenced only by the best position within its local neighborhood. This is implemented as a ring topology where each particle has a fixed number of neighbors on each side.

**Example with neighborhood size 2:**
```
Particle 0: neighbors are [48, 49, 0, 1, 2]
Particle 1: neighbors are [49, 0, 1, 2, 3]
Particle 2: neighbors are [0, 1, 2, 3, 4]
...
```

### Benefits

- **Diversity preservation:** Prevents premature convergence to local optima
- **Multiple search fronts:** Different neighborhoods can explore different regions
- **Robustness:** More effective on complex, multimodal problems
- **Information flow:** Information still propagates through the swarm, just more gradually

### Usage

```rust
use particle_swarm_opt::{Hyperparameters, Swarm};

// Use advanced configuration (includes LocalBest by default)
let hyperparams = Hyperparameters::advanced(50, 5, -10.0, 10.0);
```

Or customize the neighborhood size:

```rust
use particle_swarm_opt::strategies::LocalBest;

let topology = Box::new(LocalBest::with_size(3)); // 3 neighbors on each side
```

## 3. Pluggable Strategy System

A modular, trait-based architecture allows easy customization and experimentation with different PSO variants.

### Strategy Traits

#### InitializationStrategy

Controls how particles are initialized at the start.

**Implementations:**
- `UniformInitialization` (default) - Random uniform distribution
- `ChaoticInitialization` - Uses chaotic logistic map for better diversity

```rust
use particle_swarm_opt::strategies::*;

// Chaotic initialization
let init_strategy = Box::new(ChaoticInitialization::new());
```

#### BoundaryStrategy

Controls what happens when a particle moves outside the search space bounds.

**Implementations:**
- `ReflectBoundary` (default) - Particle bounces back (velocity inverted)
- `AbsorbBoundary` - Particle stops at boundary (velocity set to 0)
- `RandomBoundary` - Particle repositioned randomly within bounds

```rust
use particle_swarm_opt::strategies::*;

// Absorb boundary
let boundary_strategy = Box::new(AbsorbBoundary);

// Random repositioning
let boundary_strategy = Box::new(RandomBoundary);
```

#### InertiaStrategy

Controls how the inertia weight changes over time.

**Implementations:**
- `ConstantInertia` (traditional) - Fixed weight throughout
- `LinearlyDecreasingInertia` (recommended) - Decreases linearly

```rust
use particle_swarm_opt::strategies::*;

// Custom range
let inertia = Box::new(LinearlyDecreasingInertia::with_range(0.95, 0.3));

// Constant (traditional PSO)
let inertia = Box::new(ConstantInertia::new(0.7));
```

#### TopologyStrategy

Controls how particles share information.

**Implementations:**
- `GlobalBest` (traditional) - All particles follow the swarm's best
- `LocalBest` (recommended) - Particles follow neighborhood best

```rust
use particle_swarm_opt::strategies::*;

// Local best with neighborhood size 3
let topology = Box::new(LocalBest::with_size(3));

// Global best (traditional PSO)
let topology = Box::new(GlobalBest);
```

### Full Custom Configuration

You can combine any strategies:

```rust
use particle_swarm_opt::{Hyperparameters, Swarm};
use particle_swarm_opt::strategies::*;

let hyperparams = Hyperparameters::with_strategies(
    50,                                              // swarm_size
    vec![-10.0; 5],                                 // lower_bounds
    vec![10.0; 5],                                  // upper_bounds
    Box::new(ChaoticInitialization::new()),         // initialization
    Box::new(RandomBoundary),                       // boundary handling
    Box::new(LinearlyDecreasingInertia::new()),     // inertia
    Box::new(LocalBest::with_size(2)),              // topology
);

let mut swarm = Swarm::new(hyperparams, fitness_fn);
```

## Quick Start Examples

### Standard PSO (Baseline)

```rust
use particle_swarm_opt::{Hyperparameters, Swarm};

let hyperparams = Hyperparameters::new(50, 5, -10.0, 10.0);
let mut swarm = Swarm::new(hyperparams, fitness_fn);
let history = swarm.optimize(fitness_fn, 200);
```

### Advanced PSO (All Phase 5 Features)

```rust
use particle_swarm_opt::{Hyperparameters, Swarm};

// Single line to get all advanced features!
let hyperparams = Hyperparameters::advanced(50, 5, -10.0, 10.0);
let mut swarm = Swarm::new(hyperparams, fitness_fn);
let history = swarm.optimize(fitness_fn, 200);
```

### Custom Configuration

```rust
use particle_swarm_opt::{Hyperparameters, Swarm};
use particle_swarm_opt::strategies::*;

let hyperparams = Hyperparameters::with_strategies(
    50,
    vec![-10.0; 5],
    vec![10.0; 5],
    Box::new(ChaoticInitialization::new()),
    Box::new(AbsorbBoundary),
    Box::new(LinearlyDecreasingInertia::with_range(0.95, 0.3)),
    Box::new(LocalBest::with_size(3)),
);

let mut swarm = Swarm::new(hyperparams, fitness_fn);
let history = swarm.optimize(fitness_fn, 200);
```

## Running Examples

```bash
# Run the comprehensive advanced features demo
cargo run --example advanced_features

# Run other examples (still work with backward compatibility)
cargo run --example simple_optimization
cargo run --example cosm_optimization
```

## Implementation Architecture

### Trait System

All strategies implement their respective trait with a `clone_box()` method to enable cloning of trait objects:

```rust
pub trait InitializationStrategy: Send + Sync {
    fn initialize_position(
        &self,
        dimensions: usize,
        lower_bounds: &[f64],
        upper_bounds: &[f64],
        rng: &mut dyn RngCore,
    ) -> Vec<f64>;

    fn clone_box(&self) -> Box<dyn InitializationStrategy>;
}
```

This design provides:
- **Type safety** - Compile-time checking of strategy implementations
- **Extensibility** - Easy to add new strategies
- **Performance** - Trait objects with minimal overhead
- **Thread safety** - Send + Sync bounds for parallel execution

### Key Design Decisions

1. **Trait objects (Box<dyn Trait>)** instead of enums
   - More extensible - users can implement their own strategies
   - Follows the Strategy pattern from OOP

2. **RngCore instead of generic Rng**
   - Enables trait object compatibility
   - Still allows all RNG functionality

3. **Backward compatibility**
   - Old code still works with `Hyperparameters::new()`
   - New features opt-in via `Hyperparameters::advanced()`

## Performance Considerations

- **Linearly Decreasing Inertia:** No overhead, just a simple calculation per iteration
- **Local Best Topology:** Small overhead for neighborhood lookup (O(k) where k is neighborhood size)
- **Strategy Pattern:** Minimal overhead from dynamic dispatch (typically 1-2 nanoseconds per call)

Overall, the advanced features add negligible computational cost while potentially improving solution quality significantly on complex problems.

## When to Use Advanced Features

### Use Advanced Features (Hyperparameters::advanced) When:

- ✅ The problem has multiple local optima
- ✅ The search space is complex or high-dimensional
- ✅ You want more robust optimization
- ✅ Standard PSO is converging prematurely

### Use Standard PSO (Hyperparameters::new) When:

- ✅ The problem is simple with few local optima
- ✅ Fast convergence is more important than exploration
- ✅ You want traditional PSO behavior for comparison

## Testing

All features are thoroughly tested:

```bash
# Run all tests
cargo test

# Run strategy tests specifically
cargo test strategies::tests
```

## References

- Shi, Y., & Eberhart, R. (1998). A modified particle swarm optimizer. IEEE International Conference on Evolutionary Computation.
- Kennedy, J., & Mendes, R. (2002). Population structure and particle swarm performance. IEEE Congress on Evolutionary Computation.
- Clerc, M., & Kennedy, J. (2002). The particle swarm - explosion, stability, and convergence in a multidimensional complex space.

## Summary

Phase 5 delivers a production-ready PSO implementation with state-of-the-art features:

1. ✅ **Adaptive Inertia Weight** - Linearly decreasing from 0.9 to 0.4
2. ✅ **lBest Topology** - Ring neighborhood with configurable size
3. ✅ **Pluggable Strategies** - Traits for initialization, boundaries, inertia, and topology
4. ✅ **Backward Compatible** - Existing code continues to work
5. ✅ **Well Tested** - Comprehensive test coverage
6. ✅ **Documented** - Examples and API documentation

The implementation is modular, extensible, and ready for both research and production use!
