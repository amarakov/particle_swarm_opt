# Fitness Scaling Guide for Flat Landscapes

## Problem: Flat Fitness Landscapes

When optimizing functions like COSM MAXCUT, you may encounter fitness landscapes where differences between solutions are extremely small. For example:

```
Initial fitness: -5755.0500000000
Final fitness:   -5761.2000000000
Improvement:     0.11%
```

In this case, fitness values are clustered in a very narrow range (~5755 to ~5761), with differences of only 0.01-0.11%. This creates a **flat fitness landscape** that makes it extremely difficult for PSO to:

1. **Distinguish between solutions**: Small differences are masked by the large magnitude
2. **Find search directions**: The gradient is nearly zero everywhere
3. **Converge efficiently**: Random exploration dominates over directed search

## Solution: Fitness Scaling

Fitness scaling transforms fitness values to **amplify differences** while **preserving their relative ordering**. This makes the landscape more informative to the optimizer without changing which solutions are actually better.

### Key Benefits

- ✅ **Amplifies small differences** making them visible to the optimizer
- ✅ **Preserves solution ordering** (best solution remains best)
- ✅ **Improves convergence** by creating clearer search gradients
- ✅ **Transparent to user** (raw fitness still used for reporting)

## Available Scaling Strategies

### 1. Exponential Scaling ⭐ RECOMMENDED for COSM

**Best for:** Very flat landscapes with tiny differences (like COSM MAXCUT)

**Formula:** `scaled = exp(beta * (raw_fitness - min))`

**Parameters:**
- `beta`: Amplification factor (typical range: 0.001 to 0.1)
  - Smaller beta = less amplification (gentler)
  - Larger beta = more amplification (stronger)

**Example:**
```rust
use particle_swarm_opt::fitness_scaling::{FitnessScaler, ScalingStrategy};

let hyperparams = Hyperparameters::with_bounds(40, lower_bounds, upper_bounds)
    .with_fitness_scaling(
        FitnessScaler::new(ScalingStrategy::Exponential { beta: 0.01 })
    );
```

**Tuning beta:**
- Start with `beta = 0.01`
- If differences are still too small, increase to `0.05` or `0.1`
- If the landscape becomes too extreme, decrease to `0.005` or `0.001`

**When to use:**
- Fitness range is very narrow (< 1% of magnitude)
- Differences between solutions are < 0.1%
- You need aggressive amplification

---

### 2. Power Law Scaling

**Best for:** Moderately flat landscapes

**Formula:** `scaled = ((raw_fitness - min) / (max - min))^power`

**Parameters:**
- `power`: Exponent to apply (typical range: 1.5 to 3.0)
  - `power > 1`: Amplifies small differences
  - `power < 1`: Compresses large differences
  - `power = 1`: No effect (same as MinMax with target_max=1)

**Example:**
```rust
let hyperparams = Hyperparameters::with_bounds(40, lower_bounds, upper_bounds)
    .with_fitness_scaling(
        FitnessScaler::new(ScalingStrategy::PowerLaw { power: 2.0 })
    );
```

**When to use:**
- You want controlled amplification (less aggressive than exponential)
- Fitness range is moderately narrow (0.1% to 1% of magnitude)
- You want to maintain normalized [0, 1] range

---

### 3. MinMax Scaling

**Best for:** Simple normalization baseline

**Formula:** `scaled = (raw_fitness - min) / (max - min) * target_max`

**Parameters:**
- `target_max`: Target maximum value (typically 1.0)

**Example:**
```rust
let hyperparams = Hyperparameters::with_bounds(40, lower_bounds, upper_bounds)
    .with_fitness_scaling(
        FitnessScaler::new(ScalingStrategy::MinMax { target_max: 1.0 })
    );
```

**When to use:**
- You want simple [0, 1] normalization
- As a baseline to compare other strategies
- Fitness differences are moderate (> 1% of magnitude)

---

### 4. Z-Score Normalization

**Best for:** Emphasizing deviations from average

**Formula:** `scaled = (raw_fitness - mean) / std_dev`

**Parameters:** None (automatic)

**Example:**
```rust
let hyperparams = Hyperparameters::with_bounds(40, lower_bounds, upper_bounds)
    .with_fitness_scaling(
        FitnessScaler::new(ScalingStrategy::ZScore)
    );
```

**When to use:**
- You want to emphasize outliers
- Fitness distribution is roughly normal
- You care about relative position vs mean

---

### 5. Rank-Based Scaling

**Best for:** Highly irregular distributions

**Formula:** Converts fitness values to ranks (1, 2, 3, ...)

**Parameters:** None (automatic)

**Example:**
```rust
let hyperparams = Hyperparameters::with_bounds(40, lower_bounds, upper_bounds)
    .with_fitness_scaling(
        FitnessScaler::new(ScalingStrategy::Rank)
    );
```

**When to use:**
- Fitness distribution is highly irregular or has outliers
- You only care about relative ordering, not magnitude
- Most robust option but loses magnitude information

---

### 6. Logarithmic Scaling

**Best for:** Compressing very wide ranges

**Formula:** `scaled = log(1 + raw_fitness - min)`

**Parameters:** None (automatic)

**Example:**
```rust
let hyperparams = Hyperparameters::with_bounds(40, lower_bounds, upper_bounds)
    .with_fitness_scaling(
        FitnessScaler::new(ScalingStrategy::Logarithmic)
    );
```

**When to use:**
- Fitness values span many orders of magnitude
- You want to compress wide ranges
- Opposite use case from COSM (wide range, not narrow)

---

## Complete Example for COSM

Here's a complete example showing how to integrate fitness scaling into your COSM optimization:

```rust
use particle_swarm_opt::cosm::{CosmConfig, create_cosm_objective};
use particle_swarm_opt::fitness_scaling::{FitnessScaler, ScalingStrategy};
use particle_swarm_opt::{Hyperparameters, Swarm};

fn main() {
    // 1. Define your COSM configuration
    let config = CosmConfig::new(
        vec![
            "alpha_init".to_string(),
            "alpha_final".to_string(),
            "dpws_period".to_string(),
            "dpws_shift".to_string(),
            "dpws_window".to_string(),
        ],
        vec![0.0, 0.0, 0.0, 0.0, 0.0],           // Lower bounds
        vec![1.0, 0.1, 50.0, 100.0, 200.0],      // Upper bounds
    );

    // 2. Create your COSM objective function
    // Replace cosm_objective_function with your actual MAXCUT evaluation
    let objective = create_cosm_objective(&config);

    // 3. Set up hyperparameters WITH fitness scaling
    let hyperparams = Hyperparameters::with_bounds(
        40,  // swarm size
        config.lower_bounds.clone(),
        config.upper_bounds.clone(),
    )
    .with_fitness_scaling(
        // Use exponential scaling with beta=0.01 for flat landscapes
        FitnessScaler::new(ScalingStrategy::Exponential { beta: 0.01 })
    );

    // 4. Create and run the swarm
    let mut swarm = Swarm::new(hyperparams, &objective);
    let history = swarm.optimize(&objective, 50);

    // 5. Get results (raw fitness is automatically used for reporting)
    println!("Best fitness: {}", swarm.global_best_raw_fitness);
    println!("Best parameters: {:?}", swarm.global_best_position);

    // The history also contains raw (unscaled) fitness values
    for (i, iter_data) in history.iterations.iter().enumerate() {
        if i % 10 == 0 {
            println!("Iteration {}: {}", i, iter_data.global_best_fitness);
        }
    }
}
```

## How It Works Internally

### Automatic Scaling Flow

1. **Initialization:** Swarm evaluates raw fitness for all particles
2. **Update Scaler:** Statistics (min, max, mean, std) are computed from raw values
3. **Apply Scaling:** Each raw fitness is transformed using the scaling strategy
4. **PSO Operations:** All comparisons and updates use **scaled** fitness
5. **Reporting:** History and results use **raw** fitness for transparency

### Key Implementation Details

- **Dual fitness tracking:** Each particle stores both `raw_fitness` and `fitness` (scaled)
- **Per-iteration updates:** Scaler statistics are updated every iteration for adaptive scaling
- **Transparent reporting:** Users always see raw fitness in results and history
- **Preserve ordering:** Scaling never changes which solution is best, just amplifies differences

## Choosing the Right Strategy

### Decision Tree

```
Is your fitness range very narrow (< 0.1% differences)?
├─ YES → Use Exponential Scaling (start with beta=0.01)
└─ NO → Is the range moderately narrow (0.1% - 1%)?
    ├─ YES → Use Power Law (start with power=2.0)
    └─ NO → Is the distribution irregular with outliers?
        ├─ YES → Use Rank-Based Scaling
        └─ NO → Use MinMax Scaling (baseline)
```

### For COSM MAXCUT Specifically

**Recommended approach:**

1. **Start with:** Exponential scaling, `beta = 0.01`
2. **Monitor:** Check if differences are being amplified (compare scaled vs raw)
3. **Tune beta:**
   - If still too flat → increase beta to 0.05 or 0.1
   - If too extreme → decrease beta to 0.005
4. **Alternative:** Try Power Law with `power = 2.0` or `power = 3.0`

## Advanced: Comparing Strategies

Run the demo to compare all strategies on your specific problem:

```bash
cargo run --example fitness_scaling_demo
```

This will show you:
- Performance of each scaling strategy
- Convergence plots
- Final fitness values
- Recommendations for your problem

## Troubleshooting

### Scaling doesn't seem to help

**Possible causes:**
1. **Beta too small:** Try increasing beta (0.01 → 0.05 → 0.1)
2. **Wrong strategy:** Switch from MinMax to Exponential
3. **Truly flat landscape:** If all solutions are genuinely identical, scaling won't help

### PSO converges too fast

**Solution:** Reduce amplification
- Decrease beta (0.1 → 0.01 → 0.001)
- Use Power Law instead of Exponential
- Use lower power value

### Results are inconsistent

**Possible causes:**
1. **Stochastic fitness:** If your MAXCUT evaluation is stochastic, scaling amplifies noise
2. **Solution:** Run multiple trials and average results
3. **Alternative:** Use Rank-based scaling (most robust to noise)

## Performance Considerations

- **Overhead:** Scaling adds minimal overhead (~1-2% of total runtime)
- **Parallel-safe:** Scaler updates are separate from parallel fitness evaluation
- **Memory:** Negligible additional memory (just statistics storage)

## References and Further Reading

- Kennedy & Eberhart (1995): Original PSO paper
- Shi & Eberhart (1998): Inertia weight PSO
- Clerc & Kennedy (2002): Constriction coefficient PSO
- [PSO Visualization Guide](PSO_VISUALIZER.md)
- [Advanced Features Guide](examples/advanced_features.rs)

## API Reference

### FitnessScaler

```rust
pub struct FitnessScaler {
    pub strategy: ScalingStrategy,
    // internal statistics
}

impl FitnessScaler {
    pub fn new(strategy: ScalingStrategy) -> Self
    pub fn update(&mut self, fitness_values: &[f64])
    pub fn scale(&self, fitness: f64) -> f64
    pub fn scale_batch(&self, fitness_values: &[f64]) -> Vec<f64>
    pub fn stats(&self) -> (f64, f64, f64, f64) // (min, max, mean, std)
}
```

### ScalingStrategy

```rust
pub enum ScalingStrategy {
    None,
    MinMax { target_max: f64 },
    ZScore,
    Exponential { beta: f64 },
    Rank,
    PowerLaw { power: f64 },
    Logarithmic,
    RelativeToBaseline { baseline: f64 },
}
```

### Hyperparameters Integration

```rust
impl Hyperparameters {
    pub fn with_fitness_scaling(self, scaler: FitnessScaler) -> Self
}
```

## Contact and Support

If you encounter issues or have questions:
1. Check the [examples/](examples/) directory
2. Run `cargo run --example fitness_scaling_demo`
3. Review test cases in [src/fitness_scaling.rs](src/fitness_scaling.rs)
