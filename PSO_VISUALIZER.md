# PSO Visualization Guide

Comprehensive guide to visualizing Particle Swarm Optimization runs using the `particle_swarm_opt` library.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Visualization Types](#visualization-types)
  - [Convergence Curves](#convergence-curves)
  - [2D Landscape Plots](#2d-landscape-plots)
  - [Animation Frames](#animation-frames)
- [API Reference](#api-reference)
- [Complete Examples](#complete-examples)
- [Creating Animations](#creating-animations)
- [Customization](#customization)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

The visualization module provides three main capabilities:

1. **Convergence Curves**: Line plots showing fitness improvement over iterations
2. **2D Landscape Plots**: Contour plots of the objective function with particle positions
3. **Animation Frames**: Sequences of images showing swarm movement over time

All visualizations are generated using the `plotters` crate and output as high-quality PNG images.

## Quick Start

```rust
use particle_swarm_opt::{Swarm, Hyperparameters};
use particle_swarm_opt::visualization::generate_all_plots;

fn main() {
    // Define fitness function (2D sphere function)
    let fitness_fn = |x: &[f64]| x.iter().map(|v| v * v).sum();

    // Setup and run optimization
    let hyperparams = Hyperparameters::new(30, 2, -10.0, 10.0);
    let mut swarm = Swarm::new(hyperparams, &fitness_fn);
    let history = swarm.optimize(&fitness_fn, 100);

    // Generate all visualizations
    let bounds = (-10.0, 10.0, -10.0, 10.0);
    generate_all_plots(
        &fitness_fn,
        &history,
        "output",              // output directory
        Some(bounds),          // plot bounds for 2D landscapes
        true,                  // generate animation frames
        None,                  // use default plot config
    ).unwrap();

    println!("Visualizations saved to output/");
}
```

Run the example:
```bash
cargo run --example simple_visualization
```

## Visualization Types

### Convergence Curves

Convergence curves show how the global best fitness improves over iterations. This is essential for:
- Diagnosing optimization performance
- Identifying convergence speed
- Detecting premature convergence or stagnation

**Example:**

```rust
use particle_swarm_opt::visualization::plot_convergence;

// After optimization...
plot_convergence(
    &history,
    "convergence.png",
    Some("My Optimization Convergence"),  // optional title
    None,                                  // use default config
).unwrap();
```

**Output:**
- X-axis: Iteration number
- Y-axis: Best fitness value
- Blue line with markers showing fitness at each iteration

**Use Cases:**
- Works with any dimensionality (2D, 10D, 100D, etc.)
- Compare different hyperparameter settings
- Identify optimal iteration count

### 2D Landscape Plots

2D landscape plots visualize the fitness landscape with particle positions overlaid. Only works for 2D optimization problems.

**Example:**

```rust
use particle_swarm_opt::visualization::plot_2d_landscape;

let fitness_fn = |x: &[f64]| {
    // Rastrigin function
    let a = 10.0;
    a * 2.0 + x.iter().map(|v| v*v - a*(2.0*std::f64::consts::PI*v).cos()).sum::<f64>()
};

// ... run optimization ...

let bounds = (-5.12, 5.12, -5.12, 5.12);  // (x_min, x_max, y_min, y_max)

// Plot initial state
plot_2d_landscape(
    &fitness_fn,
    &history,
    0,                              // iteration 0
    "landscape_initial.png",
    bounds,
    Some("Initial Swarm State"),
    None,
).unwrap();

// Plot final state
plot_2d_landscape(
    &fitness_fn,
    &history,
    history.len() - 1,              // last iteration
    "landscape_final.png",
    bounds,
    Some("Final Swarm State"),
    None,
).unwrap();
```

**Output:**
- Background: Color-coded fitness landscape (blue=low fitness, red=high fitness)
- Black dots: Current particle positions
- Red dot: Global best position
- Resolution: 100x100 grid for contour calculation

**Use Cases:**
- Visualize exploration vs exploitation
- See how swarm converges to optima
- Identify if swarm is stuck in local minima

### Animation Frames

Generate a sequence of PNG images showing swarm movement over time. These can be stitched into GIF or video.

**Example:**

```rust
use particle_swarm_opt::visualization::generate_animation_frames;

let bounds = (-10.0, 10.0, -10.0, 10.0);

generate_animation_frames(
    &fitness_fn,
    &history,
    "animation_frames",  // output directory
    bounds,
    None,                // use default config
).unwrap();

// Creates: animation_frames/frame_0000.png, frame_0001.png, ..., frame_0099.png
```

**Output:**
- One PNG file per iteration
- Numbered sequentially: `frame_0000.png`, `frame_0001.png`, etc.
- Each frame shows the landscape with particle positions at that iteration

**Use Cases:**
- Create animations for presentations
- Study swarm dynamics over time
- Share optimization results visually

## API Reference

### `plot_convergence`

```rust
pub fn plot_convergence<P: AsRef<Path>>(
    history: &OptimizationHistory,
    output_path: P,
    title: Option<&str>,
    config: Option<PlotConfig>,
) -> Result<(), Box<dyn Error>>
```

**Parameters:**
- `history`: Optimization history from `swarm.optimize()`
- `output_path`: Where to save the PNG file
- `title`: Optional custom title (default: "PSO Convergence: Best Fitness vs Iteration")
- `config`: Optional plot configuration (default: 1200×800px, 150 DPI)

**Returns:** `Ok(())` on success, error if history is empty or I/O fails

---

### `plot_2d_landscape`

```rust
pub fn plot_2d_landscape<F, P: AsRef<Path>>(
    fitness_fn: F,
    history: &OptimizationHistory,
    iteration: usize,
    output_path: P,
    bounds: (f64, f64, f64, f64),  // (x_min, x_max, y_min, y_max)
    title: Option<&str>,
    config: Option<PlotConfig>,
) -> Result<(), Box<dyn Error>>
where
    F: Fn(&[f64]) -> f64,
```

**Parameters:**
- `fitness_fn`: The objective function to visualize
- `history`: Optimization history
- `iteration`: Which iteration to plot (0-based index)
- `output_path`: Where to save the PNG file
- `bounds`: Plot area bounds (x_min, x_max, y_min, y_max)
- `title`: Optional custom title
- `config`: Optional plot configuration

**Returns:** `Ok(())` on success, error if iteration is out of bounds or problem isn't 2D

---

### `generate_animation_frames`

```rust
pub fn generate_animation_frames<F, P: AsRef<Path>>(
    fitness_fn: F,
    history: &OptimizationHistory,
    output_dir: P,
    bounds: (f64, f64, f64, f64),
    config: Option<PlotConfig>,
) -> Result<(), Box<dyn Error>>
where
    F: Fn(&[f64]) -> f64,
```

**Parameters:**
- `fitness_fn`: The objective function to visualize
- `history`: Optimization history
- `output_dir`: Directory where frames will be saved (created if doesn't exist)
- `bounds`: Plot area bounds
- `config`: Optional plot configuration

**Returns:** `Ok(())` on success, error on I/O failure

**Note:** Creates one frame per iteration with progress indicators every 10 frames.

---

### `generate_all_plots`

```rust
pub fn generate_all_plots<F, P: AsRef<Path>>(
    fitness_fn: F,
    history: &OptimizationHistory,
    output_dir: P,
    bounds: Option<(f64, f64, f64, f64)>,
    generate_animation: bool,
    config: Option<PlotConfig>,
) -> Result<(), Box<dyn Error>>
where
    F: Fn(&[f64]) -> f64,
```

**Parameters:**
- `fitness_fn`: The objective function
- `history`: Optimization history
- `output_dir`: Base directory for all outputs
- `bounds`: Optional bounds for 2D landscapes (None skips landscape plots)
- `generate_animation`: Whether to generate animation frames
- `config`: Optional plot configuration

**Generates:**
1. `convergence.png` - Convergence curve
2. `landscape_initial.png` - Initial state (if bounds provided and 2D)
3. `landscape_final.png` - Final state (if bounds provided and 2D)
4. `animation_frames/` - Animation frames (if `generate_animation` is true)

---

### `PlotConfig`

```rust
pub struct PlotConfig {
    pub width: u32,      // Image width in pixels
    pub height: u32,     // Image height in pixels
    pub dpi: f64,        // DPI for high-resolution output
}

impl Default for PlotConfig {
    fn default() -> Self {
        Self {
            width: 1200,
            height: 800,
            dpi: 150.0,
        }
    }
}
```

**Custom Configuration Example:**

```rust
use particle_swarm_opt::visualization::{PlotConfig, plot_convergence};

let config = PlotConfig {
    width: 1920,    // High resolution for presentations
    height: 1080,
    dpi: 300.0,     // Print quality
};

plot_convergence(&history, "convergence_hd.png", None, Some(config)).unwrap();
```

## Complete Examples

### Example 1: Sphere Function Optimization

```rust
use particle_swarm_opt::{Swarm, Hyperparameters};
use particle_swarm_opt::visualization::{plot_convergence, plot_2d_landscape};

fn main() {
    // Simple sphere function: f(x,y) = x² + y²
    // Global minimum at (0, 0) with value 0
    let fitness_fn = |pos: &[f64]| -> f64 {
        pos.iter().map(|x| x * x).sum()
    };

    // Setup
    let hyperparams = Hyperparameters::new(30, 2, -10.0, 10.0);
    let mut swarm = Swarm::new(hyperparams, &fitness_fn);

    println!("Initial best: {:.6}", swarm.global_best_fitness);

    // Optimize
    let history = swarm.optimize(&fitness_fn, 50);

    println!("Final best: {:.6}", swarm.global_best_fitness);
    println!("Position: [{:.4}, {:.4}]",
             swarm.global_best_position[0],
             swarm.global_best_position[1]);

    // Create output directory
    std::fs::create_dir_all("sphere_output").unwrap();

    // 1. Convergence plot
    plot_convergence(
        &history,
        "sphere_output/convergence.png",
        Some("Sphere Function: f(x,y) = x² + y²"),
        None,
    ).unwrap();

    // 2. Landscape plots
    let bounds = (-10.0, 10.0, -10.0, 10.0);

    plot_2d_landscape(
        &fitness_fn,
        &history,
        0,
        "sphere_output/initial.png",
        bounds,
        Some("Initial Swarm Distribution"),
        None,
    ).unwrap();

    plot_2d_landscape(
        &fitness_fn,
        &history,
        history.len() - 1,
        "sphere_output/final.png",
        bounds,
        Some("Converged Swarm"),
        None,
    ).unwrap();

    println!("✓ Visualizations saved to sphere_output/");
}
```

### Example 2: Rosenbrock Function (Narrow Valley)

```rust
use particle_swarm_opt::{Swarm, Hyperparameters};
use particle_swarm_opt::visualization::generate_all_plots;

fn main() {
    // Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²
    // Global minimum at (1, 1) with value 0
    // Known for its narrow valley making it difficult to optimize
    let rosenbrock = |pos: &[f64]| -> f64 {
        let x = pos[0];
        let y = pos[1];
        (1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2)
    };

    // Use more iterations for this challenging function
    let hyperparams = Hyperparameters::new(50, 2, -2.0, 2.0);
    let mut swarm = Swarm::new(hyperparams, &rosenbrock);
    let history = swarm.optimize(&rosenbrock, 200);

    println!("Final fitness: {:.6}", swarm.global_best_fitness);
    println!("Best position: [{:.4}, {:.4}]",
             swarm.global_best_position[0],
             swarm.global_best_position[1]);
    println!("(Optimal is [1.0, 1.0])");

    // Generate all visualizations
    let bounds = (-2.0, 2.0, -2.0, 2.0);
    generate_all_plots(
        &rosenbrock,
        &history,
        "rosenbrock_output",
        Some(bounds),
        false,  // Skip animation (200 frames is a lot!)
        None,
    ).unwrap();
}
```

### Example 3: High-Dimensional Problem (Convergence Only)

```rust
use particle_swarm_opt::{Swarm, Hyperparameters};
use particle_swarm_opt::visualization::plot_convergence;

fn main() {
    // 10-dimensional sphere function
    let fitness_fn = |pos: &[f64]| -> f64 {
        pos.iter().map(|x| x * x).sum()
    };

    // 10D problem
    let hyperparams = Hyperparameters::new(100, 10, -100.0, 100.0);
    let mut swarm = Swarm::new(hyperparams, &fitness_fn);
    let history = swarm.optimize(&fitness_fn, 200);

    println!("Final fitness: {:.6}", swarm.global_best_fitness);

    // Only convergence plot (can't visualize 10D landscape!)
    plot_convergence(
        &history,
        "high_dim_convergence.png",
        Some("10D Sphere Function Optimization"),
        None,
    ).unwrap();
}
```

## Creating Animations

After generating animation frames, use FFmpeg to create GIF or MP4 files.

### Install FFmpeg

**Ubuntu/Debian:**
```bash
sudo apt-get install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html)

### Create GIF Animation

```bash
# Basic GIF (10 fps)
ffmpeg -framerate 10 -pattern_type glob -i 'animation_frames/*.png' \
       -vf "scale=800:-1" animation.gif

# High quality GIF with palette optimization
ffmpeg -framerate 10 -pattern_type glob -i 'animation_frames/*.png' \
       -vf "fps=10,scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
       animation_optimized.gif

# Slower animation (5 fps for better viewing)
ffmpeg -framerate 5 -pattern_type glob -i 'animation_frames/*.png' \
       -vf "scale=800:-1" animation_slow.gif
```

### Create MP4 Video

```bash
# Standard MP4 video
ffmpeg -framerate 10 -pattern_type glob -i 'animation_frames/*.png' \
       -c:v libx264 -pix_fmt yuv420p animation.mp4

# High quality MP4
ffmpeg -framerate 10 -pattern_type glob -i 'animation_frames/*.png' \
       -c:v libx264 -crf 18 -pix_fmt yuv420p animation_hq.mp4
```

### Parameters Explained

- `-framerate 10`: 10 frames per second (adjust as needed)
- `-pattern_type glob`: Use glob pattern for input files
- `-i 'animation_frames/*.png'`: Input file pattern
- `-vf "scale=800:-1"`: Scale to 800px width (height auto)
- `-c:v libx264`: Use H.264 codec for MP4
- `-crf 18`: Constant Rate Factor (lower = better quality, 18-28 recommended)
- `palettegen/paletteuse`: Optimize GIF colors for better quality

### Example Workflow

```rust
use particle_swarm_opt::visualization::generate_animation_frames;

// Generate frames
let bounds = (-5.0, 5.0, -5.0, 5.0);
generate_animation_frames(
    &fitness_fn,
    &history,
    "frames",
    bounds,
    None,
).unwrap();

println!("Frames generated! Create animation with:");
println!("  ffmpeg -framerate 10 -pattern_type glob -i 'frames/*.png' animation.gif");
```

Then run:
```bash
ffmpeg -framerate 10 -pattern_type glob -i 'frames/*.png' animation.gif
```

## Customization

### Custom Plot Size and DPI

```rust
use particle_swarm_opt::visualization::{PlotConfig, plot_convergence};

// 4K resolution for presentations
let config = PlotConfig {
    width: 3840,
    height: 2160,
    dpi: 300.0,
};

plot_convergence(&history, "convergence_4k.png", None, Some(config)).unwrap();
```

### Custom Plot Bounds

Adjust bounds to focus on specific regions:

```rust
// Standard bounds
let bounds = (-10.0, 10.0, -10.0, 10.0);

// Zoomed in on convergence area
let zoom_bounds = (-2.0, 2.0, -2.0, 2.0);

plot_2d_landscape(
    &fitness_fn,
    &history,
    history.len() - 1,
    "zoomed_final.png",
    zoom_bounds,
    Some("Zoomed Final State"),
    None,
).unwrap();
```

### Comparing Different Runs

```rust
// Run optimization with different hyperparameters
let configs = vec![
    ("small_swarm", 20, 0.5),
    ("medium_swarm", 50, 0.7),
    ("large_swarm", 100, 0.9),
];

for (name, size, inertia) in configs {
    let mut hyperparams = Hyperparameters::new(size, 2, -10.0, 10.0);
    hyperparams.inertia_weight = inertia;

    let mut swarm = Swarm::new(hyperparams, &fitness_fn);
    let history = swarm.optimize(&fitness_fn, 100);

    plot_convergence(
        &history,
        &format!("convergence_{}.png", name),
        Some(&format!("Swarm Size: {}, Inertia: {}", size, inertia)),
        None,
    ).unwrap();
}
```

## Best Practices

### 1. Choose Appropriate Iteration Counts for Animation

```rust
// Good: 50-100 iterations for animation
let history = swarm.optimize(&fitness_fn, 50);
generate_animation_frames(&fitness_fn, &history, "frames", bounds, None).unwrap();

// Avoid: Too many frames (slow to generate and large file size)
// let history = swarm.optimize(&fitness_fn, 1000);  // 1000 PNG files!
```

### 2. Set Bounds to Match Fitness Function

```rust
// Rastrigin: typical bounds are ±5.12
let rastrigin_bounds = (-5.12, 5.12, -5.12, 5.12);

// Rosenbrock: often tested in ±2 or ±5
let rosenbrock_bounds = (-2.0, 2.0, -2.0, 2.0);

// Sphere: can be anything, ±10 is common
let sphere_bounds = (-10.0, 10.0, -10.0, 10.0);
```

### 3. Use Landscape Plots Only for 2D Problems

```rust
// Good: 2D problem
let hyperparams = Hyperparameters::new(30, 2, -10.0, 10.0);

// For higher dimensions, use convergence plots only
let hyperparams_10d = Hyperparameters::new(100, 10, -10.0, 10.0);
// plot_2d_landscape() will return an error for 10D problems
```

### 4. Directory Organization

```rust
// Organize outputs by experiment
generate_all_plots(
    &fitness_fn,
    &history,
    "experiments/rastrigin/run_001",  // Clear organization
    Some(bounds),
    true,
    None,
).unwrap();
```

### 5. Save History for Later Visualization

```rust
use particle_swarm_opt::results::save_history_json;

// Run optimization
let history = swarm.optimize(&fitness_fn, 100);

// Save history
save_history_json(&history, "history.json").unwrap();

// Later: load and visualize
use particle_swarm_opt::results::load_history_json;
let loaded_history = load_history_json("history.json").unwrap();
plot_convergence(&loaded_history, "convergence.png", None, None).unwrap();
```

## Troubleshooting

### Problem: "Cannot plot empty history"

**Cause:** No iterations were run before calling visualization.

**Solution:**
```rust
// Wrong
let history = OptimizationHistory::new();
plot_convergence(&history, "out.png", None, None).unwrap();  // Error!

// Correct
let history = swarm.optimize(&fitness_fn, 50);  // Run at least 1 iteration
plot_convergence(&history, "out.png", None, None).unwrap();  // Works!
```

### Problem: "Landscape plotting only supports 2D problems"

**Cause:** Trying to use `plot_2d_landscape()` on 3D+ problem.

**Solution:**
```rust
// Only for 2D problems
let hyperparams = Hyperparameters::new(30, 2, -10.0, 10.0);

// For higher dimensions, use convergence plots only
let hyperparams_5d = Hyperparameters::new(50, 5, -10.0, 10.0);
// Skip landscape plots, just use plot_convergence()
```

### Problem: "Iteration X out of bounds"

**Cause:** Requesting an iteration that doesn't exist.

**Solution:**
```rust
let history = swarm.optimize(&fitness_fn, 50);  // Iterations 0-49

// Wrong
plot_2d_landscape(&fitness_fn, &history, 50, "out.png", bounds, None, None).unwrap();  // Error!

// Correct
plot_2d_landscape(&fitness_fn, &history, 49, "out.png", bounds, None, None).unwrap();  // OK

// Or use: history.len() - 1 for last iteration
plot_2d_landscape(&fitness_fn, &history, history.len() - 1, "out.png", bounds, None, None).unwrap();
```

### Problem: Animation generation is slow

**Cause:** Each frame requires evaluating the fitness function on a 100×100 grid.

**Solutions:**
```rust
// 1. Use fewer iterations
let history = swarm.optimize(&fitness_fn, 30);  // 30 frames instead of 100

// 2. Skip animation for high iteration counts
generate_all_plots(
    &fitness_fn,
    &history,
    "output",
    Some(bounds),
    false,  // Skip animation
    None,
).unwrap();

// 3. Generate only key frames manually
for i in (0..history.len()).step_by(5) {  // Every 5th iteration
    plot_2d_landscape(
        &fitness_fn,
        &history,
        i,
        &format!("frames/frame_{:04}.png", i / 5),
        bounds,
        None,
        None,
    ).unwrap();
}
```

### Problem: Plots look blurry/pixelated

**Solution:** Increase resolution and DPI:
```rust
let config = PlotConfig {
    width: 1920,
    height: 1080,
    dpi: 300.0,  // High DPI for sharp text
};

plot_convergence(&history, "convergence_hd.png", None, Some(config)).unwrap();
```

### Problem: FFmpeg command not found

**Solution:** Install FFmpeg (see [Creating Animations](#creating-animations) section).

### Problem: GIF file size is too large

**Solutions:**
```bash
# 1. Reduce frame rate
ffmpeg -framerate 5 -pattern_type glob -i 'frames/*.png' animation.gif

# 2. Scale down
ffmpeg -framerate 10 -pattern_type glob -i 'frames/*.png' \
       -vf "scale=400:-1" animation_small.gif

# 3. Use palette optimization
ffmpeg -framerate 10 -pattern_type glob -i 'frames/*.png' \
       -vf "fps=10,scale=600:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
       animation_optimized.gif

# 4. Or use MP4 instead (much smaller)
ffmpeg -framerate 10 -pattern_type glob -i 'frames/*.png' \
       -c:v libx264 -pix_fmt yuv420p animation.mp4
```

---

## Running the Examples

The library includes two complete examples:

```bash
# Simple example - basic usage
cargo run --example simple_visualization

# Comprehensive demo - multiple functions with all features
cargo run --example visualization_demo
```

**Output locations:**
- Simple: `output/simple_*.png`
- Demo: `output/rastrigin/`, `output/sphere/`, `output/rosenbrock/`

---

## Summary

The PSO visualization module provides everything needed to analyze and present optimization results:

✅ **Convergence curves** for any dimensionality
✅ **2D landscape plots** with particle positions
✅ **Animation frames** for dynamic visualization
✅ **Flexible configuration** for custom outputs
✅ **Easy FFmpeg integration** for GIF/MP4 creation

For more examples, see the `examples/` directory in the repository.

Happy optimizing! 🚀
