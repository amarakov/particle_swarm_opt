//! Visualization Module for Particle Swarm Optimization
//!
//! This module provides visualization capabilities for PSO optimization runs:
//! - Convergence curves showing fitness improvement over iterations
//! - 2D contour plots of the objective function landscape with particle positions
//! - Animation frame generation for visualizing swarm movement
//!
//! Uses the `plotters` crate for high-quality static plot generation.

use crate::OptimizationHistory;
use plotters::prelude::*;
use std::error::Error;
use std::path::Path;

/// Configuration for plot styling
#[derive(Debug, Clone)]
pub struct PlotConfig {
    /// Width of the output image in pixels
    pub width: u32,
    /// Height of the output image in pixels
    pub height: u32,
    /// DPI for high-resolution output
    pub dpi: f64,
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

/// Plots the convergence curve showing best fitness vs iteration number
///
/// Creates a line plot showing how the global best fitness improves over iterations.
/// This is essential for diagnosing optimization performance.
///
/// # Arguments
///
/// * `history` - The optimization history containing fitness values for each iteration
/// * `output_path` - Path where the PNG image will be saved
/// * `title` - Optional title for the plot
/// * `config` - Optional plot configuration (uses default if None)
///
/// # Returns
///
/// Result indicating success or error
///
/// # Example
///
/// ```no_run
/// use particle_swarm_opt::visualization::plot_convergence;
/// # use particle_swarm_opt::{Swarm, Hyperparameters};
///
/// # let fitness_fn = |x: &[f64]| x.iter().map(|v| v*v).sum();
/// # let hyperparams = Hyperparameters::new(30, 2, -10.0, 10.0);
/// # let mut swarm = Swarm::new(hyperparams, &fitness_fn);
/// # let history = swarm.optimize(&fitness_fn, 100);
/// plot_convergence(&history, "convergence.png", None, None).unwrap();
/// ```
pub fn plot_convergence<P: AsRef<Path>>(
    history: &OptimizationHistory,
    output_path: P,
    title: Option<&str>,
    config: Option<PlotConfig>,
) -> Result<(), Box<dyn Error>> {
    if history.is_empty() {
        return Err("Cannot plot empty history".into());
    }

    let config = config.unwrap_or_default();
    let root = BitMapBackend::new(output_path.as_ref(), (config.width, config.height))
        .into_drawing_area();
    root.fill(&WHITE)?;

    // Collect data points
    let data: Vec<(usize, f64)> = history
        .iterations
        .iter()
        .map(|iter| (iter.iteration, iter.global_best_fitness))
        .collect();

    let max_iteration = history.iterations.last().unwrap().iteration;
    let min_fitness = data
        .iter()
        .map(|(_, f)| *f)
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let max_fitness = data
        .iter()
        .map(|(_, f)| *f)
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    // Add some margin to y-axis
    let y_margin = (max_fitness - min_fitness) * 0.1;
    let y_min = (min_fitness - y_margin).max(0.0); // Don't go below 0 if values are positive
    let y_max = max_fitness + y_margin;

    let plot_title = title.unwrap_or("PSO Convergence: Best Fitness vs Iteration");

    let mut chart = ChartBuilder::on(&root)
        .caption(plot_title, ("sans-serif", 40).into_font())
        .margin(10)
        .x_label_area_size(50)
        .y_label_area_size(80)
        .build_cartesian_2d(0..max_iteration, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc("Iteration")
        .y_desc("Best Fitness")
        .draw()?;

    // Draw the convergence line
    chart
        .draw_series(LineSeries::new(data.clone(), &BLUE.mix(0.8)))?
        .label("Global Best Fitness")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    // Draw points at each iteration
    chart.draw_series(data.iter().map(|(x, y)| {
        Circle::new((*x, *y), 3, BLUE.filled())
    }))?;

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

/// Plots a 2D contour plot of the objective function with particle positions overlaid
///
/// Creates a visualization showing:
/// - The fitness landscape as a filled contour plot
/// - Current particle positions as scatter points
/// - The global best position as a special marker
///
/// # Arguments
///
/// * `fitness_fn` - The objective function to visualize
/// * `history` - The optimization history
/// * `iteration` - Which iteration to visualize (0-based)
/// * `output_path` - Path where the PNG image will be saved
/// * `bounds` - (x_min, x_max, y_min, y_max) for the plot area
/// * `title` - Optional title for the plot
/// * `config` - Optional plot configuration
///
/// # Returns
///
/// Result indicating success or error
///
/// # Example
///
/// ```no_run
/// use particle_swarm_opt::visualization::plot_2d_landscape;
/// # use particle_swarm_opt::{Swarm, Hyperparameters};
///
/// # let fitness_fn = |x: &[f64]| x.iter().map(|v| v*v).sum();
/// # let hyperparams = Hyperparameters::new(30, 2, -10.0, 10.0);
/// # let mut swarm = Swarm::new(hyperparams, &fitness_fn);
/// # let history = swarm.optimize(&fitness_fn, 100);
/// let bounds = (-10.0, 10.0, -10.0, 10.0);
/// plot_2d_landscape(&fitness_fn, &history, 50, "landscape.png", bounds, None, None).unwrap();
/// ```
pub fn plot_2d_landscape<F, P: AsRef<Path>>(
    fitness_fn: F,
    history: &OptimizationHistory,
    iteration: usize,
    output_path: P,
    bounds: (f64, f64, f64, f64), // (x_min, x_max, y_min, y_max)
    title: Option<&str>,
    config: Option<PlotConfig>,
) -> Result<(), Box<dyn Error>>
where
    F: Fn(&[f64]) -> f64,
{
    if iteration >= history.len() {
        return Err(format!(
            "Iteration {} out of bounds (max: {})",
            iteration,
            history.len() - 1
        )
        .into());
    }

    let iter_data = &history.iterations[iteration];
    if iter_data.particle_positions[0].len() != 2 {
        return Err("Landscape plotting only supports 2D problems".into());
    }

    let config = config.unwrap_or_default();
    let root = BitMapBackend::new(output_path.as_ref(), (config.width, config.height))
        .into_drawing_area();
    root.fill(&WHITE)?;

    let (x_min, x_max, y_min, y_max) = bounds;

    // Generate contour data
    let resolution = 100; // Grid resolution for contour plot
    let mut z_values = Vec::new();
    let mut min_z = f64::INFINITY;
    let mut max_z = f64::NEG_INFINITY;

    for i in 0..resolution {
        let y = y_min + (y_max - y_min) * (i as f64) / (resolution as f64 - 1.0);
        for j in 0..resolution {
            let x = x_min + (x_max - x_min) * (j as f64) / (resolution as f64 - 1.0);
            let z = fitness_fn(&[x, y]);
            z_values.push(z);
            min_z = min_z.min(z);
            max_z = max_z.max(z);
        }
    }

    let default_title = format!("PSO Landscape - Iteration {}", iteration);
    let plot_title = title.unwrap_or(&default_title);

    let mut chart = ChartBuilder::on(&root)
        .caption(plot_title, ("sans-serif", 40).into_font())
        .margin(10)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc("Dimension 0")
        .y_desc("Dimension 1")
        .draw()?;

    // Draw contour as a heatmap
    for i in 0..resolution {
        for j in 0..resolution {
            let y = y_min + (y_max - y_min) * (i as f64) / (resolution as f64 - 1.0);
            let x = x_min + (x_max - x_min) * (j as f64) / (resolution as f64 - 1.0);
            let z = z_values[i * resolution + j];

            // Normalize z to [0, 1] for color mapping
            let normalized = if max_z > min_z {
                (z - min_z) / (max_z - min_z)
            } else {
                0.5
            };

            // Create color gradient from blue (low) to red (high)
            let color = RGBColor(
                (255.0 * normalized) as u8,
                50,
                (255.0 * (1.0 - normalized)) as u8,
            );

            let cell_width = (x_max - x_min) / (resolution as f64);
            let cell_height = (y_max - y_min) / (resolution as f64);

            chart.draw_series(std::iter::once(Rectangle::new(
                [(x, y), (x + cell_width, y + cell_height)],
                color.mix(0.7).filled(),
            )))?;
        }
    }

    // Draw particle positions
    let particle_series: Vec<(f64, f64)> = iter_data
        .particle_positions
        .iter()
        .map(|pos| (pos[0], pos[1]))
        .collect();

    chart.draw_series(particle_series.iter().map(|(x, y)| {
        Circle::new((*x, *y), 5, BLACK.mix(0.8).filled())
    }))?
    .label("Particles")
    .legend(|(x, y)| Circle::new((x + 10, y), 5, BLACK.filled()));

    // Draw global best position
    let global_best_pos = &iter_data.global_best_position;
    chart.draw_series(std::iter::once(Circle::new(
        (global_best_pos[0], global_best_pos[1]),
        8,
        RED.filled(),
    )))?
    .label("Global Best")
    .legend(|(x, y)| Circle::new((x + 10, y), 8, RED.filled()));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

/// Generates animation frames showing the swarm's movement over iterations
///
/// Creates a sequence of PNG images, one for each iteration, showing how the swarm
/// explores the search space. These frames can be stitched together into a GIF or video
/// using external tools like FFmpeg.
///
/// # Arguments
///
/// * `fitness_fn` - The objective function to visualize
/// * `history` - The optimization history
/// * `output_dir` - Directory where frames will be saved
/// * `bounds` - (x_min, x_max, y_min, y_max) for the plot area
/// * `config` - Optional plot configuration
///
/// # Returns
///
/// Result indicating success or the first error encountered
///
/// # Example
///
/// ```no_run
/// use particle_swarm_opt::visualization::generate_animation_frames;
/// # use particle_swarm_opt::{Swarm, Hyperparameters};
///
/// # let fitness_fn = |x: &[f64]| x.iter().map(|v| v*v).sum();
/// # let hyperparams = Hyperparameters::new(30, 2, -10.0, 10.0);
/// # let mut swarm = Swarm::new(hyperparams, &fitness_fn);
/// # let history = swarm.optimize(&fitness_fn, 100);
/// let bounds = (-10.0, 10.0, -10.0, 10.0);
/// generate_animation_frames(&fitness_fn, &history, "frames", bounds, None).unwrap();
/// ```
///
/// After generating frames, you can create a GIF using FFmpeg:
/// ```bash
/// ffmpeg -framerate 10 -pattern_type glob -i 'frames/frame_*.png' -vf "scale=800:-1" animation.gif
/// ```
pub fn generate_animation_frames<F, P: AsRef<Path>>(
    fitness_fn: F,
    history: &OptimizationHistory,
    output_dir: P,
    bounds: (f64, f64, f64, f64),
    config: Option<PlotConfig>,
) -> Result<(), Box<dyn Error>>
where
    F: Fn(&[f64]) -> f64,
{
    // Create output directory if it doesn't exist
    std::fs::create_dir_all(&output_dir)?;

    println!(
        "Generating {} animation frames...",
        history.len()
    );

    for iteration in 0..history.len() {
        let frame_path = output_dir
            .as_ref()
            .join(format!("frame_{:04}.png", iteration));

        let title = format!("PSO Animation - Iteration {}", iteration);
        plot_2d_landscape(
            &fitness_fn,
            history,
            iteration,
            &frame_path,
            bounds,
            Some(&title),
            config.clone(),
        )?;

        // Progress indicator
        if (iteration + 1) % 10 == 0 || iteration == history.len() - 1 {
            println!(
                "  Generated {}/{} frames",
                iteration + 1,
                history.len()
            );
        }
    }

    println!("✓ All frames generated successfully!");
    println!("\nTo create a GIF, run:");
    println!(
        "  ffmpeg -framerate 10 -pattern_type glob -i '{}/*.png' -vf \"scale=800:-1\" animation.gif",
        output_dir.as_ref().display()
    );

    Ok(())
}

/// Convenience function to generate all common visualizations
///
/// Creates:
/// 1. Convergence plot
/// 2. Initial state landscape (iteration 0)
/// 3. Final state landscape (last iteration)
/// 4. Optionally, all animation frames
///
/// # Arguments
///
/// * `fitness_fn` - The objective function (only used for 2D landscapes)
/// * `history` - The optimization history
/// * `output_dir` - Directory where all visualizations will be saved
/// * `bounds` - (x_min, x_max, y_min, y_max) for 2D landscapes (None to skip landscape plots)
/// * `generate_animation` - Whether to generate animation frames
/// * `config` - Optional plot configuration
///
/// # Returns
///
/// Result indicating success or error
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
{
    let output_dir = output_dir.as_ref();
    std::fs::create_dir_all(output_dir)?;

    println!("\nGenerating visualizations...");

    // 1. Convergence plot
    println!("  Creating convergence plot...");
    let convergence_path = output_dir.join("convergence.png");
    plot_convergence(history, &convergence_path, None, config.clone())?;
    println!("    ✓ Saved to {}", convergence_path.display());

    // 2. Landscape plots (only for 2D problems)
    if let Some(bounds) = bounds {
        // Check if problem is 2D
        if !history.is_empty() && history.iterations[0].particle_positions[0].len() == 2 {
            println!("  Creating initial landscape plot...");
            let initial_path = output_dir.join("landscape_initial.png");
            plot_2d_landscape(
                &fitness_fn,
                history,
                0,
                &initial_path,
                bounds,
                Some("PSO Landscape - Initial State (Iteration 0)"),
                config.clone(),
            )?;
            println!("    ✓ Saved to {}", initial_path.display());

            println!("  Creating final landscape plot...");
            let final_path = output_dir.join("landscape_final.png");
            let final_iteration = history.len() - 1;
            plot_2d_landscape(
                &fitness_fn,
                history,
                final_iteration,
                &final_path,
                bounds,
                Some(&format!(
                    "PSO Landscape - Final State (Iteration {})",
                    final_iteration
                )),
                config.clone(),
            )?;
            println!("    ✓ Saved to {}", final_path.display());

            // 3. Animation frames
            if generate_animation {
                println!("  Generating animation frames...");
                let frames_dir = output_dir.join("animation_frames");
                generate_animation_frames(&fitness_fn, history, &frames_dir, bounds, config)?;
            }
        } else {
            println!("  Skipping landscape plots (not a 2D problem)");
        }
    }

    println!("\n✓ All visualizations generated successfully!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Hyperparameters, Swarm};

    #[test]
    fn test_plot_convergence() {
        let fitness_fn = |position: &[f64]| -> f64 { position.iter().map(|x| x * x).sum() };

        let hyperparams = Hyperparameters::new(20, 2, -10.0, 10.0);
        let mut swarm = Swarm::new(hyperparams, &fitness_fn);
        let history = swarm.optimize(&fitness_fn, 20);

        let temp_path = "/tmp/test_convergence.png";
        let result = plot_convergence(&history, temp_path, None, None);

        assert!(result.is_ok());
        assert!(std::path::Path::new(temp_path).exists());

        // Cleanup
        let _ = std::fs::remove_file(temp_path);
    }

    #[test]
    fn test_plot_2d_landscape() {
        let fitness_fn = |position: &[f64]| -> f64 { position.iter().map(|x| x * x).sum() };

        let hyperparams = Hyperparameters::new(20, 2, -5.0, 5.0);
        let mut swarm = Swarm::new(hyperparams, &fitness_fn);
        let history = swarm.optimize(&fitness_fn, 10);

        let temp_path = "/tmp/test_landscape.png";
        let bounds = (-5.0, 5.0, -5.0, 5.0);
        let result = plot_2d_landscape(&fitness_fn, &history, 0, temp_path, bounds, None, None);

        assert!(result.is_ok());
        assert!(std::path::Path::new(temp_path).exists());

        // Cleanup
        let _ = std::fs::remove_file(temp_path);
    }

    #[test]
    fn test_plot_2d_landscape_invalid_iteration() {
        let fitness_fn = |position: &[f64]| -> f64 { position.iter().map(|x| x * x).sum() };

        let hyperparams = Hyperparameters::new(10, 2, -5.0, 5.0);
        let mut swarm = Swarm::new(hyperparams, &fitness_fn);
        let history = swarm.optimize(&fitness_fn, 5);

        let temp_path = "/tmp/test_landscape_invalid.png";
        let bounds = (-5.0, 5.0, -5.0, 5.0);
        let result = plot_2d_landscape(&fitness_fn, &history, 10, temp_path, bounds, None, None);

        assert!(result.is_err());
    }

    #[test]
    fn test_empty_history_convergence() {
        let history = OptimizationHistory::new();
        let temp_path = "/tmp/test_empty_convergence.png";
        let result = plot_convergence(&history, temp_path, None, None);

        assert!(result.is_err());
    }
}
