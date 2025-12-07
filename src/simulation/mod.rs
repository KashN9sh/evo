mod world;
mod target;
mod camera;
mod ui;

pub use world::*;
pub use target::*;
pub use ui::*;

use bevy::prelude::*;

pub struct SimulationPlugin;

impl Plugin for SimulationPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SimulationStats>()
            .add_systems(Startup, (spawn_ground, spawn_target, setup_ui, camera::spawn_orbit_camera))
            .add_systems(Update, (
                update_ui_stats,
                camera::orbit_camera_mouse,
                camera::orbit_camera_scroll,
                camera::follow_creature,
                camera::update_orbit_camera,
            ));
    }
}

