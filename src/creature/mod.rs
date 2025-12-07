mod structure;
mod physics;
mod control;

pub use structure::*;
pub use physics::*;
pub use control::*;

use bevy::prelude::*;

pub struct CreaturePlugin;

impl Plugin for CreaturePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (apply_muscle_forces, setup_joints, enable_muscles_after_delay));
    }
}

// Включаем мышцы через небольшую задержку после создания существа
fn enable_muscles_after_delay(
    mut muscles: Query<&mut Muscle>,
    time: Res<Time>,
) {
    // Включаем мышцы через 0.5 секунды после старта
    if time.elapsed_seconds() > 0.5 {
        for mut muscle in muscles.iter_mut() {
            if !muscle.enabled {
                muscle.enabled = true;
            }
        }
    }
}

