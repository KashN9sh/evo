mod neural;
mod rl;
mod genetic;

// pub use neural::NeuralNetwork; // Используется внутри модулей
pub use rl::RLAgent; // Для будущего использования RL
pub use genetic::*;

use bevy::prelude::*;

pub struct LearningPlugin;

impl Plugin for LearningPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GeneticAlgorithm>()
            .add_systems(Update, (
                apply_genetic_control,
                update_fitness,
                evolve_population,
                // update_rl_agent, // Отключено, так как RL агент не используется
            ));
    }
}

