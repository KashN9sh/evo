use bevy::prelude::*;
use rand::Rng;
use crate::creature::{Creature, Muscle};
use crate::learning::neural::NeuralNetwork;
use crate::simulation::Target;

#[derive(Component)]
pub struct RLAgent {
    pub network: NeuralNetwork,
    pub state_size: usize,
    pub action_size: usize,
    pub learning_rate: f32,
    pub discount_factor: f32,
    pub epsilon: f32, // Для epsilon-greedy
    pub total_reward: f32,
    pub episode_steps: u32,
}

impl RLAgent {
    // Для будущего использования RL
    #[allow(dead_code)]
    pub fn new(state_size: usize, action_size: usize, hidden_layers: &[usize]) -> Self {
        let mut layer_sizes = vec![state_size];
        layer_sizes.extend_from_slice(hidden_layers);
        layer_sizes.push(action_size);

        Self {
            network: NeuralNetwork::new(&layer_sizes),
            state_size,
            action_size,
            learning_rate: 0.01,
            discount_factor: 0.99,
            epsilon: 0.1,
            total_reward: 0.0,
            episode_steps: 0,
        }
    }

    pub fn get_action(&mut self, state: &[f32]) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        
        // Epsilon-greedy: случайное действие с вероятностью epsilon
        if rng.gen_range(0.0..1.0) < self.epsilon {
            (0..self.action_size)
                .map(|_| rng.gen_range(0.0..1.0))
                .collect()
        } else {
            self.network.forward(state)
        }
    }

    // Для будущего использования RL
    #[allow(dead_code)]
    pub fn compute_reward(
        creature_pos: Vec3,
        target_pos: Vec3,
        previous_distance: f32,
    ) -> f32 {
        let current_distance = creature_pos.distance(target_pos);
        let distance_reward = (previous_distance - current_distance) * 10.0;
        let target_reward = if current_distance < 0.5 {
            100.0 // Достигли цели
        } else {
            0.0
        };
        
        distance_reward + target_reward
    }
}

// Для будущего использования RL
#[allow(dead_code)]
pub fn update_rl_agent(
    mut agents: Query<(&mut RLAgent, &Creature)>,
    creatures: Query<&Transform, (With<Creature>, Without<Target>)>,
    targets: Query<&Transform, With<Target>>,
    mut muscles: Query<&mut Muscle>,
    _time: Res<Time>,
) {
    for (mut agent, creature) in agents.iter_mut() {
        // Получаем состояние существа
        let Ok(creature_transform) = creatures.get(creature.bones[0]) else { continue };
        let Ok(target_transform) = targets.get_single() else { continue };

        // Состояние: позиция относительно цели, ориентация
        let relative_pos = creature_transform.translation - target_transform.translation;
        let mut state = vec![
            relative_pos.x,
            relative_pos.y,
            relative_pos.z,
            creature_transform.rotation.x,
            creature_transform.rotation.y,
            creature_transform.rotation.z,
            creature_transform.rotation.w,
        ];

        // Добавляем информацию о текущих активациях мышц
        for &muscle_entity in &creature.muscles {
            if let Ok(muscle) = muscles.get(muscle_entity) {
                state.push(muscle.activation);
            }
        }

        // Получаем действия от сети
        let actions = agent.get_action(&state);

        // Применяем активации к мышцам
        for (i, &muscle_entity) in creature.muscles.iter().enumerate() {
            if let Ok(mut muscle) = muscles.get_mut(muscle_entity) {
                if i < actions.len() {
                    muscle.activation = actions[i].clamp(0.0, 1.0);
                }
            }
        }
    }
}

