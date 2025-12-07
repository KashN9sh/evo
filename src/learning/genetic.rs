use bevy::prelude::*;
use bevy_rapier3d::prelude::*;
use rand::Rng;
use crate::creature::{Creature, Muscle, Bone};
use crate::learning::neural::NeuralNetwork;
use crate::simulation::Target;

#[derive(Component)]
pub struct GeneticIndividual {
    pub network: NeuralNetwork,
    pub fitness: f32,
    pub generation: u32,
    pub layer_sizes: Vec<usize>,
}

#[derive(Resource)]
pub struct GeneticAlgorithm {
    pub population_size: usize,
    pub mutation_rate: f32,
    pub mutation_strength: f32,
    pub crossover_rate: f32,
    pub elite_count: usize,
    pub current_generation: u32,
}

impl Default for GeneticAlgorithm {
    fn default() -> Self {
        Self {
            population_size: 10, // Уменьшено для начальной популяции
            mutation_rate: 0.15, // Увеличено с 0.1
            mutation_strength: 0.5, // Увеличено с 0.2
            crossover_rate: 0.7,
            elite_count: 2, // Уменьшено для меньшей популяции
            current_generation: 0,
        }
    }
}

impl GeneticIndividual {
    pub fn new(layer_sizes: &[usize]) -> Self {
        Self {
            network: NeuralNetwork::new(layer_sizes),
            fitness: 0.0,
            generation: 0,
            layer_sizes: layer_sizes.to_vec(),
        }
    }

    // Функция для будущего использования при работе с популяцией
    #[allow(dead_code)]
    pub fn from_parents(parent1: &Self, parent2: &Self, mutation_rate: f32, mutation_strength: f32) -> Self {
        let weights1 = parent1.network.get_weights();
        let weights2 = parent2.network.get_weights();
        
        assert_eq!(weights1.len(), weights2.len());
        
        let mut rng = rand::thread_rng();
        let mut child_weights = Vec::new();
        
        // Кроссовер: равномерный
        for i in 0..weights1.len() {
            if rng.gen::<f32>() < 0.5 {
                child_weights.push(weights1[i]);
            } else {
                child_weights.push(weights2[i]);
            }
        }
        
        // Мутация
        for weight in &mut child_weights {
            if rng.gen::<f32>() < mutation_rate {
                *weight += rng.gen_range(-mutation_strength..mutation_strength);
            }
        }
        
        let child = Self {
            network: NeuralNetwork::from_weights(&parent1.layer_sizes, &child_weights),
            fitness: 0.0,
            generation: parent1.generation + 1,
            layer_sizes: parent1.layer_sizes.clone(),
        };
        
        child
    }
}

pub fn apply_genetic_control(
    mut individuals: Query<(&mut GeneticIndividual, &Creature)>,
    bones: Query<(&Transform, &RapierRigidBodyHandle), (With<Bone>, Without<Target>)>,
    targets: Query<&Transform, With<Target>>,
    mut muscles: Query<&mut Muscle>,
    mut rapier_context: ResMut<RapierContext>,
) {
    for (individual, creature) in individuals.iter_mut() {
        // Получаем центр масс существа (средняя позиция всех костей)
        let mut creature_pos = Vec3::ZERO;
        let mut bone_count = 0;
        let mut creature_vel = Vec3::ZERO;
        
        for &bone_entity in &creature.bones {
            if let Ok((bone_transform, rb_handle)) = bones.get(bone_entity) {
                creature_pos += bone_transform.translation;
                bone_count += 1;
                
                // Получаем скорость кости
                if let Some(rb) = rapier_context.bodies.get(rb_handle.0) {
                    let vel_rapier = rb.linvel();
                    creature_vel += Vec3::new(vel_rapier.x, vel_rapier.y, vel_rapier.z);
                }
            }
        }
        
        if bone_count == 0 {
            continue;
        }
        
        creature_pos /= bone_count as f32;
        creature_vel /= bone_count as f32;
        
        let Ok(target_transform) = targets.get_single() else { continue };

        // Состояние: нормализованная позиция относительно цели, скорость, ориентация
        let relative_pos = creature_pos - target_transform.translation;
        let distance = relative_pos.length();
        
        // Нормализуем входные данные для лучшей работы сети
        let max_distance = 50.0; // Максимальное ожидаемое расстояние
        let normalized_distance = (distance / max_distance).min(1.0);
        let normalized_pos = relative_pos.normalize_or_zero() * normalized_distance;
        
        // Вычисляем среднюю ориентацию существа (из костей)
        let mut avg_rotation = Quat::IDENTITY;
        let mut rotation_count = 0;
        for &bone_entity in &creature.bones {
            if let Ok((bone_transform, _)) = bones.get(bone_entity) {
                avg_rotation = avg_rotation.slerp(bone_transform.rotation, 1.0 / (rotation_count + 1) as f32);
                rotation_count += 1;
            }
        }
        
        // Вычисляем углы суставов (относительная ориентация между костями)
        let mut joint_angles = Vec::new();
        // Для каждого сустава вычисляем угол между костями
        // Упрощенный подход: используем относительную ориентацию костей
        for &bone_entity in &creature.bones {
            if let Ok((bone_transform, _)) = bones.get(bone_entity) {
                // Используем углы Эйлера от ориентации кости
                let (roll, pitch, yaw) = bone_transform.rotation.to_euler(EulerRot::XYZ);
                joint_angles.push(roll);
                joint_angles.push(pitch);
                joint_angles.push(yaw);
            }
        }
        // Нормализуем углы суставов (приводим к [-1, 1])
        let normalized_joint_angles: Vec<f32> = joint_angles.iter()
            .map(|&angle| (angle / std::f32::consts::PI).tanh())
            .collect();
        
        let mut state = vec![
            normalized_pos.x,
            normalized_pos.y,
            normalized_pos.z,
            // Ориентация (кватернион, нормализованный)
            avg_rotation.x,
            avg_rotation.y,
            avg_rotation.z,
            avg_rotation.w,
            // Нормализованная скорость
            (creature_vel.x / 10.0).tanh(), // tanh для ограничения
            (creature_vel.y / 10.0).tanh(),
            (creature_vel.z / 10.0).tanh(),
            // Расстояние до цели (нормализованное)
            normalized_distance,
        ];
        
        // Добавляем углы суставов (нормализованные)
        state.extend_from_slice(&normalized_joint_angles);

        // Добавляем информацию о текущих активациях мышц
        for &muscle_entity in &creature.muscles {
            if let Ok(muscle) = muscles.get(muscle_entity) {
                state.push(muscle.activation);
            }
        }

        // Получаем действия от нейронной сети
        let actions = individual.network.forward(&state);

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

pub fn update_fitness(
    mut individuals: Query<&mut GeneticIndividual>,
    creatures: Query<&Creature>,
    bones: Query<(&Transform, &RapierRigidBodyHandle), (With<Bone>, Without<Target>)>,
    muscles: Query<&Muscle>,
    targets: Query<&Transform, With<Target>>,
    rapier_context: Res<RapierContext>,
) {
    let Ok(target_transform) = targets.get_single() else { return };

    // Находим первое существо (так как у нас только одно)
    let creature = if let Some(creature) = creatures.iter().next() {
        creature
    } else {
        return;
    };
    
    // Вычисляем центр масс существа
    let mut creature_pos = Vec3::ZERO;
    let mut bone_count = 0;
    let mut creature_vel = Vec3::ZERO;
    
    // Находим все кости существа
    for &bone_entity in &creature.bones {
        if let Ok((bone_transform, rb_handle)) = bones.get(bone_entity) {
            creature_pos += bone_transform.translation;
            bone_count += 1;
            
            // Получаем скорость
            if let Some(rb) = rapier_context.bodies.get(rb_handle.0) {
                let vel_rapier = rb.linvel();
                creature_vel += Vec3::new(vel_rapier.x, vel_rapier.y, vel_rapier.z);
            }
        }
    }
    
    if bone_count == 0 {
        return;
    }
    
    creature_pos /= bone_count as f32;
    creature_vel /= bone_count as f32;
    let speed = creature_vel.length();
    
    // Обновляем фитнес для всех индивидов (у нас один)
    for mut individual in individuals.iter_mut() {
        // Вычисляем фитнес на основе расстояния до цели
        let distance = creature_pos.distance(target_transform.translation);
        
        // Улучшенная функция фитнеса:
        // - Бонус за близость к цели (экспоненциальный)
        // - Бонус за движение к цели (скорость в направлении цели)
        // - Штраф за стояние на месте
        let direction_to_target = (target_transform.translation - creature_pos).normalize_or_zero();
        let velocity_toward_target = creature_vel.dot(direction_to_target).max(0.0);
        
        // Вычисляем штраф за энергию (сумма активаций всех мышц)
        let mut total_energy = 0.0;
        for &muscle_entity in &creature.muscles {
            if let Ok(muscle) = muscles.get(muscle_entity) {
                total_energy += muscle.activation; // Суммируем активации всех мышц
            }
        }
        let energy_penalty = -total_energy * 2.0; // Штраф за использование энергии
        
        let fitness = if distance < 0.5 {
            // Достигли цели - большой бонус
            10000.0 - distance * 100.0
        } else {
            // Базовый фитнес: обратно пропорционален расстоянию
            let base_fitness = 1000.0 / (1.0 + distance);
            // Бонус за движение к цели
            let movement_bonus = velocity_toward_target * 50.0;
            // Штраф за стояние на месте
            let speed_penalty = if speed < 0.1 { -10.0 } else { 0.0 };
            
            base_fitness + movement_bonus + speed_penalty + energy_penalty
        };
        
        individual.fitness = fitness.max(0.0); // Фитнес не может быть отрицательным
    }
}

pub fn evolve_population(
    mut ga: ResMut<GeneticAlgorithm>,
    mut individuals: Query<(&mut GeneticIndividual, &Creature)>,
    time: Res<Time>,
) {
    // Эволюция происходит каждые 5 секунд (было 10)
    let elapsed = time.elapsed_seconds();
    let evolution_interval = 5.0;
    
    // Проверяем, прошло ли достаточно времени с последней эволюции
    if elapsed < (ga.current_generation as f32 + 1.0) * evolution_interval {
        return;
    }
    
    println!("=== Evolution: Generation {} ===", ga.current_generation + 1);

    // Собираем всех особей
    let mut individuals_vec: Vec<_> = individuals.iter_mut().collect();
    
    if individuals_vec.is_empty() {
        return;
    }

    // Сортируем по фитнесу (лучшие первыми)
    individuals_vec.sort_by(|a, b| {
        b.0.fitness.partial_cmp(&a.0.fitness).unwrap()
    });

    // Выводим статистику
    let best_fitness = individuals_vec[0].0.fitness;
    let avg_fitness = individuals_vec.iter().map(|(ind, _)| ind.fitness).sum::<f32>() / individuals_vec.len() as f32;
    println!("Best fitness: {:.2}, Avg fitness: {:.2}", best_fitness, avg_fitness);
    
    // Если только одна особь - делаем мутацию
    if individuals_vec.len() == 1 {
        let (ref mut individual, _) = individuals_vec[0];
        let weights = individual.network.get_weights();
        let mut new_weights = weights.clone();
        
        // Увеличиваем силу мутации для одной особи
        let mutation_strength = ga.mutation_strength * 1.5;
        
        // Мутируем веса
        let mut rng = rand::thread_rng();
        let mut mutations = 0;
        for weight in &mut new_weights {
            if rng.gen::<f32>() < ga.mutation_rate {
                *weight += rng.gen_range(-mutation_strength..mutation_strength);
                mutations += 1;
            }
        }
        
        println!("Mutated {} weights (strength: {:.2})", mutations, mutation_strength);
        
        individual.network.set_weights(&new_weights);
        individual.fitness = 0.0; // Сбрасываем фитнес для нового поколения
        individual.generation += 1;
        ga.current_generation += 1;
        return;
    }

    // Если несколько особей - делаем полную эволюцию с кроссовером
    let elite_count = ga.elite_count.min(individuals_vec.len());
    
    // Сначала копируем данные элитных особей для использования в кроссовере
    let elite_individuals: Vec<GeneticIndividual> = individuals_vec[0..elite_count]
        .iter()
        .map(|(ind, _)| GeneticIndividual {
            network: NeuralNetwork::from_weights(&ind.layer_sizes, &ind.network.get_weights()),
            fitness: ind.fitness,
            generation: ind.generation,
            layer_sizes: ind.layer_sizes.clone(),
        })
        .collect();
    
    // Элитные особи сохраняются без изменений
    // Остальные заменяются потомками от лучших особей
    
    let mut rng = rand::thread_rng();
    
    // Создаем новых особей через кроссовер и мутацию
    for i in elite_count..individuals_vec.len() {
        let (ref mut individual, _) = individuals_vec[i];
        
        // Выбираем двух родителей из лучших особей
        let parent1_idx = rng.gen_range(0..elite_count);
        let parent2_idx = rng.gen_range(0..elite_count);
        
        let parent1 = &elite_individuals[parent1_idx];
        let parent2 = &elite_individuals[parent2_idx];
        
        // Создаем потомка через кроссовер
        let child = GeneticIndividual::from_parents(
            parent1,
            parent2,
            ga.mutation_rate,
            ga.mutation_strength,
        );
        
        // Заменяем текущую особь потомком
        individual.network = child.network;
        individual.fitness = 0.0;
        individual.generation = child.generation;
    }

    // Создаем новое поколение
    ga.current_generation += 1;
    println!("Evolved population: {} elite, {} new offspring", elite_count, individuals_vec.len() - elite_count);
}

// Функция для будущего использования при работе с популяцией
#[allow(dead_code)]
pub fn select_parents(individuals: &[&GeneticIndividual]) -> (usize, usize) {
    // Турнирная селекция
    let mut rng = rand::thread_rng();
    let tournament_size = 3;
    
    let mut tournament1 = Vec::new();
    let mut tournament2 = Vec::new();
    
    for _ in 0..tournament_size {
        tournament1.push(rng.gen_range(0..individuals.len()));
        tournament2.push(rng.gen_range(0..individuals.len()));
    }
    
    let parent1_idx = *tournament1.iter().max_by(|&&a, &&b| {
        individuals[a].fitness.partial_cmp(&individuals[b].fitness).unwrap()
    }).unwrap();
    
    let parent2_idx = *tournament2.iter().max_by(|&&a, &&b| {
        individuals[a].fitness.partial_cmp(&individuals[b].fitness).unwrap()
    }).unwrap();
    
    (parent1_idx, parent2_idx)
}

