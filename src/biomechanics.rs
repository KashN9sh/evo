use crate::creature::{Creature, Muscle};

pub struct Biomechanics {
    pub friction_coefficient: f32,
    pub gravity: f32,
}

#[derive(Clone)]
pub struct Joint {
    pub id: usize,
    pub bone1_id: usize, // Первая кость
    pub bone2_id: usize, // Вторая кость
    pub angle: f32, // Текущий угол между костями
    pub ligament: Ligament, // Связка, ограничивающая движение
    pub position: cgmath::Point3<f32>, // Позиция сустава (точка соединения костей)
}

#[derive(Clone)]
pub struct Ligament {
    pub stiffness: f32, // Жесткость связки
    pub damping: f32, // Демпфирование
    pub min_angle: f32, // Минимальный угол сустава
    pub max_angle: f32, // Максимальный угол сустава
}

#[derive(Clone)]
pub struct Lever {
    pub length: f32,
    pub fulcrum_position: f32, // 0.0 = начало, 1.0 = конец
    pub mechanical_advantage: f32,
}

impl Biomechanics {
    pub fn new() -> Self {
        Self {
            friction_coefficient: 0.1,
            gravity: 0.0,
        }
    }

    pub fn calculate_muscle_force(&self, muscle: &Muscle, activation: f32) -> f32 {
        muscle.strength * activation * muscle.efficiency
    }

    pub fn calculate_movement_energy(&self, creature: &Creature, velocity: f32) -> f32 {
        let mass = self.calculate_mass(creature);
        let avg_efficiency = creature.genome.muscles.iter()
            .map(|m| m.efficiency)
            .sum::<f32>() / creature.genome.muscles.len() as f32;
        
        0.5 * mass * velocity * velocity / avg_efficiency.max(0.1)
    }

    pub fn calculate_mass(&self, creature: &Creature) -> f32 {
        // Масса вычисляется как сумма масс всех костей
        creature.genome.bones.iter().map(|b| b.mass).sum()
    }
    
    /// Рассчитывает момент вращения от мышц вокруг суставов
    /// Мышцы прикреплены к концевым суставам костей и изменяют угол между костями
    pub fn calculate_joint_torques_from_muscles(
        &self,
        creature: &Creature,
        muscle_activations: &[f32],
    ) -> Vec<(usize, f32)> {
        // Возвращаем список (joint_id, torque) для каждого сустава
        let mut joint_torques: std::collections::HashMap<usize, f32> = std::collections::HashMap::new();
        
        // Рассчитываем момент вращения от каждой мышцы
        for (i, activation) in muscle_activations.iter().enumerate() {
            if i < creature.genome.muscles.len() {
                let muscle = &creature.genome.muscles[i];
                let force_magnitude = self.calculate_muscle_force(muscle, *activation);
                
                // Находим кости и суставы
                if let (Some(bone1), Some(bone2)) = (
                    creature.genome.bones.get(muscle.bone1_id),
                    creature.genome.bones.get(muscle.bone2_id),
                ) {
                    // Находим концевой сустав первой кости
                    if let Some(end_joint1) = creature.genome.joints.iter().find(|j| j.id == muscle.end_joint1_id) {
                        // Находим концевой сустав второй кости
                        if let Some(end_joint2) = creature.genome.joints.iter().find(|j| j.id == muscle.end_joint2_id) {
                            // Находим сустав между костями (где изменяется угол)
                            if let Some(joint) = creature.genome.joints.iter().find(|j| j.id == muscle.joint_id) {
                                // Вычисляем плечо рычага - расстояние от концевого сустава до сустава между костями
                                // Плечо = длина кости (от сустава между костями до концевого сустава)
                                let lever_arm1 = bone1.length;
                                let lever_arm2 = bone2.length;
                                
                                // Момент вращения = сила * плечо рычага
                                // Мышца тянет концы костей, создавая момент вращения вокруг сустава между костями
                                // Момент пропорционален силе мышцы и длине костей
                                // Увеличиваем коэффициент для более сильного эффекта
                                let torque = force_magnitude * (lever_arm1 + lever_arm2) * 0.5 * 10.0; // Увеличили в 10 раз
                                
                                // Направление момента: мышца сжимается (активация > 0.5) - уменьшает угол, разжимается - увеличивает
                                // Но для движения нужно, чтобы угол изменялся циклически
                                // Используем синусоидальную зависимость от активации для создания колебаний
                                let direction = (*activation * 2.0 - 1.0); // Преобразуем 0-1 в -1..1
                                let final_torque = torque * direction;
                                
                                *joint_torques.entry(muscle.joint_id).or_insert(0.0) += final_torque;
                                
                                // Отладочная информация
                                if creature.id == 0 && muscle.joint_id < 3 {
                                    eprintln!("DEBUG TORQUE: Мышца {}: активация={:.4}, сила={:.4}, момент={:.4}", 
                                        i, activation, force_magnitude, final_torque);
                                }
                            }
                        }
                    }
                }
            }
        }
        
        joint_torques.into_iter().collect()
    }
    
    /// Рассчитывает движение на основе активации мышц и углов суставов
    /// Движение происходит за счет того, что кости отталкиваются от земли
    /// Возвращает вектор силы, который будет применен к существу
    pub fn calculate_movement_from_muscles(
        &self,
        creature: &Creature,
        creature_position: &cgmath::Point3<f32>,
        muscle_activations: &[f32],
        previous_joint_states: &[crate::creature::JointState],
        dt: f32,
    ) -> cgmath::Vector3<f32> {
        // Используем текущие состояния суставов из существа
        self.calculate_movement_from_muscles_with_states(
            creature,
            creature_position,
            muscle_activations,
            previous_joint_states,
            &creature.joint_states,
            dt,
        )
    }
    
    /// Рассчитывает движение с явно указанными текущими состояниями суставов
    pub fn calculate_movement_from_muscles_with_states(
        &self,
        creature: &Creature,
        creature_position: &cgmath::Point3<f32>,
        muscle_activations: &[f32],
        previous_joint_states: &[crate::creature::JointState],
        current_joint_states: &[crate::creature::JointState],
        dt: f32,
    ) -> cgmath::Vector3<f32> {
        let mut total_force = cgmath::Vector3::new(0.0, 0.0, 0.0);
        
        // Вычисляем реальные позиции концов всех костей в мировых координатах
        // и проверяем, какие кости касаются земли (Z = 0 или близко к 0)
        let ground_level = 0.0;
        let ground_tolerance = 0.5; // Увеличиваем толерантность для контакта с землей
        
        // Отладочная информация (только для первого существа)
        let mut debug_bone_count = 0;
        
        // Для каждой кости вычисляем позицию ее конца в мировых координатах
        for bone in &creature.genome.bones {
            // Находим сустав, где начинается кость
            let start_joint = creature.genome.joints.iter().find(|j| j.bone1_id == bone.id);
            // Позиция начала кости в мировых координатах = позиция существа + локальная позиция сустава
            let start_pos_local = if let Some(joint) = start_joint {
                joint.position
            } else {
                bone.position
            };
            let start_pos = cgmath::Point3::new(
                creature_position.x + start_pos_local.x,
                creature_position.y + start_pos_local.y,
                creature_position.z + start_pos_local.z,
            );
            
            // Вычисляем угол кости с учетом состояния суставов
            let mut bone_angle = bone.angle;
            // Если у кости есть родитель, учитываем угол сустава
            if let Some(parent_id) = bone.parent_bone_id {
                if let Some(joint) = creature.genome.joints.iter().find(|j| 
                    (j.bone1_id == parent_id && j.bone2_id == bone.id) ||
                    (j.bone2_id == parent_id && j.bone1_id == bone.id)
                ) {
                    if let Some(joint_state) = current_joint_states.iter().find(|js| js.joint_id == joint.id) {
                        bone_angle += joint_state.angle;
                    }
                }
            }
            
            // Позиция конца кости
            let bone_end_x = start_pos.x + bone_angle.cos() * bone.length;
            let bone_end_y = start_pos.y + bone_angle.sin() * bone.length;
            let bone_end_z = start_pos.z;
            
            // Проверяем, касается ли конец кости земли
            if bone_end_z <= ground_level + ground_tolerance {
                // Вычисляем угловую скорость сустава (изменение угла во времени)
                // Ищем сустав, который соединяет эту кость с родительской костью
                let angular_velocity = if let Some(parent_id) = bone.parent_bone_id {
                    // Ищем сустав между родительской костью и этой костью
                    if let Some(joint) = creature.genome.joints.iter().find(|j| 
                        (j.bone1_id == parent_id && j.bone2_id == bone.id) ||
                        (j.bone2_id == parent_id && j.bone1_id == bone.id)
                    ) {
                        // Находим предыдущее и текущее состояние этого сустава
                        if let (Some(prev_state), Some(curr_state)) = (
                            previous_joint_states.iter().find(|js| js.joint_id == joint.id),
                            current_joint_states.iter().find(|js| js.joint_id == joint.id)
                        ) {
                            let ang_vel = (curr_state.angle - prev_state.angle) / dt;
                            
                            // Отладочная информация
                            if creature.id == 0 && bone.id < 2 {
                                eprintln!("DEBUG BIOMECH ANGVEL: Кость {}: сустав {} найден, угловая скорость = {:.4} (предыдущий угол={:.4}, текущий угол={:.4})", 
                                    bone.id, joint.id, ang_vel, prev_state.angle, curr_state.angle);
                            }
                            
                            ang_vel
                        } else {
                            if creature.id == 0 && bone.id < 2 {
                                eprintln!("DEBUG BIOMECH ANGVEL: Кость {}: сустав {} найден, но состояния не найдены", bone.id, joint.id);
                            }
                            0.0
                        }
                    } else {
                        if creature.id == 0 && bone.id < 2 {
                            eprintln!("DEBUG BIOMECH ANGVEL: Кость {}: сустав между родителем {} и костью {} не найден", bone.id, parent_id, bone.id);
                        }
                        0.0
                    }
                } else {
                    // У корневой кости нет родителя, но может быть сустав, где bone1_id == bone2_id == bone.id
                    // Ищем сустав, где кость соединена сама с собой (начальный сустав)
                    if let Some(joint) = creature.genome.joints.iter().find(|j| 
                        j.bone1_id == bone.id && j.bone2_id == bone.id
                    ) {
                        if let (Some(prev_state), Some(curr_state)) = (
                            previous_joint_states.iter().find(|js| js.joint_id == joint.id),
                            current_joint_states.iter().find(|js| js.joint_id == joint.id)
                        ) {
                            (curr_state.angle - prev_state.angle) / dt
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    }
                };
                
                // Если кость движется вниз (угловая скорость отрицательная для вертикальных костей)
                // или движется горизонтально, она отталкивается от земли
                // Сила отталкивания зависит от скорости движения кости
                let bone_velocity_x = -angular_velocity * bone.length * bone_angle.sin();
                let bone_velocity_y = angular_velocity * bone.length * bone_angle.cos();
                
                // Сила отталкивания пропорциональна скорости движения кости и силе мышц
                let muscle_force_sum: f32 = muscle_activations.iter()
                    .enumerate()
                    .filter(|(i, _)| *i < creature.genome.muscles.len())
                    .map(|(i, activation)| {
                        let muscle = &creature.genome.muscles[i];
                        if muscle.bone1_id == bone.id || muscle.bone2_id == bone.id {
                            self.calculate_muscle_force(muscle, *activation)
                        } else {
                            0.0
                        }
                    })
                    .sum();
                
                // Сила отталкивания направлена вверх и вперед (противоположно движению кости)
                let push_force_scale = (bone_velocity_x * bone_velocity_x + bone_velocity_y * bone_velocity_y).sqrt() * muscle_force_sum * 0.1;
                
                // Направление силы: противоположно движению кости (отталкивание)
                if push_force_scale > 0.0 {
                    let force_dir_x = -bone_velocity_x / (bone_velocity_x * bone_velocity_x + bone_velocity_y * bone_velocity_y).sqrt().max(0.001);
                    let force_dir_y = -bone_velocity_y / (bone_velocity_x * bone_velocity_x + bone_velocity_y * bone_velocity_y).sqrt().max(0.001);
                    
                    total_force.x += force_dir_x * push_force_scale;
                    total_force.y += force_dir_y * push_force_scale;
                    
                    // Отладочная информация (только для первого существа)
                    if creature.id == 0 && bone.id < 2 {
                        eprintln!("DEBUG BIOMECH: Кость {} касается земли (Z={:.2})", bone.id, bone_end_z);
                        eprintln!("DEBUG BIOMECH: Угловая скорость: {:.4}, Сила мышц: {:.4}", angular_velocity, muscle_force_sum);
                        eprintln!("DEBUG BIOMECH: Скорость кости: ({:.4}, {:.4})", bone_velocity_x, bone_velocity_y);
                        eprintln!("DEBUG BIOMECH: Сила отталкивания: ({:.4}, {:.4})", force_dir_x * push_force_scale, force_dir_y * push_force_scale);
                    }
                } else if creature.id == 0 && bone.id < 2 {
                    eprintln!("DEBUG BIOMECH: Кость {} касается земли, но push_force_scale = 0 (угловая скорость: {:.4}, сила мышц: {:.4})", bone.id, angular_velocity, muscle_force_sum);
                }
            } else if creature.id == 0 && bone.id < 2 {
                eprintln!("DEBUG BIOMECH: Кость {} НЕ касается земли (Z={:.2}, требуется <= {:.2})", bone.id, bone_end_z, ground_level + ground_tolerance);
            }
        }
        
        // Ограничиваем максимальную силу
        let max_force = 10.0;
        let force_magnitude = (total_force.x * total_force.x + total_force.y * total_force.y).sqrt();
        if force_magnitude > max_force {
            total_force.x = (total_force.x / force_magnitude) * max_force;
            total_force.y = (total_force.y / force_magnitude) * max_force;
        }
        
        // Отладочная информация
        if creature.id == 0 {
            eprintln!("DEBUG BIOMECH: Общая сила движения: ({:.4}, {:.4}, {:.4})", total_force.x, total_force.y, total_force.z);
        }
        
        total_force
    }

    pub fn apply_friction(&self, velocity: &mut cgmath::Vector2<f32>, dt: f32) {
        let mag = (velocity.x * velocity.x + velocity.y * velocity.y).sqrt();
        if mag > 0.0 {
            let friction_x = velocity.x / mag * self.friction_coefficient * mag;
            let friction_y = velocity.y / mag * self.friction_coefficient * mag;
            velocity.x = (velocity.x - friction_x * dt).max(0.0);
            velocity.y = (velocity.y - friction_y * dt).max(0.0);
        }
    }
    
    pub fn calculate_lever_mechanical_advantage(&self, creature: &Creature) -> f32 {
        // Рассчитываем механическое преимущество на основе структуры костей и суставов
        let mut total_advantage = 1.0;
        let mut lever_count = 0;
        
        // Для каждого сустава вычисляем механическое преимущество
        for joint in &creature.genome.joints {
            if let (Some(bone1), Some(bone2)) = (
                creature.genome.bones.get(joint.bone1_id),
                creature.genome.bones.get(joint.bone2_id),
            ) {
                let total_length = bone1.length + bone2.length;
                if total_length > 0.0 {
                    // Фулькрум примерно на 30% от начала
                    let fulcrum_pos = 0.3;
                    let mechanical_advantage = fulcrum_pos / (1.0 - fulcrum_pos);
                    total_advantage *= mechanical_advantage;
                    lever_count += 1;
                }
            }
        }
        
        if lever_count > 0 {
            (total_advantage as f32).powf(1.0 / lever_count as f32) // Среднее геометрическое
        } else {
            1.0
        }
    }
    
    // Эта функция больше не используется, так как суставы теперь в геноме
    // Удалена для упрощения
    
    pub fn calculate_effective_force(&self, creature: &Creature, muscle_force: f32) -> f32 {
        let mechanical_advantage = self.calculate_lever_mechanical_advantage(creature);
        muscle_force * mechanical_advantage
    }
}
