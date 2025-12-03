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
    
    /// Рассчитывает движение на основе активации мышц и углов суставов
    /// Возвращает вектор силы, который будет применен к существу
    pub fn calculate_movement_from_muscles(
        &self,
        creature: &Creature,
        muscle_activations: &[f32],
        dt: f32,
    ) -> cgmath::Vector3<f32> {
        let mut total_force = cgmath::Vector3::new(0.0, 0.0, 0.0);
        
        // Рассчитываем силу от каждой мышцы
        for (i, activation) in muscle_activations.iter().enumerate() {
            if i < creature.genome.muscles.len() {
                let muscle = &creature.genome.muscles[i];
                let force_magnitude = self.calculate_muscle_force(muscle, *activation);
                
                // Находим кости, к которым прикреплена мышца
                if let (Some(bone1), Some(bone2)) = (
                    creature.genome.bones.get(muscle.bone1_id),
                    creature.genome.bones.get(muscle.bone2_id),
                ) {
                    // Находим состояние сустава
                    if let Some(joint_state) = creature.joint_states.iter().find(|js| js.joint_id == muscle.joint_id) {
                        // Вычисляем направление силы на основе угла между костями
                        let bone1_angle = bone1.angle;
                        let bone2_angle = bone2.angle + joint_state.angle;
                        
                        // Средний угол для направления силы
                        let force_angle = (bone1_angle + bone2_angle) / 2.0;
                        
                        total_force.x += force_angle.cos() * force_magnitude;
                        total_force.y += force_angle.sin() * force_magnitude;
                        total_force.z = 0.0; // Движение в плоскости XY
                    }
                }
            }
        }
        
        // Применяем механическое преимущество рычагов
        let mechanical_advantage = self.calculate_lever_mechanical_advantage(creature);
        total_force *= mechanical_advantage;
        
        // Ограничиваем максимальную силу
        let max_force = 10.0;
        let force_magnitude = (total_force.x * total_force.x + total_force.y * total_force.y).sqrt();
        if force_magnitude > max_force {
            total_force.x = (total_force.x / force_magnitude) * max_force;
            total_force.y = (total_force.y / force_magnitude) * max_force;
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
