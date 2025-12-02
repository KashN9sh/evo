use crate::creature::{Creature, Muscle};

pub struct Biomechanics {
    pub friction_coefficient: f32,
    pub gravity: f32,
}

#[derive(Clone)]
pub struct Joint {
    pub angle: f32,
    pub min_angle: f32,
    pub max_angle: f32,
    pub stiffness: f32,
    pub damping: f32,
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
        let torso_mass = creature.genome.body_parts.torso.size * 10.0;
        let legs_mass: f32 = creature.genome.body_parts.legs.iter()
            .map(|leg| leg.segments.iter().map(|s| s.length * s.width * 5.0).sum::<f32>())
            .sum();
        let arms_mass: f32 = creature.genome.body_parts.arms.iter()
            .map(|arm| arm.segments.iter().map(|s| s.length * s.width * 5.0).sum::<f32>())
            .sum();
        
        torso_mass + legs_mass + arms_mass
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
    
    // Детальная биомеханика: суставы и рычаги
    pub fn calculate_joint_constraints(&self, creature: &Creature) -> Vec<Joint> {
        let mut joints = Vec::new();
        
        // Создаем суставы для ног
        for leg in &creature.genome.body_parts.legs {
            for (i, _segment) in leg.segments.iter().enumerate() {
                if i > 0 {
                    // Сустав между сегментами
                    let min_angle = -std::f32::consts::PI / 3.0; // -60 градусов
                    let max_angle = std::f32::consts::PI / 3.0;  // +60 градусов
                    joints.push(Joint {
                        angle: 0.0,
                        min_angle,
                        max_angle,
                        stiffness: 10.0,
                        damping: 0.5,
                    });
                }
            }
        }
        
        // Создаем суставы для рук
        for arm in &creature.genome.body_parts.arms {
            for (i, _segment) in arm.segments.iter().enumerate() {
                if i > 0 {
                    let min_angle = -std::f32::consts::PI / 2.0; // -90 градусов
                    let max_angle = std::f32::consts::PI / 2.0;  // +90 градусов
                    joints.push(Joint {
                        angle: 0.0,
                        min_angle,
                        max_angle,
                        stiffness: 8.0,
                        damping: 0.4,
                    });
                }
            }
        }
        
        joints
    }
    
    pub fn calculate_lever_mechanical_advantage(&self, creature: &Creature) -> f32 {
        let mut total_advantage = 1.0;
        let mut lever_count = 0;
        
        // Рассчитываем механическое преимущество для ног
        for leg in &creature.genome.body_parts.legs {
            if leg.segments.len() >= 2 {
                let total_length: f32 = leg.segments.iter().map(|s| s.length).sum();
                let fulcrum_pos = 0.3; // Сустав примерно на 30% от начала
                let mechanical_advantage = fulcrum_pos / (1.0 - fulcrum_pos);
                total_advantage *= mechanical_advantage;
                lever_count += 1;
            }
        }
        
        // Рассчитываем для рук
        for arm in &creature.genome.body_parts.arms {
            if arm.segments.len() >= 2 {
                let total_length: f32 = arm.segments.iter().map(|s| s.length).sum();
                let fulcrum_pos = 0.25; // Сустав на 25% от начала
                let mechanical_advantage = fulcrum_pos / (1.0 - fulcrum_pos);
                total_advantage *= mechanical_advantage;
                lever_count += 1;
            }
        }
        
        if lever_count > 0 {
            (total_advantage as f32).powf(1.0 / lever_count as f32) // Среднее геометрическое
        } else {
            1.0
        }
    }
    
    pub fn apply_joint_constraints(&self, joints: &mut [Joint], target_angles: &[f32], dt: f32) {
        for (joint, &target_angle) in joints.iter_mut().zip(target_angles.iter()) {
            // Ограничиваем целевой угол пределами сустава
            let clamped_target = target_angle.max(joint.min_angle).min(joint.max_angle);
            
            // Пружинная модель для движения сустава
            let angle_error = clamped_target - joint.angle;
            let torque = angle_error * joint.stiffness;
            let angular_velocity = torque - joint.angle * joint.damping;
            
            joint.angle += angular_velocity * dt;
            
            // Ограничиваем угол пределами
            joint.angle = joint.angle.max(joint.min_angle).min(joint.max_angle);
        }
    }
    
    pub fn calculate_effective_force(&self, creature: &Creature, muscle_force: f32) -> f32 {
        let mechanical_advantage = self.calculate_lever_mechanical_advantage(creature);
        muscle_force * mechanical_advantage
    }
}
