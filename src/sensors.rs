use crate::creature::SensorType;
use cgmath::prelude::*;

pub struct VisionSensor {
    pub range: f32,
    pub angle_of_view: f32,
    pub ray_count: usize,
    pub resolution: f32,
    pub color_vision: bool,
}

pub struct HearingSensor {
    pub range: f32,
    pub sensitivity: f32,
    pub frequency_range: (f32, f32),
    pub directionality: f32,
}

pub struct TouchSensor {
    pub sensitivity: f32,
    pub contact_radius: f32,
    pub receptor_count: usize,
}

pub struct SmellSensor {
    pub range: f32,
    pub sensitivity: f32,
    pub discrimination: f32,
}

pub struct SensorData {
    pub vision: Option<VisionData>,
    pub hearing: Option<HearingData>,
    pub touch: Option<TouchData>,
    pub smell: Option<SmellData>,
}

pub struct VisionData {
    pub rays: Vec<RayHit>,
}

#[derive(Clone)]
pub struct RayHit {
    pub distance: f32,
    pub object_type: ObjectType,
    pub color: [f32; 3],
    pub size: f32,
}

pub struct HearingData {
    pub sounds: Vec<Sound>,
}

pub struct Sound {
    pub distance: f32,
    pub direction: f32,
    pub volume: f32,
    pub sound_type: SoundType,
    pub frequency: f32,
}

pub struct TouchData {
    pub contacts: Vec<Contact>,
}

pub struct Contact {
    pub pressure: f32,
    pub object_type: ObjectType,
    pub direction: f32,
    pub texture: f32,
}

pub struct SmellData {
    pub smells: Vec<Smell>,
}

pub struct Smell {
    pub concentration: f32,
    pub direction: f32,
    pub smell_type: SmellType,
}

#[derive(Clone, Copy)]
pub enum ObjectType {
    Food,
    Creature,
    Obstacle,
    Boundary,
}

#[derive(Clone, Copy)]
pub enum SoundType {
    Movement,
    Reproduction,
    Collision,
    Eating,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SmellType {
    Food,
    SameSpecies,
    DifferentSpecies,
}

impl VisionSensor {
    pub fn sense(&self, position: cgmath::Point3<f32>, angle: f32) -> VisionData {
        // Это заглушка - реальная реализация требует доступа к другим объектам
        // В реальной реализации здесь будет raycasting для обнаружения объектов
        VisionData {
            rays: vec![RayHit {
                distance: self.range,
                object_type: ObjectType::Boundary,
                color: [0.5, 0.5, 0.5],
                size: 0.0,
            }; self.ray_count],
        }
    }
    
    pub fn sense_with_objects(
        &self,
        position: cgmath::Point3<f32>,
        angle: f32,
        foods: &[(cgmath::Point3<f32>, bool)],
        creatures: &[(cgmath::Point3<f32>, u32)],
        arena_size: (f32, f32),
    ) -> VisionData {
        let mut rays = Vec::new();
        let angle_step = self.angle_of_view / self.ray_count as f32;
        
        for i in 0..self.ray_count {
            let ray_angle = angle - self.angle_of_view / 2.0 + (i as f32 * angle_step);
            let direction = cgmath::Vector3::new(ray_angle.cos(), ray_angle.sin(), 0.0);
            
            let mut closest_hit = RayHit {
                distance: self.range,
                object_type: ObjectType::Boundary,
                color: [0.5, 0.5, 0.5],
                size: 0.0,
            };
            
            // Проверяем столкновение с границами
            let mut t = f32::MAX;
            if direction.x > 0.0 {
                t = t.min((arena_size.0 - position.x) / direction.x);
            } else if direction.x < 0.0 {
                t = t.min(-position.x / direction.x);
            }
            if direction.y > 0.0 {
                t = t.min((arena_size.1 - position.y) / direction.y);
            } else if direction.y < 0.0 {
                t = t.min(-position.y / direction.y);
            }
            
            if t < self.range && t < closest_hit.distance {
                closest_hit.distance = t;
                closest_hit.object_type = ObjectType::Boundary;
            }
            
            // Проверяем столкновение с едой
            for (food_pos, is_eaten) in foods {
                if *is_eaten {
                    continue;
                }
                let to_food = *food_pos - position;
                let dist_to_center = (to_food.x * to_food.x + to_food.y * to_food.y).sqrt();
                if dist_to_center < 10.0 && dist_to_center > 0.0 {
                    let to_food_norm = cgmath::Vector3::new(to_food.x / dist_to_center, to_food.y / dist_to_center, to_food.z / dist_to_center);
                    let dot = direction.x * to_food_norm.x + direction.y * to_food_norm.y + direction.z * to_food_norm.z;
                    if dot > 0.9 {
                        let hit_dist = dist_to_center - 5.0;
                        if hit_dist < closest_hit.distance && hit_dist > 0.0 {
                            closest_hit.distance = hit_dist;
                            closest_hit.object_type = ObjectType::Food;
                            closest_hit.color = [0.0, 1.0, 0.0];
                            closest_hit.size = 5.0;
                        }
                    }
                }
            }
            
            // Проверяем столкновение с существами
            for (creature_pos, _species_id) in creatures {
                let to_creature = *creature_pos - position;
                let dist_to_center = (to_creature.x * to_creature.x + to_creature.y * to_creature.y).sqrt();
                if dist_to_center < 30.0 && dist_to_center > 0.0 {
                    let to_creature_norm = cgmath::Vector3::new(to_creature.x / dist_to_center, to_creature.y / dist_to_center, to_creature.z / dist_to_center);
                    let dot = direction.x * to_creature_norm.x + direction.y * to_creature_norm.y + direction.z * to_creature_norm.z;
                    if dot > 0.9 {
                        let hit_dist = dist_to_center - 15.0;
                        if hit_dist < closest_hit.distance && hit_dist > 0.0 {
                            closest_hit.distance = hit_dist;
                            closest_hit.object_type = ObjectType::Creature;
                            closest_hit.color = [1.0, 0.0, 0.0];
                            closest_hit.size = 15.0;
                        }
                    }
                }
            }
            
            rays.push(closest_hit);
        }
        
        VisionData { rays }
    }
}

impl HearingSensor {
    pub fn sense(&self, position: cgmath::Point3<f32>) -> HearingData {
        HearingData {
            sounds: vec![],
        }
    }
    
    pub fn sense_with_sounds(
        &self,
        position: cgmath::Point3<f32>,
        sounds: &[(cgmath::Point3<f32>, f32, SoundType)],
    ) -> HearingData {
        let mut detected_sounds = Vec::new();
        
        for (sound_pos, volume, sound_type) in sounds {
            let to_sound = *sound_pos - position;
            let distance = (to_sound.x * to_sound.x + to_sound.y * to_sound.y + to_sound.z * to_sound.z).sqrt();
            
            if distance <= self.range {
                let attenuation = 1.0 / (1.0 + distance / 50.0);
                let perceived_volume = volume * attenuation * self.sensitivity;
                
                if perceived_volume > 0.01 {
                    let direction = to_sound.y.atan2(to_sound.x);
                    detected_sounds.push(Sound {
                        distance,
                        direction,
                        volume: perceived_volume,
                        sound_type: *sound_type,
                        frequency: match sound_type {
                            SoundType::Movement => 100.0,
                            SoundType::Reproduction => 200.0,
                            SoundType::Collision => 500.0,
                            SoundType::Eating => 300.0,
                        },
                    });
                }
            }
        }
        
        HearingData {
            sounds: detected_sounds,
        }
    }
}

impl TouchSensor {
    pub fn sense(&self, _position: cgmath::Point3<f32>) -> TouchData {
        TouchData {
            contacts: vec![],
        }
    }
}

impl SmellSensor {
    pub fn sense(&self, position: cgmath::Point3<f32>) -> SmellData {
        SmellData {
            smells: vec![],
        }
    }
    
    pub fn sense_with_sources(
        &self,
        position: cgmath::Point3<f32>,
        smell_sources: &[(cgmath::Point3<f32>, f32, SmellType)],
    ) -> SmellData {
        let mut detected_smells = Vec::new();
        
        for (smell_pos, intensity, smell_type) in smell_sources {
            let to_smell = *smell_pos - position;
            let distance = (to_smell.x * to_smell.x + to_smell.y * to_smell.y + to_smell.z * to_smell.z).sqrt();
            
            if distance <= self.range {
                // Диффузия запаха: концентрация уменьшается с расстоянием
                let concentration = intensity * self.sensitivity / (1.0 + distance * distance / 1000.0);
                
                if concentration > 0.01 {
                    let direction = to_smell.y.atan2(to_smell.x);
                    detected_smells.push(Smell {
                        concentration,
                        direction,
                        smell_type: *smell_type,
                    });
                }
            }
        }
        
        SmellData {
            smells: detected_smells,
        }
    }
}

