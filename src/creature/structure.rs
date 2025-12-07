use bevy::prelude::*;
use bevy_rapier3d::prelude::RapierImpulseJointHandle;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreatureConfig {
    pub bones: Vec<BoneConfig>,
    pub joints: Vec<JointConfig>,
    pub muscles: Vec<MuscleConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoneConfig {
    pub id: String,
    pub length: f32,
    pub mass: f32,
    pub radius: f32,
    pub position: [f32; 3],
    pub rotation: [f32; 3], // Euler angles in degrees
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointConfig {
    pub id: String,
    pub bone_a: String,
    pub bone_b: String,
    pub anchor_a: [f32; 3], // Local position on bone A
    pub anchor_b: [f32; 3], // Local position on bone B
    pub axis: [f32; 3], // Rotation axis in local space
    pub limits: Option<JointLimits>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointLimits {
    pub min: f32, // Minimum angle in radians
    pub max: f32, // Maximum angle in radians
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MuscleConfig {
    pub id: String,
    pub bone_a: String,
    pub attachment_a: [f32; 3], // Local position on bone A
    pub bone_b: String,
    pub attachment_b: [f32; 3], // Local position on bone B
    pub max_force: f32,
    pub rest_length: f32,
}

#[derive(Component)]
pub struct Bone {
    pub id: String,
    #[allow(dead_code)]
    pub config: BoneConfig, // Используется для отладки и будущих функций
}

#[derive(Component)]
pub struct Joint {
    #[allow(dead_code)]
    pub id: String, // Для отладки
    pub config: JointConfig,
    pub handle: Option<RapierImpulseJointHandle>,
}

#[derive(Component)]
pub struct Muscle {
    #[allow(dead_code)]
    pub id: String, // Для отладки
    pub config: MuscleConfig,
    pub activation: f32, // 0.0 to 1.0
    pub bone_a_entity: Entity,
    pub bone_b_entity: Entity,
    pub enabled: bool, // Включена ли мышца (для задержки на старте)
}

impl Muscle {
    pub fn new(id: String, config: MuscleConfig, bone_a: Entity, bone_b: Entity) -> Self {
        Self {
            id,
            config,
            activation: 0.0,
            bone_a_entity: bone_a,
            bone_b_entity: bone_b,
            enabled: false, // Отключаем на старте
        }
    }
}

#[derive(Component)]
pub struct Creature {
    #[allow(dead_code)]
    pub id: String, // Для отладки
    pub bones: Vec<Entity>,
    #[allow(dead_code)]
    pub joints: Vec<Entity>, // Для будущего использования
    pub muscles: Vec<Entity>,
}

