use bevy::prelude::*;
use bevy_rapier3d::prelude::*;
use std::fs;

mod creature;
mod learning;
mod simulation;

use creature::{CreaturePlugin, CreatureConfig, create_creature_from_config};
use learning::{LearningPlugin, GeneticIndividual};
use simulation::SimulationPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(RapierPhysicsPlugin::<NoUserData>::default())
        .add_plugins(RapierDebugRenderPlugin::default())
        .add_plugins(CreaturePlugin)
        .add_plugins(LearningPlugin)
        .add_plugins(SimulationPlugin)
        .add_systems(Startup, (setup, setup_gravity))
        .run();
}

fn setup_gravity(mut rapier_config: ResMut<RapierConfiguration>) {
    // Устанавливаем гравитацию (по умолчанию -9.81 по Y, но убедимся что она включена)
    rapier_config.gravity = Vec3::new(0.0, -9.81, 0.0);
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Освещение
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            illuminance: 3000.0,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.5, -0.5, 0.0)),
        ..default()
    });

    // Камера будет создана в SimulationPlugin

    // Загружаем конфигурацию существа
    let config_path = "config/creature.json";
    let config_str = fs::read_to_string(config_path)
        .unwrap_or_else(|_| {
            // Создаем простую конфигурацию по умолчанию
            create_default_config()
        });
    
    let config: CreatureConfig = serde_json::from_str(&config_str)
        .unwrap_or_else(|e| {
            eprintln!("Failed to parse config: {:?}, using default", e);
            // Если не удалось загрузить, используем дефолтную
            serde_json::from_str(&create_default_config()).unwrap()
        });

    println!("Loaded creature config: {} bones, {} joints, {} muscles", 
             config.bones.len(), config.joints.len(), config.muscles.len());

    // Создаем популяцию существ для генетического алгоритма
    let population_size = 10; // Начинаем с небольшой популяции для тестирования
    let target_pos = Vec3::new(10.0, 1.0, 0.0); // Позиция цели
    let spawn_distance = 8.0; // Расстояние от цели, на котором спавнятся существа
    let spawn_height = 2.0; // Высота спавна
    
    for i in 0..population_size {
        // Располагаем существ по кругу вокруг цели на одинаковом расстоянии
        let angle = (i as f32 / population_size as f32) * std::f32::consts::TAU; // Угол в радианах
        let x = target_pos.x + angle.cos() * spawn_distance;
        let z = target_pos.z + angle.sin() * spawn_distance;
        
        let creature_entity = create_creature_from_config(
            &mut commands, 
            config.clone(), 
            Vec3::new(x, spawn_height, z),
            &mut meshes,
            &mut materials,
        );
        
        // Создаем нейронную сеть: вход (позиция + ориентация + скорость + расстояние + углы суставов + активации мышц), скрытые слои, выход (активации мышц)
        // Вход: 3 (позиция) + 4 (ориентация) + 3 (скорость) + 1 (расстояние) + 15 (углы суставов, 5 костей * 3 угла) + 4 (активации мышц) = 30
        // Выход: 4 (активации мышц)
        commands.entity(creature_entity).insert(GeneticIndividual::new(&[30, 24, 16, 4]));
    }
    
    println!("Created {} creatures for genetic algorithm, all at distance {} from target", 
             population_size, spawn_distance);
}

fn create_default_config() -> String {
    r#"{
        "bones": [
            {
                "id": "torso",
                "length": 1.0,
                "mass": 10.0,
                "radius": 0.15,
                "position": [0.0, 0.0, 0.0],
                "rotation": [0.0, 0.0, 0.0]
            },
            {
                "id": "upper_arm_l",
                "length": 0.6,
                "mass": 2.0,
                "radius": 0.08,
                "position": [-0.3, 0.5, 0.0],
                "rotation": [0.0, 0.0, 0.0]
            },
            {
                "id": "upper_arm_r",
                "length": 0.6,
                "mass": 2.0,
                "radius": 0.08,
                "position": [0.3, 0.5, 0.0],
                "rotation": [0.0, 0.0, 0.0]
            },
            {
                "id": "thigh_l",
                "length": 0.7,
                "mass": 3.0,
                "radius": 0.1,
                "position": [-0.2, -0.5, 0.0],
                "rotation": [0.0, 0.0, 0.0]
            },
            {
                "id": "thigh_r",
                "length": 0.7,
                "mass": 3.0,
                "radius": 0.1,
                "position": [0.2, -0.5, 0.0],
                "rotation": [0.0, 0.0, 0.0]
            }
        ],
        "joints": [
            {
                "id": "shoulder_l",
                "bone_a": "torso",
                "bone_b": "upper_arm_l",
                "anchor_a": [-0.3, 0.5, 0.0],
                "anchor_b": [0.0, 0.3, 0.0],
                "axis": [0.0, 0.0, 1.0],
                "limits": null
            },
            {
                "id": "shoulder_r",
                "bone_a": "torso",
                "bone_b": "upper_arm_r",
                "anchor_a": [0.3, 0.5, 0.0],
                "anchor_b": [0.0, 0.3, 0.0],
                "axis": [0.0, 0.0, 1.0],
                "limits": null
            },
            {
                "id": "hip_l",
                "bone_a": "torso",
                "bone_b": "thigh_l",
                "anchor_a": [-0.2, -0.5, 0.0],
                "anchor_b": [0.0, 0.35, 0.0],
                "axis": [1.0, 0.0, 0.0],
                "limits": null
            },
            {
                "id": "hip_r",
                "bone_a": "torso",
                "bone_b": "thigh_r",
                "anchor_a": [0.2, -0.5, 0.0],
                "anchor_b": [0.0, 0.35, 0.0],
                "axis": [1.0, 0.0, 0.0],
                "limits": null
            }
        ],
        "muscles": [
            {
                "id": "shoulder_l_flex",
                "bone_a": "torso",
                "attachment_a": [-0.3, 0.3, 0.0],
                "bone_b": "upper_arm_l",
                "attachment_b": [0.0, 0.0, 0.0],
                "max_force": 50.0,
                "rest_length": 0.4
            },
            {
                "id": "shoulder_r_flex",
                "bone_a": "torso",
                "attachment_a": [0.3, 0.3, 0.0],
                "bone_b": "upper_arm_r",
                "attachment_b": [0.0, 0.0, 0.0],
                "max_force": 50.0,
                "rest_length": 0.4
            },
            {
                "id": "hip_l_flex",
                "bone_a": "torso",
                "attachment_a": [-0.2, -0.3, 0.0],
                "bone_b": "thigh_l",
                "attachment_b": [0.0, 0.0, 0.0],
                "max_force": 80.0,
                "rest_length": 0.5
            },
            {
                "id": "hip_r_flex",
                "bone_a": "torso",
                "attachment_a": [0.2, -0.3, 0.0],
                "bone_b": "thigh_r",
                "attachment_b": [0.0, 0.0, 0.0],
                "max_force": 80.0,
                "rest_length": 0.5
            }
        ]
    }"#.to_string()
}
