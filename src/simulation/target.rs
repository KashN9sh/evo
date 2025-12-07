use bevy::prelude::*;
use bevy::render::mesh::shape::UVSphere;

#[derive(Component)]
pub struct Target {
    #[allow(dead_code)]
    pub position: Vec3, // Используется через Transform
}

pub fn spawn_target(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>, mut materials: ResMut<Assets<StandardMaterial>>) {
    // Создаем целевую точку
    let target_pos = Vec3::new(10.0, 1.0, 0.0);
    
    // Создаем визуализацию цели - большая яркая красная сфера
    commands.spawn((
        Target {
            position: target_pos,
        },
        PbrBundle {
            mesh: meshes.add(Mesh::from(UVSphere {
                radius: 1.0,
                ..default()
            })), // Увеличиваем размер до 1.0
            material: materials.add(StandardMaterial {
                base_color: Color::rgb(1.0, 0.0, 0.0), // Яркий красный
                emissive: Color::rgb(2.0, 0.0, 0.0), // Сильное свечение
                ..default()
            }),
            transform: Transform::from_translation(target_pos),
            ..default()
        },
    ));
}

