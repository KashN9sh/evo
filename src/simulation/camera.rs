use bevy::prelude::*;
use bevy::input::mouse::MouseMotion;
use crate::creature::Creature;

#[derive(Component)]
pub struct OrbitCamera {
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub target: Vec3,
    pub sensitivity: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            distance: 10.0,
            yaw: 0.0,
            pitch: 0.5,
            target: Vec3::ZERO,
            sensitivity: 0.01,
        }
    }
}


pub fn spawn_orbit_camera(mut commands: Commands) {
    // Начальная позиция камеры - смотрит на место, где будет существо (0, 2, 0)
    let initial_target = Vec3::new(0.0, 2.0, 0.0);
    let initial_distance: f32 = 15.0;
    let initial_yaw: f32 = 0.0;
    let initial_pitch: f32 = 0.3; // Немного сверху
    
    // Вычисляем начальную позицию камеры
    let x = initial_distance * initial_pitch.cos() * initial_yaw.sin();
    let y = initial_distance * initial_pitch.sin();
    let z = initial_distance * initial_pitch.cos() * initial_yaw.cos();
    let initial_pos = initial_target + Vec3::new(x, y, z);
    
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_translation(initial_pos).looking_at(initial_target, Vec3::Y),
            ..default()
        },
        OrbitCamera {
            distance: initial_distance,
            yaw: initial_yaw,
            pitch: initial_pitch,
            target: initial_target,
            sensitivity: 0.01,
        },
    ));
}

pub fn orbit_camera_mouse(
    mut camera_query: Query<&mut OrbitCamera>,
    mut mouse_motion_events: EventReader<MouseMotion>,
    mouse_buttons: Res<Input<MouseButton>>,
) {
    if !mouse_buttons.pressed(MouseButton::Right) {
        return;
    }

    let mut total_delta = Vec2::ZERO;
    for event in mouse_motion_events.read() {
        total_delta += event.delta;
    }

    if total_delta.length_squared() > 0.0 {
        for mut camera in camera_query.iter_mut() {
            camera.yaw -= total_delta.x * camera.sensitivity;
            camera.pitch -= total_delta.y * camera.sensitivity;
            
            // Ограничиваем pitch, чтобы камера не переворачивалась
            camera.pitch = camera.pitch.clamp(-1.5, 1.5);
        }
    }
}

pub fn orbit_camera_scroll(
    mut camera_query: Query<&mut OrbitCamera>,
    mut scroll_events: EventReader<bevy::input::mouse::MouseWheel>,
) {
    for event in scroll_events.read() {
        for mut camera in camera_query.iter_mut() {
            camera.distance -= event.y * 0.5;
            camera.distance = camera.distance.clamp(2.0, 50.0);
        }
    }
}

pub fn update_orbit_camera(
    mut camera_query: Query<(&mut Transform, &OrbitCamera)>,
) {
    for (mut transform, camera) in camera_query.iter_mut() {
        // Вычисляем позицию камеры на основе углов и расстояния (сферические координаты)
        let x = camera.distance * camera.pitch.cos() * camera.yaw.sin();
        let y = camera.distance * camera.pitch.sin();
        let z = camera.distance * camera.pitch.cos() * camera.yaw.cos();
        
        let camera_pos = camera.target + Vec3::new(x, y, z);
        transform.translation = camera_pos;
        transform.look_at(camera.target, Vec3::Y);
    }
}

pub fn follow_creature(
    mut camera_query: Query<&mut OrbitCamera>,
    creatures: Query<&Creature>,
    bones: Query<(&Transform, &crate::creature::Bone), Without<OrbitCamera>>,
) {
    // Находим единственное существо через его кости
    for creature in creatures.iter() {
        // Пытаемся найти туловище (torso) - первую кость, или используем центр масс
        let mut creature_pos = None;
        
        // Сначала ищем туловище
        for &bone_entity in &creature.bones {
            if let Ok((bone_transform, bone)) = bones.get(bone_entity) {
                if bone.id == "torso" {
                    creature_pos = Some(bone_transform.translation);
                    break;
                }
            }
        }
        
        // Если туловище не найдено, используем центр масс всех костей
        if creature_pos.is_none() {
            let mut total_pos = Vec3::ZERO;
            let mut bone_count = 0;
            
            for &bone_entity in &creature.bones {
                if let Ok((bone_transform, _)) = bones.get(bone_entity) {
                    total_pos += bone_transform.translation;
                    bone_count += 1;
                }
            }
            
            if bone_count > 0 {
                creature_pos = Some(total_pos / bone_count as f32);
            }
        }
        
        if let Some(pos) = creature_pos {
            for mut camera in camera_query.iter_mut() {
                // Плавно обновляем цель камеры (быстрее для лучшего следования)
                camera.target = camera.target.lerp(pos, 0.2);
            }
            
            // Выходим после первого найденного существа
            break;
        }
    }
}

