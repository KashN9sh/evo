use bevy::prelude::*;
use bevy_rapier3d::prelude::*;
use crate::creature::structure::*;

pub fn create_creature_from_config(
    commands: &mut Commands,
    config: CreatureConfig,
    position: Vec3,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
) -> Entity {
    let creature_entity = commands.spawn(Creature {
        id: "creature_1".to_string(),
        bones: vec![],
        joints: vec![],
        muscles: vec![],
    }).id();

    let mut bone_entities = std::collections::HashMap::new();

    // Создаем кости
    for bone_config in &config.bones {
        let bone_pos = Vec3::from_array(bone_config.position) + position;
        
        // Создаем меш цилиндра для визуализации кости (капсула = цилиндр с закругленными концами)
        // Используем цилиндр как приближение
        use bevy::render::mesh::shape::Cylinder;
        let bone_mesh = meshes.add(
            Mesh::from(Cylinder {
                radius: bone_config.radius,
                height: bone_config.length,
                resolution: 16,
                segments: 1,
            })
        );
        // Разные цвета для разных костей для отладки
        let bone_color = match bone_config.id.as_str() {
            "torso" => Color::rgb(1.0, 0.0, 0.0), // Красный
            "upper_arm_l" => Color::rgb(0.0, 1.0, 0.0), // Зеленый
            "upper_arm_r" => Color::rgb(0.0, 0.0, 1.0), // Синий
            "thigh_l" => Color::rgb(1.0, 1.0, 0.0), // Желтый
            "thigh_r" => Color::rgb(1.0, 0.0, 1.0), // Пурпурный
            _ => Color::rgb(0.8, 0.7, 0.6), // Бежевый по умолчанию
        };
        
        let bone_material = materials.add(StandardMaterial {
            base_color: bone_color,
            perceptual_roughness: 0.7,
            metallic: 0.1,
            ..default()
        });
        
        let bone_entity = commands.spawn((
            Bone {
                id: bone_config.id.clone(),
                config: bone_config.clone(),
            },
            RigidBody::Dynamic,
            Collider::capsule(
                Vec3::Y * bone_config.length / 2.0,
                Vec3::Y * -bone_config.length / 2.0,
                bone_config.radius,
            ),
            PbrBundle {
                mesh: bone_mesh,
                material: bone_material,
                transform: Transform::from_translation(bone_pos)
                .with_rotation(Quat::from_euler(
                    EulerRot::XYZ,
                    bone_config.rotation[0].to_radians(),
                    bone_config.rotation[1].to_radians(),
                    bone_config.rotation[2].to_radians(),
                )),
                ..default()
            },
            ColliderMassProperties::Mass(bone_config.mass),
            // Отключаем сон на старте, чтобы кости не засыпали сразу
            Sleeping::disabled(),
            // Добавляем небольшое демпфирование для стабильности
            Damping {
                linear_damping: 0.5,
                angular_damping: 0.5,
            },
        )).id();

        bone_entities.insert(bone_config.id.clone(), bone_entity);
        
        println!("Created bone '{}' at position {:?}", bone_config.id, bone_pos);
        
        // НЕ добавляем как дочерний элемент, чтобы трансформации были независимыми
        // commands.entity(creature_entity).insert_children(0, &[bone_entity]);
    }

    // Обновляем список костей в существе
    let creature_bones: Vec<Entity> = bone_entities.values().copied().collect();
    
    // Создаем суставы
    let mut joint_entities = vec![];
    for joint_config in &config.joints {
        if let (Some(&_bone_a_entity), Some(&_bone_b_entity)) = (
            bone_entities.get(&joint_config.bone_a),
            bone_entities.get(&joint_config.bone_b),
        ) {
            let joint_entity = commands.spawn(Joint {
                id: joint_config.id.clone(),
                config: joint_config.clone(),
                handle: None,
            }).id();

            joint_entities.push(joint_entity);
            // Суставы не нужно добавлять как дочерние элементы
            // commands.entity(creature_entity).insert_children(0, &[joint_entity]);
        }
    }

    // Создаем мышцы
    let mut muscle_entities = vec![];
    for muscle_config in &config.muscles {
        if let (Some(&bone_a_entity), Some(&bone_b_entity)) = (
            bone_entities.get(&muscle_config.bone_a),
            bone_entities.get(&muscle_config.bone_b),
        ) {
            let muscle_entity = commands.spawn(Muscle::new(
                muscle_config.id.clone(),
                muscle_config.clone(),
                bone_a_entity,
                bone_b_entity,
            )).id();

            muscle_entities.push(muscle_entity);
            // Мышцы не нужно добавлять как дочерние элементы
            // commands.entity(creature_entity).insert_children(0, &[muscle_entity]);
        }
    }

    // Обновляем существо со всеми компонентами
    commands.entity(creature_entity).insert(Creature {
        id: "creature_1".to_string(),
        bones: creature_bones,
        joints: joint_entities,
        muscles: muscle_entities,
    });

    creature_entity
}

pub fn setup_joints(
    mut commands: Commands,
    mut joints: Query<(Entity, &mut Joint)>,
    bones: Query<(Entity, &Bone)>,
    time: Res<Time>,
) {
    // Задержка перед созданием суставов, чтобы кости стабилизировались
    if time.elapsed_seconds() < 0.1 {
        return;
    }
    
    for (_joint_entity, mut joint) in joints.iter_mut() {
        // Пропускаем, если сустав уже создан
        if joint.handle.is_some() {
            continue;
        }

        let joint_config = &joint.config;
        
        // Находим Entity костей для этого сустава
        let mut bone_a_entity: Option<Entity> = None;
        let mut bone_b_entity: Option<Entity> = None;

        for (bone_entity, bone) in bones.iter() {
            if bone.id == joint_config.bone_a {
                bone_a_entity = Some(bone_entity);
            }
            if bone.id == joint_config.bone_b {
                bone_b_entity = Some(bone_entity);
            }
        }

        if let (Some(entity_a), Some(entity_b)) = (bone_a_entity, bone_b_entity) {
            // Вычисляем точки крепления в локальных координатах
            let anchor_a_local = Vec3::from_array(joint_config.anchor_a);
            let anchor_b_local = Vec3::from_array(joint_config.anchor_b);
            
            // Создаем revolute joint (вращательный сустав)
            let axis = Vec3::from_array(joint_config.axis).normalize();
            
            // Создаем сустав через Rapier
            let mut joint_builder = RevoluteJointBuilder::new(axis)
                .local_anchor1(anchor_a_local)
                .local_anchor2(anchor_b_local);
            
            if let Some(limits) = &joint_config.limits {
                joint_builder = joint_builder.limits([limits.min, limits.max]);
            }
            
            // Создаем сустав через команды Bevy
            // ImpulseJoint::new принимает Entity родителя и GenericJoint
            let impulse_joint = ImpulseJoint::new(entity_b, joint_builder);
            commands.entity(entity_a).insert(impulse_joint);
            
            println!("Created joint '{}' between '{}' and '{}'", 
                     joint_config.id, joint_config.bone_a, joint_config.bone_b);
            
            // Помечаем сустав как созданный
            // Используем unsafe для создания фиктивного handle для отслеживания
            // В реальности handle будет доступен через компонент ImpulseJoint на кости
            unsafe {
                use std::mem;
                let fake_handle: RapierImpulseJointHandle = mem::zeroed();
                joint.handle = Some(fake_handle);
            }
        } else {
            eprintln!("Failed to create joint '{}': bones '{}' or '{}' not found", 
                     joint_config.id, joint_config.bone_a, joint_config.bone_b);
        }
    }
}

