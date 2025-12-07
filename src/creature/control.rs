use bevy::prelude::*;
use bevy_rapier3d::prelude::*;
use crate::creature::structure::*;

pub fn apply_muscle_forces(
    mut muscles: Query<&mut Muscle>,
    bones: Query<(&Transform, &RapierRigidBodyHandle)>,
    mut rapier_context: ResMut<RapierContext>,
) {
    for muscle in muscles.iter_mut() {
        // Пропускаем, если мышца не включена или не активирована
        if !muscle.enabled || muscle.activation <= 0.0 {
            continue;
        }

        // Получаем трансформы костей
        let Ok((transform_a, handle_a)) = bones.get(muscle.bone_a_entity) else { continue };
        let Ok((transform_b, handle_b)) = bones.get(muscle.bone_b_entity) else { continue };

        // Вычисляем точки крепления в мировых координатах
        let attachment_a_local = Vec3::from_array(muscle.config.attachment_a);
        let attachment_b_local = Vec3::from_array(muscle.config.attachment_b);
        
        let attachment_a_world = transform_a.transform_point(attachment_a_local);
        let attachment_b_world = transform_b.transform_point(attachment_b_local);

        // Вычисляем направление и длину мышцы
        let direction = attachment_b_world - attachment_a_world;
        let current_length = direction.length();
        let rest_length = muscle.config.rest_length;

        // Вычисляем силу (пружина с активацией)
        let stretch = current_length - rest_length;
        let force_magnitude = muscle.activation * muscle.config.max_force * (1.0 + stretch / rest_length);
        
        if force_magnitude > 0.0 && current_length > 0.001 {
            let force_direction = direction / current_length;
            let force = force_direction * force_magnitude;

            // Применяем силы к костям
            use bevy_rapier3d::rapier::na::{Point3, Vector3};
            if let Some(rb_a) = rapier_context.bodies.get_mut(handle_a.0) {
                let impulse: Vector3<f32> = (force * 0.5).into();
                let point: Point3<f32> = Point3::from(attachment_a_world);
                rb_a.apply_impulse_at_point(impulse, point, true);
            }
            if let Some(rb_b) = rapier_context.bodies.get_mut(handle_b.0) {
                let impulse: Vector3<f32> = (-force * 0.5).into();
                let point: Point3<f32> = Point3::from(attachment_b_world);
                rb_b.apply_impulse_at_point(impulse, point, true);
            }
        }
    }
}

