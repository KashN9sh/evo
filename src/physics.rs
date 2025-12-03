use cgmath::Point3;

pub struct PhysicsWorld {
    pub arena_size: (f32, f32),
}

impl PhysicsWorld {
    pub fn new(width: f32, height: f32) -> Self {
        Self {
            arena_size: (width, height),
        }
    }

    pub fn check_boundary_collision(&self, position: &mut Point3<f32>, radius: f32) -> bool {
        let mut collided = false;
        
        if position.x - radius < 0.0 {
            position.x = radius;
            collided = true;
        }
        if position.x + radius > self.arena_size.0 {
            position.x = self.arena_size.0 - radius;
            collided = true;
        }
        if position.y - radius < 0.0 {
            position.y = radius;
            collided = true;
        }
        if position.y + radius > self.arena_size.1 {
            position.y = self.arena_size.1 - radius;
            collided = true;
        }
        // Z координата ограничена от -10 до 10 (для 3D пространства)
        if position.z < -10.0 {
            position.z = -10.0;
            collided = true;
        }
        if position.z > 10.0 {
            position.z = 10.0;
            collided = true;
        }

        collided
    }

    pub fn check_collision(
        &self,
        pos1: &Point3<f32>,
        radius1: f32,
        pos2: &Point3<f32>,
        radius2: f32,
    ) -> bool {
        let dx = pos1.x - pos2.x;
        let dy = pos1.y - pos2.y;
        let dz = pos1.z - pos2.z;
        let distance = (dx * dx + dy * dy + dz * dz).sqrt();
        distance < radius1 + radius2
    }
}
