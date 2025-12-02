use cgmath::Point2;

pub struct PhysicsWorld {
    pub arena_size: (f32, f32),
}

impl PhysicsWorld {
    pub fn new(width: f32, height: f32) -> Self {
        Self {
            arena_size: (width, height),
        }
    }

    pub fn check_boundary_collision(&self, position: &mut Point2<f32>, radius: f32) -> bool {
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

        collided
    }

    pub fn check_collision(
        &self,
        pos1: &Point2<f32>,
        radius1: f32,
        pos2: &Point2<f32>,
        radius2: f32,
    ) -> bool {
        let dx = pos1.x - pos2.x;
        let dy = pos1.y - pos2.y;
        let distance = (dx * dx + dy * dy).sqrt();
        distance < radius1 + radius2
    }
}
