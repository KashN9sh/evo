use cgmath::Point2;

pub struct Food {
    pub position: Point2<f32>,
    pub energy_value: f32,
    pub is_eaten: bool,
}

impl Food {
    pub fn new(position: Point2<f32>, energy_value: f32) -> Self {
        Self {
            position,
            energy_value,
            is_eaten: false,
        }
    }
}

pub struct FoodSystem {
    pub foods: Vec<Food>,
    pub spawn_rate: f32,
    pub spawn_timer: f32,
}

impl FoodSystem {
    pub fn new() -> Self {
        Self {
            foods: vec![],
            spawn_rate: 0.17, // Генерируем еду каждые ~0.17 секунды (в 3 раза чаще)
            spawn_timer: 0.0,
        }
    }

    pub fn update(&mut self, dt: f32, arena_size: (f32, f32), rng: &mut impl rand::Rng) {
        self.spawn_timer += dt;
        
        // Ограничиваем максимальное количество еды на арене
        const MAX_FOOD: usize = 240; // В 3 раза больше
        
        if self.spawn_timer >= self.spawn_rate && self.foods.len() < MAX_FOOD {
            self.spawn_timer = 0.0;
            self.spawn_food(arena_size, rng);
        }

        self.foods.retain(|f| !f.is_eaten);
    }

    fn spawn_food(&mut self, arena_size: (f32, f32), rng: &mut impl rand::Rng) {
        // Генерируем 12 кусков еды за раз (в 3 раза больше)
        for _ in 0..12 {
            let x = rng.gen_range(10.0..(arena_size.0 - 10.0));
            let y = rng.gen_range(10.0..(arena_size.1 - 10.0));
            self.foods.push(Food::new(
                Point2::new(x, y),
                rng.gen_range(30.0..60.0),
            ));
        }
    }
}

