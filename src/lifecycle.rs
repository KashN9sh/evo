use crate::creature::Creature;
use crate::food::Food;

pub struct LifecycleSystem {
    pub min_reproduction_age: f32,
    pub reproduction_energy_threshold: f32,
    pub reproduction_energy_cost: f32,
    pub max_age: Option<f32>,
}

impl LifecycleSystem {
    pub fn new() -> Self {
        Self {
            min_reproduction_age: 2.0, // Увеличиваем минимальный возраст для размножения
            reproduction_energy_threshold: 50.0, // Увеличиваем порог энергии для размножения
            reproduction_energy_cost: 25.0, // Увеличиваем стоимость размножения
            max_age: Some(30.0), // Уменьшаем максимальный возраст для более быстрой смерти
        }
    }

    pub fn update_age(&self, creature: &mut Creature, dt: f32) {
        if creature.is_alive {
            creature.age += dt;
            
            // Старение увеличивает метаболизм (больше затрат энергии)
            // Это делается через увеличение базового метаболизма
            let age_factor = 1.0 + (creature.age / 30.0) * 0.3; // К 30 годам метаболизм увеличивается на 30%
            let original_metabolism = creature.genome.metabolism.base_metabolism / (1.0 + ((creature.age - dt) / 30.0) * 0.3);
            creature.genome.metabolism.base_metabolism = original_metabolism * age_factor;
            
            if let Some(max_age) = self.max_age {
                if creature.age >= max_age {
                    creature.is_alive = false;
                }
            }
        }
    }

    pub fn can_reproduce(&self, creature: &Creature) -> bool {
        creature.is_alive
            && creature.age >= self.min_reproduction_age
            && creature.energy >= self.reproduction_energy_threshold
    }

    pub fn try_eat(&self, creature: &mut Creature, food: &mut Food, distance: f32) -> bool {
        if food.is_eaten || distance > 30.0 {
            return false;
        }

        food.is_eaten = true;
        let metabolism = crate::metabolism::Metabolism::new(
            creature.genome.metabolism.base_metabolism,
            creature.genome.metabolism.energy_conversion_efficiency,
            creature.genome.metabolism.digestion_efficiency,
        );
        metabolism.consume_food(creature, food.energy_value);
        true
    }
}

