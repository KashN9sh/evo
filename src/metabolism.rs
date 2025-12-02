use crate::creature::Creature;

pub struct Metabolism {
    pub base_metabolism_rate: f32,
    pub energy_conversion_efficiency: f32,
    pub digestion_efficiency: f32,
}

impl Metabolism {
    pub fn new(base_metabolism: f32, conversion_efficiency: f32, digestion_efficiency: f32) -> Self {
        Self {
            base_metabolism_rate: base_metabolism,
            energy_conversion_efficiency: conversion_efficiency,
            digestion_efficiency: digestion_efficiency,
        }
    }

    pub fn update(&self, creature: &mut Creature, dt: f32) {
        if !creature.is_alive {
            return;
        }

        let mut energy_cost = self.base_metabolism_rate * dt;

        for sensor in &creature.genome.sensors {
            energy_cost += sensor.maintenance_cost * dt;
        }
        
        // Ограничиваем максимальный расход энергии
        energy_cost = energy_cost.min(0.5 * dt);

        creature.energy -= energy_cost;

        if creature.energy <= 0.0 {
            creature.energy = 0.0;
            creature.is_alive = false;
        }
    }

    pub fn consume_food(&self, creature: &mut Creature, food_energy: f32) {
        let digested_energy = food_energy * self.digestion_efficiency;
        creature.energy = (creature.energy + digested_energy).min(100.0);
        
        let excess = (creature.energy + digested_energy) - 100.0;
        if excess > 0.0 {
            creature.energy_storage = (creature.energy_storage + excess * 0.5).min(50.0);
        }
    }

    pub fn calculate_movement_cost(&self, _creature: &Creature, velocity: f32, mass: f32) -> f32 {
        let kinetic_energy = 0.5 * mass * velocity * velocity;
        kinetic_energy / self.energy_conversion_efficiency
    }
}

