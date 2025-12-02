use crate::creature::{Creature, Genome, SensorGene, SensorType, BodyParts, Muscle, MetabolismGenes};
use crate::neural::{self, ConnectionGene, NodeGene, NodeType, ActivationFunction};
use rand::Rng;

pub struct EvolutionSystem {
    pub compatibility_threshold: f32,
    pub mutation_rate: f32,
    pub crossover_rate: f32,
}

impl EvolutionSystem {
    pub fn new() -> Self {
        Self {
            compatibility_threshold: 1.5, // Уменьшаем порог для более частого видообразования
            mutation_rate: 0.8, // Увеличиваем частоту мутаций для более заметной эволюции
            crossover_rate: 0.7,
        }
    }

    pub fn calculate_compatibility(&self, genome1: &Genome, genome2: &Genome) -> f32 {
        // Вычисляем совместимость нейронной сети
        let mut excess = 0;
        let mut disjoint = 0;
        let mut matching = 0;
        let mut weight_diff = 0.0;

        let max_innovation1 = genome1.neural_network.connection_genes.iter()
            .map(|g| g.innovation_number)
            .max()
            .unwrap_or(0);
        let max_innovation2 = genome2.neural_network.connection_genes.iter()
            .map(|g| g.innovation_number)
            .max()
            .unwrap_or(0);
        let max_innovation = max_innovation1.max(max_innovation2);

        let mut innovation_map1: std::collections::HashMap<u64, &ConnectionGene> = 
            genome1.neural_network.connection_genes.iter()
                .map(|g| (g.innovation_number, g))
                .collect();
        let mut innovation_map2: std::collections::HashMap<u64, &ConnectionGene> = 
            genome2.neural_network.connection_genes.iter()
                .map(|g| (g.innovation_number, g))
                .collect();

        for innovation in 0..=max_innovation {
            let gene1 = innovation_map1.get(&innovation);
            let gene2 = innovation_map2.get(&innovation);

            match (gene1, gene2) {
                (Some(g1), Some(g2)) => {
                    matching += 1;
                    weight_diff += (g1.weight - g2.weight).abs();
                }
                (Some(_), None) | (None, Some(_)) => {
                    if innovation > max_innovation1.min(max_innovation2) {
                        excess += 1;
                    } else {
                        disjoint += 1;
                    }
                }
                (None, None) => {}
            }
        }

        let n = genome1.neural_network.connection_genes.len().max(
            genome2.neural_network.connection_genes.len()
        ).max(1) as f32;

        let c1 = 1.0;
        let c2 = 1.0;
        let c3 = 0.4;

        let neural_compatibility = c1 * excess as f32 / n + c2 * disjoint as f32 / n + c3 * weight_diff / matching.max(1) as f32;
        
        // Добавляем различия в частях тела
        let body_diff = {
            let size_diff = (genome1.body_parts.torso.size - genome2.body_parts.torso.size).abs();
            let shape_diff = (genome1.body_parts.torso.shape - genome2.body_parts.torso.shape).abs();
            let legs_diff = (genome1.body_parts.legs.len() as f32 - genome2.body_parts.legs.len() as f32).abs();
            let arms_diff = (genome1.body_parts.arms.len() as f32 - genome2.body_parts.arms.len() as f32).abs();
            size_diff + shape_diff * 0.5 + legs_diff * 0.3 + arms_diff * 0.2
        };
        
        // Добавляем различия в органах чувств
        let sensor_diff = {
            let mut diff = 0.0;
            let sensor_types1: std::collections::HashSet<_> = genome1.sensors.iter().map(|s| s.sensor_type).collect();
            let sensor_types2: std::collections::HashSet<_> = genome2.sensors.iter().map(|s| s.sensor_type).collect();
            
            // Различия в типах органов
            let unique1 = sensor_types1.difference(&sensor_types2).count();
            let unique2 = sensor_types2.difference(&sensor_types1).count();
            diff += (unique1 + unique2) as f32 * 0.5;
            
            // Различия в параметрах одинаковых органов
            for s1 in &genome1.sensors {
                if let Some(s2) = genome2.sensors.iter().find(|s| s.sensor_type == s1.sensor_type) {
                    diff += (s1.development - s2.development).abs() * 0.3;
                    diff += (s1.range - s2.range).abs() * 0.2;
                    diff += (s1.sensitivity - s2.sensitivity).abs() * 0.2;
                }
            }
            diff
        };
        
        // Добавляем различия в мышцах
        let muscle_diff = {
            let count_diff = (genome1.muscles.len() as f32 - genome2.muscles.len() as f32).abs();
            let mut param_diff = 0.0;
            let min_len = genome1.muscles.len().min(genome2.muscles.len());
            for i in 0..min_len {
                param_diff += (genome1.muscles[i].strength - genome2.muscles[i].strength).abs();
                param_diff += (genome1.muscles[i].speed - genome2.muscles[i].speed).abs();
                param_diff += (genome1.muscles[i].efficiency - genome2.muscles[i].efficiency).abs();
            }
            count_diff * 0.3 + param_diff * 0.1
        };
        
        // Добавляем различия в метаболизме
        let metabolism_diff = {
            (genome1.metabolism.base_metabolism - genome2.metabolism.base_metabolism).abs() * 10.0 +
            (genome1.metabolism.energy_conversion_efficiency - genome2.metabolism.energy_conversion_efficiency).abs() * 2.0 +
            (genome1.metabolism.digestion_efficiency - genome2.metabolism.digestion_efficiency).abs() * 2.0
        };
        
        // Общая совместимость: нейросеть + все остальное
        neural_compatibility + body_diff * 0.3 + sensor_diff * 0.2 + muscle_diff * 0.2 + metabolism_diff * 0.1
    }

    pub fn mutate(&self, genome: &mut Genome, rng: &mut impl Rng) {
        // Мутации нейросети
        if rng.gen::<f32>() < self.mutation_rate {
            self.mutate_weights(genome, rng);
        }
        if rng.gen::<f32>() < self.mutation_rate * 0.3 {
            self.mutate_add_connection(genome, rng);
        }
        if rng.gen::<f32>() < self.mutation_rate * 0.15 {
            self.mutate_add_node(genome, rng);
        }
        
        // Мутации органов чувств
        if rng.gen::<f32>() < self.mutation_rate * 0.4 {
            self.mutate_sensors(genome, rng);
        }
        
        // Мутации частей тела
        if rng.gen::<f32>() < self.mutation_rate * 0.5 {
            self.mutate_body_parts(genome, rng);
        }
        
        // Мутации мышц
        if rng.gen::<f32>() < self.mutation_rate * 0.4 {
            self.mutate_muscles(genome, rng);
        }
        
        // Мутации метаболизма
        if rng.gen::<f32>() < self.mutation_rate * 0.4 {
            self.mutate_metabolism(genome, rng);
        }
    }

    fn mutate_weights(&self, genome: &mut Genome, rng: &mut impl Rng) {
        for conn in &mut genome.neural_network.connection_genes {
            if rng.gen::<f32>() < 0.3 {
                conn.weight += rng.gen::<f32>() * 0.3 - 0.15;
                conn.weight = conn.weight.clamp(-5.0, 5.0);
            }
        }
    }

    fn mutate_add_connection(&self, genome: &mut Genome, rng: &mut impl Rng) {
        if genome.neural_network.node_genes.len() < 2 {
            return;
        }

        let from_idx = rng.gen_range(0..genome.neural_network.node_genes.len());
        let to_idx = rng.gen_range(0..genome.neural_network.node_genes.len());

        if from_idx == to_idx {
            return;
        }

        let from_node = genome.neural_network.node_genes[from_idx].id;
        let to_node = genome.neural_network.node_genes[to_idx].id;

        let innovation = neural::get_next_innovation();
        genome.neural_network.connection_genes.push(ConnectionGene {
            innovation_number: innovation,
            from_node,
            to_node,
            weight: rng.gen::<f32>() * 2.0 - 1.0,
            enabled: true,
        });
    }

    fn mutate_add_node(&self, genome: &mut Genome, rng: &mut impl Rng) {
        if genome.neural_network.connection_genes.is_empty() {
            return;
        }

        let conn_idx = rng.gen_range(0..genome.neural_network.connection_genes.len());
        
        let (from_node, to_node, weight) = {
            let conn = &genome.neural_network.connection_genes[conn_idx];
            if !conn.enabled {
                return;
            }
            (conn.from_node, conn.to_node, conn.weight)
        };

        genome.neural_network.connection_genes[conn_idx].enabled = false;

        let new_node_id = genome.neural_network.node_genes.len() as u64;
        genome.neural_network.node_genes.push(NodeGene {
            id: new_node_id,
            node_type: NodeType::Hidden,
            activation_function: ActivationFunction::Sigmoid,
        });

        let innovation1 = neural::get_next_innovation();
        let innovation2 = neural::get_next_innovation();

        genome.neural_network.connection_genes.push(ConnectionGene {
            innovation_number: innovation1,
            from_node,
            to_node: new_node_id,
            weight: 1.0,
            enabled: true,
        });

        genome.neural_network.connection_genes.push(ConnectionGene {
            innovation_number: innovation2,
            from_node: new_node_id,
            to_node,
            weight,
            enabled: true,
        });
    }

    fn mutate_sensors(&self, genome: &mut Genome, rng: &mut impl Rng) {
        // Изменение параметров существующих органов чувств
        if rng.gen::<f32>() < 0.5 && !genome.sensors.is_empty() {
            let idx = rng.gen_range(0..genome.sensors.len());
            let sensor = &mut genome.sensors[idx];
            
            // Мутация развития органа (может улучшаться или деградировать)
            sensor.development += rng.gen::<f32>() * 0.3 - 0.15;
            sensor.development = sensor.development.clamp(0.1, 1.0);
            
            // Мутация радиуса
            sensor.range += rng.gen::<f32>() * 0.3 - 0.15;
            sensor.range = sensor.range.max(0.1).min(2.0);
            
            // Мутация чувствительности
            sensor.sensitivity += rng.gen::<f32>() * 0.3 - 0.15;
            sensor.sensitivity = sensor.sensitivity.max(0.1).min(1.0);
        }

        // Добавление нового органа чувств (более вероятно)
        if rng.gen::<f32>() < 0.5 {
            let sensor_type = match rng.gen_range(0..4) {
                0 => SensorType::Vision,
                1 => SensorType::Hearing,
                2 => SensorType::Touch,
                _ => SensorType::Smell,
            };

            if !genome.sensors.iter().any(|s| s.sensor_type == sensor_type) {
                genome.sensors.push(SensorGene {
                    sensor_type,
                    development: 0.2 + rng.gen::<f32>() * 0.3, // Начальное развитие 0.2-0.5
                    range: 0.3 + rng.gen::<f32>() * 0.4,
                    sensitivity: 0.3 + rng.gen::<f32>() * 0.4,
                    maintenance_cost: 0.005 + rng.gen::<f32>() * 0.01,
                    active_cost: 0.002 + rng.gen::<f32>() * 0.005,
                });
            }
        }

        // Удаление органа чувств (редко, только если их много)
        if rng.gen::<f32>() < 0.15 && genome.sensors.len() > 2 {
            let idx = rng.gen_range(0..genome.sensors.len());
            genome.sensors.remove(idx);
        }
    }
    
    fn mutate_body_parts(&self, genome: &mut Genome, rng: &mut impl Rng) {
        // Мутация размера туловища
        if rng.gen::<f32>() < 0.8 {
            genome.body_parts.torso.size += rng.gen::<f32>() * 0.5 - 0.25;
            genome.body_parts.torso.size = genome.body_parts.torso.size.max(0.5).min(3.0);
        }
        
        // Мутация формы туловища
        if rng.gen::<f32>() < 0.6 {
            genome.body_parts.torso.shape += rng.gen::<f32>() * 0.3 - 0.15;
            genome.body_parts.torso.shape = genome.body_parts.torso.shape.clamp(0.0, 1.0);
        }
        
        // Мутация цвета (небольшие изменения)
        if rng.gen::<f32>() < 0.5 {
            for i in 0..3 {
                genome.body_parts.torso.color[i] += rng.gen::<f32>() * 0.2 - 0.1;
                genome.body_parts.torso.color[i] = genome.body_parts.torso.color[i].clamp(0.0, 1.0);
            }
        }
        
        // Мутация ног: добавление/удаление ноги
        if rng.gen::<f32>() < 0.4 {
            if rng.gen::<f32>() < 0.5 && genome.body_parts.legs.len() < 6 {
                // Добавить ногу
                genome.body_parts.legs.push(crate::creature::Leg {
                    segments: vec![crate::creature::Segment {
                        length: 0.3 + rng.gen::<f32>() * 0.4,
                        width: 0.05 + rng.gen::<f32>() * 0.1,
                    }],
                    position: rng.gen::<f32>() * std::f32::consts::PI * 2.0,
                });
            } else if genome.body_parts.legs.len() > 1 {
                // Удалить ногу
                let idx = rng.gen_range(0..genome.body_parts.legs.len());
                genome.body_parts.legs.remove(idx);
            }
        }
        
        // Мутация параметров ног
        if rng.gen::<f32>() < 0.7 && !genome.body_parts.legs.is_empty() {
            let leg_idx = rng.gen_range(0..genome.body_parts.legs.len());
            let leg = &mut genome.body_parts.legs[leg_idx];
            
            // Добавить/удалить сегмент
            if rng.gen::<f32>() < 0.3 {
                if rng.gen::<f32>() < 0.5 && leg.segments.len() < 4 {
                    leg.segments.push(crate::creature::Segment {
                        length: 0.3 + rng.gen::<f32>() * 0.3,
                        width: 0.05 + rng.gen::<f32>() * 0.08,
                    });
                } else if leg.segments.len() > 1 {
                    let seg_idx = rng.gen_range(0..leg.segments.len());
                    leg.segments.remove(seg_idx);
                }
            }
            
            // Изменить параметры сегментов
            for segment in &mut leg.segments {
                segment.length += rng.gen::<f32>() * 0.2 - 0.1;
                segment.length = segment.length.max(0.1).min(1.0);
                segment.width += rng.gen::<f32>() * 0.05 - 0.025;
                segment.width = segment.width.max(0.02).min(0.3);
            }
        }
        
        // Мутация рук: добавление/удаление руки
        if rng.gen::<f32>() < 0.3 {
            if rng.gen::<f32>() < 0.5 && genome.body_parts.arms.len() < 4 {
                // Добавить руку
                genome.body_parts.arms.push(crate::creature::Arm {
                    segments: vec![crate::creature::Segment {
                        length: 0.2 + rng.gen::<f32>() * 0.3,
                        width: 0.04 + rng.gen::<f32>() * 0.08,
                    }],
                    position: rng.gen::<f32>() * std::f32::consts::PI * 2.0,
                });
            } else if !genome.body_parts.arms.is_empty() {
                // Удалить руку
                let idx = rng.gen_range(0..genome.body_parts.arms.len());
                genome.body_parts.arms.remove(idx);
            }
        }
        
        // Мутация параметров рук
        if rng.gen::<f32>() < 0.6 && !genome.body_parts.arms.is_empty() {
            let arm_idx = rng.gen_range(0..genome.body_parts.arms.len());
            let arm = &mut genome.body_parts.arms[arm_idx];
            
            // Добавить/удалить сегмент
            if rng.gen::<f32>() < 0.3 {
                if rng.gen::<f32>() < 0.5 && arm.segments.len() < 3 {
                    arm.segments.push(crate::creature::Segment {
                        length: 0.2 + rng.gen::<f32>() * 0.25,
                        width: 0.04 + rng.gen::<f32>() * 0.06,
                    });
                } else if arm.segments.len() > 1 {
                    let seg_idx = rng.gen_range(0..arm.segments.len());
                    arm.segments.remove(seg_idx);
                }
            }
            
            // Изменить параметры сегментов
            for segment in &mut arm.segments {
                segment.length += rng.gen::<f32>() * 0.15 - 0.075;
                segment.length = segment.length.max(0.1).min(0.8);
                segment.width += rng.gen::<f32>() * 0.04 - 0.02;
                segment.width = segment.width.max(0.02).min(0.25);
            }
        }
    }
    
    fn mutate_muscles(&self, genome: &mut Genome, rng: &mut impl Rng) {
        // Добавить/удалить мышцу
        if rng.gen::<f32>() < 0.4 {
            if rng.gen::<f32>() < 0.5 && genome.muscles.len() < 5 {
                // Добавить мышцу
                genome.muscles.push(Muscle {
                    strength: 0.5 + rng.gen::<f32>() * 1.0,
                    speed: 0.5 + rng.gen::<f32>() * 1.0,
                    efficiency: 0.2 + rng.gen::<f32>() * 0.3,
                    endurance: 0.5 + rng.gen::<f32>() * 1.0,
                });
            } else if genome.muscles.len() > 1 {
                // Удалить мышцу
                let idx = rng.gen_range(0..genome.muscles.len());
                genome.muscles.remove(idx);
            }
        }
        
        // Изменить параметры мышц
        if rng.gen::<f32>() < 0.8 && !genome.muscles.is_empty() {
            let muscle_idx = rng.gen_range(0..genome.muscles.len());
            let muscle = &mut genome.muscles[muscle_idx];
            
            muscle.strength += rng.gen::<f32>() * 0.3 - 0.15;
            muscle.strength = muscle.strength.max(0.1).min(3.0);
            
            muscle.speed += rng.gen::<f32>() * 0.3 - 0.15;
            muscle.speed = muscle.speed.max(0.1).min(3.0);
            
            muscle.efficiency += rng.gen::<f32>() * 0.1 - 0.05;
            muscle.efficiency = muscle.efficiency.max(0.1).min(0.8);
            
            muscle.endurance += rng.gen::<f32>() * 0.3 - 0.15;
            muscle.endurance = muscle.endurance.max(0.1).min(3.0);
        }
    }
    
    fn mutate_metabolism(&self, genome: &mut Genome, rng: &mut impl Rng) {
        // Мутация базового метаболизма
        genome.metabolism.base_metabolism += rng.gen::<f32>() * 0.01 - 0.005;
        genome.metabolism.base_metabolism = genome.metabolism.base_metabolism.max(0.005).min(0.1);
        
        // Мутация эффективности преобразования энергии
        genome.metabolism.energy_conversion_efficiency += rng.gen::<f32>() * 0.1 - 0.05;
        genome.metabolism.energy_conversion_efficiency = genome.metabolism.energy_conversion_efficiency.max(0.1).min(0.9);
        
        // Мутация эффективности пищеварения
        genome.metabolism.digestion_efficiency += rng.gen::<f32>() * 0.1 - 0.05;
        genome.metabolism.digestion_efficiency = genome.metabolism.digestion_efficiency.max(0.5).min(1.0);
    }

    pub fn crossover(&self, parent1: &Genome, parent2: &Genome, rng: &mut impl Rng) -> Genome {
        let mut child = Genome::default();
        
        let mut innovation_map1: std::collections::HashMap<u64, &ConnectionGene> = 
            parent1.neural_network.connection_genes.iter()
                .map(|g| (g.innovation_number, g))
                .collect();
        let mut innovation_map2: std::collections::HashMap<u64, &ConnectionGene> = 
            parent2.neural_network.connection_genes.iter()
                .map(|g| (g.innovation_number, g))
                .collect();

        let max_innovation = parent1.neural_network.connection_genes.iter()
            .map(|g| g.innovation_number)
            .chain(parent2.neural_network.connection_genes.iter().map(|g| g.innovation_number))
            .max()
            .unwrap_or(0);

        for innovation in 0..=max_innovation {
            let gene1 = innovation_map1.get(&innovation);
            let gene2 = innovation_map2.get(&innovation);

            match (gene1, gene2) {
                (Some(g1), Some(g2)) => {
                    let chosen = if rng.gen::<f32>() < 0.5 { (*g1).clone() } else { (*g2).clone() };
                    child.neural_network.connection_genes.push(chosen);
                }
                (Some(g), None) | (None, Some(g)) => {
                    if rng.gen::<f32>() < 0.5 {
                        child.neural_network.connection_genes.push((*g).clone());
                    }
                }
                (None, None) => {}
            }
        }

        let max_node_id = parent1.neural_network.node_genes.iter()
            .map(|n| n.id)
            .chain(parent2.neural_network.node_genes.iter().map(|n| n.id))
            .max()
            .unwrap_or(0);

        for node_id in 0..=max_node_id {
            let node1 = parent1.neural_network.node_genes.iter().find(|n| n.id == node_id);
            let node2 = parent2.neural_network.node_genes.iter().find(|n| n.id == node_id);

            match (node1, node2) {
                (Some(n1), Some(n2)) => {
                    let chosen = if rng.gen::<f32>() < 0.5 { n1 } else { n2 };
                    if !child.neural_network.node_genes.iter().any(|n| n.id == chosen.id) {
                        child.neural_network.node_genes.push(chosen.clone());
                    }
                }
                (Some(n), None) | (None, Some(n)) => {
                    if rng.gen::<f32>() < 0.5 && !child.neural_network.node_genes.iter().any(|n2| n2.id == n.id) {
                        child.neural_network.node_genes.push(n.clone());
                    }
                }
                (None, None) => {}
            }
        }

        // Скрещивание частей тела - среднее между родителями
        child.body_parts = Self::crossover_body_parts(&parent1.body_parts, &parent2.body_parts, rng);
        
        // Скрещивание мышц - среднее между родителями
        child.muscles = Self::crossover_muscles(&parent1.muscles, &parent2.muscles, rng);
        
        // Скрещивание органов чувств - объединение и усреднение
        child.sensors = Self::crossover_sensors(&parent1.sensors, &parent2.sensors, rng);
        
        // Скрещивание метаболизма - среднее между родителями
        child.metabolism = Self::crossover_metabolism(&parent1.metabolism, &parent2.metabolism, rng);

        child
    }

    fn crossover_body_parts(parent1: &BodyParts, parent2: &BodyParts, rng: &mut impl Rng) -> BodyParts {
        // Среднее между родителями
        BodyParts {
            torso: crate::creature::Torso {
                size: (parent1.torso.size + parent2.torso.size) / 2.0,
                shape: (parent1.torso.shape + parent2.torso.shape) / 2.0,
                color: [
                    (parent1.torso.color[0] + parent2.torso.color[0]) / 2.0,
                    (parent1.torso.color[1] + parent2.torso.color[1]) / 2.0,
                    (parent1.torso.color[2] + parent2.torso.color[2]) / 2.0,
                ],
            },
            legs: if rng.gen::<f32>() < 0.5 {
                parent1.legs.clone()
            } else {
                parent2.legs.clone()
            },
            arms: if rng.gen::<f32>() < 0.5 {
                parent1.arms.clone()
            } else {
                parent2.arms.clone()
            },
        }
    }

    fn crossover_muscles(parent1: &Vec<Muscle>, parent2: &Vec<Muscle>, rng: &mut impl Rng) -> Vec<Muscle> {
        let max_len = parent1.len().max(parent2.len());
        let mut result = Vec::new();
        
        for i in 0..max_len {
            let m1 = parent1.get(i);
            let m2 = parent2.get(i);
            
            match (m1, m2) {
                (Some(m1), Some(m2)) => {
                    // Среднее между двумя мышцами
                    result.push(Muscle {
                        strength: (m1.strength + m2.strength) / 2.0,
                        speed: (m1.speed + m2.speed) / 2.0,
                        efficiency: (m1.efficiency + m2.efficiency) / 2.0,
                        endurance: (m1.endurance + m2.endurance) / 2.0,
                    });
                }
                (Some(m), None) | (None, Some(m)) => {
                    result.push(m.clone());
                }
                (None, None) => {}
            }
        }
        
        if result.is_empty() {
            // Если нет мышц, создаем одну базовую
            result.push(Muscle {
                strength: 1.0,
                speed: 1.0,
                efficiency: 0.3,
                endurance: 1.0,
            });
        }
        
        result
    }

    fn crossover_sensors(parent1: &Vec<SensorGene>, parent2: &Vec<SensorGene>, rng: &mut impl Rng) -> Vec<SensorGene> {
        let mut result = Vec::new();
        let mut used_types = std::collections::HashSet::new();
        
        // Объединяем органы чувств от обоих родителей
        for sensor in parent1.iter().chain(parent2.iter()) {
            if !used_types.contains(&sensor.sensor_type) {
                used_types.insert(sensor.sensor_type);
                result.push(sensor.clone());
            } else {
                // Если уже есть такой тип, усредняем параметры
                if let Some(existing) = result.iter_mut().find(|s| s.sensor_type == sensor.sensor_type) {
                    existing.development = (existing.development + sensor.development) / 2.0;
                    existing.range = (existing.range + sensor.range) / 2.0;
                    existing.sensitivity = (existing.sensitivity + sensor.sensitivity) / 2.0;
                    existing.maintenance_cost = (existing.maintenance_cost + sensor.maintenance_cost) / 2.0;
                    existing.active_cost = (existing.active_cost + sensor.active_cost) / 2.0;
                }
            }
        }
        
        result
    }

    fn crossover_metabolism(parent1: &MetabolismGenes, parent2: &MetabolismGenes, rng: &mut impl Rng) -> MetabolismGenes {
        // Среднее между родителями
        MetabolismGenes {
            base_metabolism: (parent1.base_metabolism + parent2.base_metabolism) / 2.0,
            energy_conversion_efficiency: (parent1.energy_conversion_efficiency + parent2.energy_conversion_efficiency) / 2.0,
            digestion_efficiency: (parent1.digestion_efficiency + parent2.digestion_efficiency) / 2.0,
        }
    }
}

impl Clone for ConnectionGene {
    fn clone(&self) -> Self {
        Self {
            innovation_number: self.innovation_number,
            from_node: self.from_node,
            to_node: self.to_node,
            weight: self.weight,
            enabled: self.enabled,
        }
    }
}

impl Clone for NodeGene {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            node_type: self.node_type,
            activation_function: self.activation_function,
        }
    }
}

