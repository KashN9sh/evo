use crate::creature::{Creature, Genome, SensorGene, SensorType, Muscle, MetabolismGenes, Bone};
use crate::biomechanics::{Joint, Ligament};
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
        
        // Добавляем различия в структуре костей и суставов
        let body_diff = {
            let bones_diff = (genome1.bones.len() as f32 - genome2.bones.len() as f32).abs();
            let joints_diff = (genome1.joints.len() as f32 - genome2.joints.len() as f32).abs();
            
            let mut bones_param_diff = 0.0;
            let min_bones = genome1.bones.len().min(genome2.bones.len());
            for i in 0..min_bones {
                bones_param_diff += (genome1.bones[i].length - genome2.bones[i].length).abs();
                bones_param_diff += (genome1.bones[i].width - genome2.bones[i].width).abs();
                bones_param_diff += (genome1.bones[i].mass - genome2.bones[i].mass).abs() * 0.1;
            }
            
            let mut joints_param_diff = 0.0;
            let min_joints = genome1.joints.len().min(genome2.joints.len());
            for i in 0..min_joints {
                joints_param_diff += (genome1.joints[i].ligament.stiffness - genome2.joints[i].ligament.stiffness).abs() * 0.1;
                joints_param_diff += (genome1.joints[i].ligament.damping - genome2.joints[i].ligament.damping).abs();
            }
            
            bones_diff * 0.5 + joints_diff * 0.3 + bones_param_diff * 0.1 + joints_param_diff * 0.1
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
        
        // Мутации частей тела (увеличена частота)
        if rng.gen::<f32>() < self.mutation_rate * 0.8 {
            self.mutate_body_parts(genome, rng);
        }
        
        // Мутации мышц (увеличена частота)
        if rng.gen::<f32>() < self.mutation_rate * 0.7 {
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
        // Мутация костей: изменение параметров существующих костей
        for bone in &mut genome.bones {
            if rng.gen::<f32>() < 0.6 {
                bone.length += rng.gen::<f32>() * 0.2 - 0.1;
                bone.length = bone.length.max(0.1).min(2.0);
            }
            if rng.gen::<f32>() < 0.5 {
                bone.width += rng.gen::<f32>() * 0.05 - 0.025;
                bone.width = bone.width.max(0.02).min(0.5);
            }
            if rng.gen::<f32>() < 0.4 {
                bone.mass += rng.gen::<f32>() * 1.0 - 0.5;
                bone.mass = bone.mass.max(0.5).min(20.0);
            }
            if rng.gen::<f32>() < 0.3 {
                bone.angle += rng.gen::<f32>() * 0.3 - 0.15;
            }
        }
        
        // Мутация суставов: изменение параметров связок
        for joint in &mut genome.joints {
            if rng.gen::<f32>() < 0.5 {
                joint.ligament.stiffness += rng.gen::<f32>() * 2.0 - 1.0;
                joint.ligament.stiffness = joint.ligament.stiffness.max(1.0).min(20.0);
            }
            if rng.gen::<f32>() < 0.4 {
                joint.ligament.damping += rng.gen::<f32>() * 0.2 - 0.1;
                joint.ligament.damping = joint.ligament.damping.max(0.1).min(1.0);
            }
            if rng.gen::<f32>() < 0.3 {
                let range_change = rng.gen::<f32>() * 0.5 - 0.25;
                joint.ligament.min_angle += range_change;
                joint.ligament.max_angle += range_change;
            }
        }
        
        // Добавление новой кости (увеличена вероятность)
        if rng.gen::<f32>() < 0.6 && genome.bones.len() < 15 {
            let parent_id = if genome.bones.len() > 0 {
                Some(rng.gen_range(0..genome.bones.len()))
            } else {
                None
            };
            
            let new_bone_id = genome.bones.len();
            let new_bone = Bone {
                id: new_bone_id,
                length: 0.3 + rng.gen::<f32>() * 0.4,
                width: 0.05 + rng.gen::<f32>() * 0.1,
                mass: 1.0 + rng.gen::<f32>() * 2.0,
                position: cgmath::Point3::new(
                    rng.gen::<f32>() * 0.5 - 0.25,
                    rng.gen::<f32>() * 0.5 - 0.25,
                    0.0,
                ),
                angle: rng.gen::<f32>() * std::f32::consts::PI * 2.0,
                parent_bone_id: parent_id,
            };
            
            genome.bones.push(new_bone);
            
            // ВСЕГДА создаем сустав между новой костью и родительской (если есть родитель)
            if let Some(parent_id) = parent_id {
                let joint = Joint {
                    id: genome.joints.len(),
                    bone1_id: parent_id,
                    bone2_id: new_bone_id,
                    angle: 0.0,
                    ligament: Ligament {
                        stiffness: 8.0 + rng.gen::<f32>() * 4.0,
                        damping: 0.3 + rng.gen::<f32>() * 0.3,
                        min_angle: -std::f32::consts::PI / 3.0,
                        max_angle: std::f32::consts::PI / 3.0,
                    },
                    position: genome.bones[parent_id].position,
                };
                genome.joints.push(joint);
            }
        }
        
        // Также можем добавить новый сустав между существующими костями (если его еще нет)
        if rng.gen::<f32>() < 0.3 && genome.bones.len() >= 2 && genome.joints.len() < genome.bones.len() * 2 {
            let bone1_id = rng.gen_range(0..genome.bones.len());
            let bone2_id = rng.gen_range(0..genome.bones.len());
            
            if bone1_id != bone2_id {
                // Проверяем, нет ли уже сустава между этими костями
                let joint_exists = genome.joints.iter().any(|j| 
                    (j.bone1_id == bone1_id && j.bone2_id == bone2_id) ||
                    (j.bone1_id == bone2_id && j.bone2_id == bone1_id)
                );
                
                if !joint_exists {
                    let joint = Joint {
                        id: genome.joints.len(),
                        bone1_id,
                        bone2_id,
                        angle: 0.0,
                        ligament: Ligament {
                            stiffness: 8.0 + rng.gen::<f32>() * 4.0,
                            damping: 0.3 + rng.gen::<f32>() * 0.3,
                            min_angle: -std::f32::consts::PI / 3.0,
                            max_angle: std::f32::consts::PI / 3.0,
                        },
                        position: genome.bones[bone1_id].position,
                    };
                    genome.joints.push(joint);
                }
            }
        }
        
        // Удаление кости (очень редко, только если костей много)
        if rng.gen::<f32>() < 0.1 && genome.bones.len() > 2 {
            let idx = rng.gen_range(1..genome.bones.len()); // Не удаляем корневую кость
            genome.bones.remove(idx);
            // Удаляем связанные суставы и мышцы
            genome.joints.retain(|j| j.bone1_id != idx && j.bone2_id != idx);
            genome.muscles.retain(|m| m.bone1_id != idx && m.bone2_id != idx);
        }
    }
    
    fn mutate_muscles(&self, genome: &mut Genome, rng: &mut impl Rng) {
        // Добавить/удалить мышцу (увеличена вероятность добавления)
        if rng.gen::<f32>() < 0.7 {
            if rng.gen::<f32>() < 0.8 && genome.muscles.len() < 20 {
                // Пытаемся добавить мышцу через существующий сустав
                if genome.joints.len() > 0 {
                    let joint_idx = rng.gen_range(0..genome.joints.len());
                    let joint = &genome.joints[joint_idx];
                    
                    genome.muscles.push(Muscle {
                        strength: 0.5 + rng.gen::<f32>() * 1.0,
                        speed: 0.5 + rng.gen::<f32>() * 1.0,
                        efficiency: 0.2 + rng.gen::<f32>() * 0.3,
                        endurance: 0.5 + rng.gen::<f32>() * 1.0,
                        bone1_id: joint.bone1_id,
                        bone2_id: joint.bone2_id,
                        joint_id: joint.id,
                        attachment_point1: rng.gen::<f32>(),
                        attachment_point2: rng.gen::<f32>(),
                    });
                } else if genome.bones.len() >= 2 {
                    // Если нет суставов, но есть кости - создаем новый сустав и мышцу
                    let bone1_id = rng.gen_range(0..genome.bones.len());
                    let bone2_id = rng.gen_range(0..genome.bones.len());
                    
                    if bone1_id != bone2_id {
                        // Создаем новый сустав
                        let new_joint_id = genome.joints.len();
                        let joint = Joint {
                            id: new_joint_id,
                            bone1_id,
                            bone2_id,
                            angle: 0.0,
                            ligament: Ligament {
                                stiffness: 8.0 + rng.gen::<f32>() * 4.0,
                                damping: 0.3 + rng.gen::<f32>() * 0.3,
                                min_angle: -std::f32::consts::PI / 3.0,
                                max_angle: std::f32::consts::PI / 3.0,
                            },
                            position: genome.bones[bone1_id].position,
                        };
                        genome.joints.push(joint);
                        
                        // Создаем мышцу через новый сустав
                        genome.muscles.push(Muscle {
                            strength: 0.5 + rng.gen::<f32>() * 1.0,
                            speed: 0.5 + rng.gen::<f32>() * 1.0,
                            efficiency: 0.2 + rng.gen::<f32>() * 0.3,
                            endurance: 0.5 + rng.gen::<f32>() * 1.0,
                            bone1_id,
                            bone2_id,
                            joint_id: new_joint_id,
                            attachment_point1: rng.gen::<f32>(),
                            attachment_point2: rng.gen::<f32>(),
                        });
                    }
                }
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
            
            // Мутация точек прикрепления
            if rng.gen::<f32>() < 0.3 {
                muscle.attachment_point1 += rng.gen::<f32>() * 0.2 - 0.1;
                muscle.attachment_point1 = muscle.attachment_point1.clamp(0.0, 1.0);
            }
            if rng.gen::<f32>() < 0.3 {
                muscle.attachment_point2 += rng.gen::<f32>() * 0.2 - 0.1;
                muscle.attachment_point2 = muscle.attachment_point2.clamp(0.0, 1.0);
            }
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

        // Скрещивание костей и суставов - среднее между родителями
        child.bones = Self::crossover_bones(&parent1.bones, &parent2.bones, rng);
        child.joints = Self::crossover_joints(&parent1.joints, &parent2.joints, rng);
        
        // Скрещивание мышц - среднее между родителями
        child.muscles = Self::crossover_muscles(&parent1.muscles, &parent2.muscles, rng);
        
        // Скрещивание органов чувств - объединение и усреднение
        child.sensors = Self::crossover_sensors(&parent1.sensors, &parent2.sensors, rng);
        
        // Скрещивание метаболизма - среднее между родителями
        child.metabolism = Self::crossover_metabolism(&parent1.metabolism, &parent2.metabolism, rng);

        child
    }

    fn crossover_bones(parent1: &Vec<Bone>, parent2: &Vec<Bone>, rng: &mut impl Rng) -> Vec<Bone> {
        // Выбираем родителя с меньшим количеством костей или случайного
        let (source, other) = if parent1.len() <= parent2.len() {
            (parent1, parent2)
        } else {
            (parent2, parent1)
        };
        
        let mut result = source.clone();
        
        // Добавляем недостающие кости от другого родителя (если есть)
        if result.len() < other.len() && rng.gen::<f32>() < 0.5 {
            for i in result.len()..other.len() {
                if i < other.len() {
                    let mut bone = other[i].clone();
                    bone.id = result.len();
                    result.push(bone);
                }
            }
        }
        
        // Смешиваем параметры костей
        let min_len = result.len().min(other.len());
        for i in 0..min_len {
            if rng.gen::<f32>() < 0.5 {
                result[i].length = (result[i].length + other[i].length) / 2.0;
                result[i].width = (result[i].width + other[i].width) / 2.0;
                result[i].mass = (result[i].mass + other[i].mass) / 2.0;
            }
        }
        
        result
    }
    
    fn crossover_joints(parent1: &Vec<Joint>, parent2: &Vec<Joint>, rng: &mut impl Rng) -> Vec<Joint> {
        // Выбираем родителя с меньшим количеством суставов или случайного
        let (source, other) = if parent1.len() <= parent2.len() {
            (parent1, parent2)
        } else {
            (parent2, parent1)
        };
        
        let mut result = source.clone();
        
        // Добавляем недостающие суставы от другого родителя (если есть)
        if result.len() < other.len() && rng.gen::<f32>() < 0.5 {
            for i in result.len()..other.len() {
                if i < other.len() {
                    let mut joint = other[i].clone();
                    joint.id = result.len();
                    result.push(joint);
                }
            }
        }
        
        // Смешиваем параметры связок
        let min_len = result.len().min(other.len());
        for i in 0..min_len {
            if rng.gen::<f32>() < 0.5 {
                result[i].ligament.stiffness = (result[i].ligament.stiffness + other[i].ligament.stiffness) / 2.0;
                result[i].ligament.damping = (result[i].ligament.damping + other[i].ligament.damping) / 2.0;
            }
        }
        
        result
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
                        bone1_id: m1.bone1_id,
                        bone2_id: m1.bone2_id,
                        joint_id: m1.joint_id,
                        attachment_point1: (m1.attachment_point1 + m2.attachment_point1) / 2.0,
                        attachment_point2: (m1.attachment_point2 + m2.attachment_point2) / 2.0,
                    });
                }
                (Some(m), None) | (None, Some(m)) => {
                    result.push(m.clone()); // Клонируем мышцу со всеми полями
                }
                (None, None) => {}
            }
        }
        
        if result.is_empty() {
            // Если нет мышц, создаем одну базовую (требует наличия костей и суставов)
            // Это должно быть обработано на уровне выше
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

