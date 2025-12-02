use cgmath::Point2;

pub struct Creature {
    pub id: u64,
    pub species_id: u32,
    pub position: Point2<f32>,
    pub velocity: cgmath::Vector2<f32>,
    pub energy: f32,
    pub energy_storage: f32,
    pub age: f32,
    pub is_alive: bool,
    pub genome: Genome,
}

#[derive(Clone)]
pub struct Genome {
    pub body_parts: BodyParts,
    pub muscles: Vec<Muscle>,
    pub sensors: Vec<SensorGene>,
    pub metabolism: MetabolismGenes,
    pub neural_network: crate::neural::Genome,
}

#[derive(Clone)]
pub struct BodyParts {
    pub torso: Torso,
    pub legs: Vec<Leg>,
    pub arms: Vec<Arm>,
}

#[derive(Clone)]
pub struct Torso {
    pub size: f32,
    pub shape: f32,
    pub color: [f32; 3],
}

#[derive(Clone)]
pub struct Leg {
    pub segments: Vec<Segment>,
    pub position: f32,
}

#[derive(Clone)]
pub struct Arm {
    pub segments: Vec<Segment>,
    pub position: f32,
}

#[derive(Clone)]
pub struct Segment {
    pub length: f32,
    pub width: f32,
}

#[derive(Clone)]
pub struct Muscle {
    pub strength: f32,
    pub speed: f32,
    pub efficiency: f32,
    pub endurance: f32,
}

#[derive(Clone)]
pub struct SensorGene {
    pub sensor_type: SensorType,
    pub development: f32,
    pub range: f32,
    pub sensitivity: f32,
    pub maintenance_cost: f32,
    pub active_cost: f32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum SensorType {
    Vision,
    Hearing,
    Touch,
    Smell,
}

#[derive(Clone)]
pub struct MetabolismGenes {
    pub base_metabolism: f32,
    pub energy_conversion_efficiency: f32,
    pub digestion_efficiency: f32,
}

impl Creature {
    pub fn new(id: u64, species_id: u32, position: Point2<f32>) -> Self {
        Self {
            id,
            species_id,
            position,
            velocity: cgmath::Vector2::new(0.0, 0.0),
            energy: 100.0,
            energy_storage: 0.0,
            age: 0.0,
            is_alive: true,
            genome: Genome::default(),
        }
    }
}

impl Default for Genome {
    fn default() -> Self {
        let mut neural_genome = crate::neural::Genome::default();
        
        // Создаем базовую нейросеть: входы (еда, энергия) -> выходы (движение)
        // Вход 0: близость еды
        // Вход 1: направление к еде X
        // Вход 2: направление к еде Y
        // Вход 3: энергия
        // Вход 4: возраст
        // Вход 5: скорость
        for i in 0..6 {
            neural_genome.node_genes.push(crate::neural::NodeGene {
                id: i,
                node_type: crate::neural::NodeType::Input,
                activation_function: crate::neural::ActivationFunction::Linear,
            });
        }
        
        // Выходы: угол движения и скорость
        neural_genome.node_genes.push(crate::neural::NodeGene {
            id: 6,
            node_type: crate::neural::NodeType::Output,
            activation_function: crate::neural::ActivationFunction::Sigmoid,
        });
        neural_genome.node_genes.push(crate::neural::NodeGene {
            id: 7,
            node_type: crate::neural::NodeType::Output,
            activation_function: crate::neural::ActivationFunction::Sigmoid,
        });
        
        // Связи: еда -> движение (существа будут двигаться к еде)
        neural_genome.connection_genes.push(crate::neural::ConnectionGene {
            innovation_number: crate::neural::get_next_innovation(),
            from_node: 1, // направление X к еде
            to_node: 6, // угол движения
            weight: 2.0,
            enabled: true,
        });
        neural_genome.connection_genes.push(crate::neural::ConnectionGene {
            innovation_number: crate::neural::get_next_innovation(),
            from_node: 2, // направление Y к еде
            to_node: 6, // угол движения
            weight: 2.0,
            enabled: true,
        });
        neural_genome.connection_genes.push(crate::neural::ConnectionGene {
            innovation_number: crate::neural::get_next_innovation(),
            from_node: 0, // близость еды
            to_node: 7, // скорость
            weight: 3.0,
            enabled: true,
        });
        neural_genome.connection_genes.push(crate::neural::ConnectionGene {
            innovation_number: crate::neural::get_next_innovation(),
            from_node: 3, // энергия
            to_node: 7, // скорость (больше энергии = быстрее)
            weight: 1.0,
            enabled: true,
        });
        
        Self {
            body_parts: BodyParts {
                torso: Torso {
                    size: 1.0,
                    shape: 0.5,
                    color: [0.5, 0.5, 0.5],
                },
                legs: vec![Leg {
                    segments: vec![Segment {
                        length: 0.5,
                        width: 0.1,
                    }],
                    position: 0.0,
                }],
                arms: vec![],
            },
            muscles: vec![Muscle {
                strength: 1.0,
                speed: 1.0,
                efficiency: 0.3,
                endurance: 1.0,
            }],
            sensors: vec![], // Убираем органы чувств по умолчанию, чтобы не тратить энергию
            metabolism: MetabolismGenes {
                base_metabolism: 0.05, // Увеличиваем базовый метаболизм для более быстрой смерти
                energy_conversion_efficiency: 0.3,
                digestion_efficiency: 0.9,
            },
            neural_network: neural_genome,
        }
    }
}

