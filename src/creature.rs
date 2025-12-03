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
    pub joint_states: Vec<JointState>, // Текущее состояние суставов (углы)
    pub muscle_activations: Vec<f32>, // Активация каждой мышцы (0.0-1.0)
}

#[derive(Clone)]
pub struct JointState {
    pub joint_id: usize,
    pub angle: f32, // Текущий угол сустава
}

#[derive(Clone)]
pub struct Genome {
    pub bones: Vec<Bone>, // Кости существа
    pub joints: Vec<crate::biomechanics::Joint>, // Суставы между костями
    pub muscles: Vec<Muscle>, // Мышцы, прикрепленные к костям
    pub sensors: Vec<SensorGene>,
    pub metabolism: MetabolismGenes,
    pub neural_network: crate::neural::Genome,
}


#[derive(Clone)]
pub struct Bone {
    pub id: usize,
    pub length: f32,
    pub width: f32,
    pub mass: f32, // Масса кости
    pub position: cgmath::Point2<f32>, // Позиция начала кости (относительно существа)
    pub angle: f32, // Угол кости относительно существа
    pub parent_bone_id: Option<usize>, // Родительская кость (для построения скелета)
}

#[derive(Clone)]
pub struct Ligament {
    pub stiffness: f32, // Жесткость связки
    pub damping: f32, // Демпфирование
    pub min_angle: f32, // Минимальный угол сустава
    pub max_angle: f32, // Максимальный угол сустава
}

#[derive(Clone)]
pub struct Muscle {
    pub strength: f32,
    pub speed: f32,
    pub efficiency: f32,
    pub endurance: f32,
    // Мышца прикреплена к двум костям через суставы
    pub bone1_id: usize, // ID первой кости
    pub bone2_id: usize, // ID второй кости
    pub joint_id: usize, // ID сустава между костями
    pub attachment_point1: f32, // Точка прикрепления на первой кости (0.0-1.0)
    pub attachment_point2: f32, // Точка прикрепления на второй кости (0.0-1.0)
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
        let genome = Genome::default();
        let muscle_count = genome.muscles.len();
        
        // Инициализируем состояния суставов
        let joint_states: Vec<JointState> = genome.joints.iter()
            .map(|j| JointState {
                joint_id: j.id,
                angle: 0.0,
            })
            .collect();
        
        Self {
            id,
            species_id,
            position,
            velocity: cgmath::Vector2::new(0.0, 0.0),
            energy: 100.0,
            energy_storage: 0.0,
            age: 0.0,
            is_alive: true,
            genome,
            joint_states,
            muscle_activations: vec![0.0; muscle_count],
        }
    }
    
    pub fn initialize_joints_and_muscles(&mut self) {
        // Инициализируем состояния суставов на основе генома
        self.joint_states = self.genome.joints.iter()
            .map(|j| JointState {
                joint_id: j.id,
                angle: 0.0,
            })
            .collect();
        self.muscle_activations = vec![0.0; self.genome.muscles.len()];
    }
}

impl Default for Genome {
    fn default() -> Self {
        let mut neural_genome = crate::neural::Genome::default();
        
        // Создаем базовую нейросеть: входы (еда, энергия, состояние суставов) -> выходы (активация мышц)
        // Вход 0: близость еды
        // Вход 1: направление к еде X
        // Вход 2: направление к еде Y
        // Вход 3: энергия
        // Вход 4: возраст
        // Вход 5: скорость X
        // Вход 6: скорость Y
        // Вход 7+: углы суставов (динамически добавляются)
        // Входы для состояния суставов будут добавляться динамически при наличии суставов
        for i in 0..7 {
            neural_genome.node_genes.push(crate::neural::NodeGene {
                id: i,
                node_type: crate::neural::NodeType::Input,
                activation_function: crate::neural::ActivationFunction::Linear,
            });
        }
        
        // Выходы: активация каждой мышцы (0.0-1.0)
        // Базовые выходы для первой мышцы (минимум 1 мышца всегда есть)
        let base_muscle_outputs = 1;
        for i in 0..base_muscle_outputs {
            neural_genome.node_genes.push(crate::neural::NodeGene {
                id: 7 + i,
                node_type: crate::neural::NodeType::Output,
                activation_function: crate::neural::ActivationFunction::Sigmoid,
            });
        }
        
        // Связи: еда -> активация мышц (существа будут учиться двигаться к еде)
        // Связь от направления к еде к первой мышце
        neural_genome.connection_genes.push(crate::neural::ConnectionGene {
            innovation_number: crate::neural::get_next_innovation(),
            from_node: 1, // направление X к еде
            to_node: 7, // активация первой мышцы
            weight: 1.0,
            enabled: true,
        });
        neural_genome.connection_genes.push(crate::neural::ConnectionGene {
            innovation_number: crate::neural::get_next_innovation(),
            from_node: 2, // направление Y к еде
            to_node: 7, // активация первой мышцы
            weight: 1.0,
            enabled: true,
        });
        neural_genome.connection_genes.push(crate::neural::ConnectionGene {
            innovation_number: crate::neural::get_next_innovation(),
            from_node: 0, // близость еды
            to_node: 7, // активация мышцы (больше еды = больше активации)
            weight: 2.0,
            enabled: true,
        });
        neural_genome.connection_genes.push(crate::neural::ConnectionGene {
            innovation_number: crate::neural::get_next_innovation(),
            from_node: 3, // энергия
            to_node: 7, // активация мышцы (больше энергии = больше активации)
            weight: 0.5,
            enabled: true,
        });
        
        // Создаем базовую структуру из костей, суставов и мышц
        // Начинаем с простого существа: туловище + одна нога
        
        // Кость 0: туловище (корневая кость)
        let torso = Bone {
            id: 0,
            length: 1.0,
            width: 0.3,
            mass: 10.0,
            position: Point2::new(0.0, 0.0),
            angle: 0.0,
            parent_bone_id: None,
        };
        
        // Кость 1: нога
        let leg = Bone {
            id: 1,
            length: 0.5,
            width: 0.1,
            mass: 2.0,
            position: Point2::new(0.0, -0.5), // Под туловищем
            angle: std::f32::consts::PI / 2.0, // Вертикально вниз
            parent_bone_id: Some(0),
        };
        
        let bones = vec![torso, leg];
        
        // Сустав между туловищем и ногой
        let joint = crate::biomechanics::Joint {
            id: 0,
            bone1_id: 0,
            bone2_id: 1,
            angle: 0.0,
            ligament: crate::biomechanics::Ligament {
                stiffness: 10.0,
                damping: 0.5,
                min_angle: -std::f32::consts::PI / 3.0, // -60 градусов
                max_angle: std::f32::consts::PI / 3.0,  // +60 градусов
            },
            position: Point2::new(0.0, -0.5),
        };
        
        let joints = vec![joint];
        
        // Мышца, соединяющая туловище и ногу
        let muscle = Muscle {
            strength: 1.0,
            speed: 1.0,
            efficiency: 0.3,
            endurance: 1.0,
            bone1_id: 0,
            bone2_id: 1,
            joint_id: 0,
            attachment_point1: 0.5, // Середина туловища
            attachment_point2: 0.0, // Начало ноги
        };
        
        let muscles = vec![muscle];
        
        Self {
            bones,
            joints,
            muscles,
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

