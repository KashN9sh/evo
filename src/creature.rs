use cgmath::Point3;

pub struct Creature {
    pub id: u64,
    pub species_id: u32,
    pub position: Point3<f32>,
    pub velocity: cgmath::Vector3<f32>,
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
    pub position: cgmath::Point3<f32>, // Позиция начала кости (относительно существа)
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
    // Мышца прикреплена к концевым суставам двух костей
    pub bone1_id: usize, // ID первой кости
    pub bone2_id: usize, // ID второй кости
    pub joint_id: usize, // ID сустава между костями (где мышца изменяет угол)
    pub end_joint1_id: usize, // ID концевого сустава первой кости (где прикреплена мышца)
    pub end_joint2_id: usize, // ID концевого сустава второй кости (где прикреплена мышца)
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
    pub fn new(id: u64, species_id: u32, position: Point3<f32>) -> Self {
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
            velocity: cgmath::Vector3::new(0.0, 0.0, 0.0),
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
        
        // Создаем начальную структуру: две кости, соединенные суставом, с мышцей
        // Структура: сустав - кость - сустав - кость - сустав
        // Мышца прикреплена к концевым суставам костей
        
        let mut bones = Vec::new();
        let mut joints = Vec::new();
        let mut muscles = Vec::new();
        
        // Кость 0: первая кость (корневая)
        let bone0_length: f32 = 1.0;
        let bone0_angle: f32 = 0.0; // Горизонтально
        let bone0_pos = Point3::new(0.0, 0.0, 0.0);
        
        // Вычисляем конец кости 0
        let bone0_end_x = bone0_pos.x + bone0_angle.cos() * bone0_length;
        let bone0_end_y = bone0_pos.y + bone0_angle.sin() * bone0_length;
        let bone0_end_z = bone0_pos.z;
        
        // Кость 1: вторая кость
        let bone1_length: f32 = 1.0;
        let bone1_angle: f32 = std::f32::consts::PI / 2.0; // Вертикально вниз
        
        // Вычисляем конец кости 1
        let bone1_end_x = bone0_end_x + bone1_angle.cos() * bone1_length;
        let bone1_end_y = bone0_end_y + bone1_angle.sin() * bone1_length;
        let bone1_end_z = bone0_end_z;
        
        // Создаем кости
        let bone0 = Bone {
            id: 0,
            length: bone0_length,
            width: 0.2,
            mass: 5.0,
            position: bone0_pos,
            angle: bone0_angle,
            parent_bone_id: None,
        };
        bones.push(bone0);
        
        let bone1 = Bone {
            id: 1,
            length: bone1_length,
            width: 0.2,
            mass: 5.0,
            position: Point3::new(0.0, 0.0, 0.0), // Позиция определяется суставом
            angle: bone1_angle,
            parent_bone_id: Some(0),
        };
        bones.push(bone1);
        
        // Сустав 0: начало кости 0 (корневой сустав)
        let joint0 = crate::biomechanics::Joint {
            id: 0,
            bone1_id: 0,
            bone2_id: 0, // Кость соединена сама с собой в начале
            angle: 0.0,
            ligament: crate::biomechanics::Ligament {
                stiffness: 10.0,
                damping: 0.5,
                min_angle: -std::f32::consts::PI / 3.0,
                max_angle: std::f32::consts::PI / 3.0,
            },
            position: Point3::new(0.0, 0.0, 0.0), // Начало кости 0
        };
        joints.push(joint0);
        
        // Сустав 1: конец кости 0 и начало кости 1 (соединяет кости)
        let joint1 = crate::biomechanics::Joint {
            id: 1,
            bone1_id: 0,
            bone2_id: 1, // Соединяет кость 0 и кость 1
            angle: 0.0,
            ligament: crate::biomechanics::Ligament {
                stiffness: 10.0,
                damping: 0.5,
                min_angle: -std::f32::consts::PI / 3.0,
                max_angle: std::f32::consts::PI / 3.0,
            },
            position: Point3::new(bone0_end_x, bone0_end_y, bone0_end_z), // Конец кости 0, начало кости 1
        };
        joints.push(joint1);
        
        // Сустав 2: конец кости 1
        let joint2 = crate::biomechanics::Joint {
            id: 2,
            bone1_id: 1,
            bone2_id: 1, // Кость соединена сама с собой в конце
            angle: 0.0,
            ligament: crate::biomechanics::Ligament {
                stiffness: 10.0,
                damping: 0.5,
                min_angle: -std::f32::consts::PI / 3.0,
                max_angle: std::f32::consts::PI / 3.0,
            },
            position: Point3::new(bone1_end_x, bone1_end_y, bone1_end_z), // Конец кости 1
        };
        joints.push(joint2);
        
        // Мышца прикреплена к концевым суставам костей (сустав 0 и сустав 2)
        // Мышца изменяет угол в суставе 1 (между костями)
        let muscle = Muscle {
            strength: 1.0,
            speed: 1.0,
            efficiency: 0.3,
            endurance: 1.0,
            bone1_id: 0,
            bone2_id: 1,
            joint_id: 1, // Сустав между костями, где изменяется угол
            end_joint1_id: 0, // Концевой сустав кости 0 (начало)
            end_joint2_id: 2, // Концевой сустав кости 1 (конец)
        };
        muscles.push(muscle);
        
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

