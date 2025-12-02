use winit::{
    event::WindowEvent,
    window::Window,
};
use wgpu::util::DeviceExt;
use cgmath::prelude::*;
use crate::creature::Creature;
use crate::food::{FoodSystem, Food};
use crate::physics::PhysicsWorld;
use crate::metabolism::Metabolism;
use crate::lifecycle::LifecycleSystem;
use crate::biomechanics::Biomechanics;
use crate::sensors::{SensorData, VisionSensor, HearingSensor, TouchSensor, SmellSensor};
use crate::neural::Network;
use rand::Rng;
use std::collections::HashMap;
use crate::sensors::{Sound, SoundType, Smell, SmellType};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

struct ActiveSound {
    position: cgmath::Point2<f32>,
    sound_type: SoundType,
    volume: f32,
    lifetime: f32,
    max_lifetime: f32,
}

struct SmellField {
    food_smells: Vec<SmellSource>,
    creature_smells: Vec<SmellSource>,
}

struct SmellSource {
    position: cgmath::Point2<f32>,
    smell_type: SmellType,
    intensity: f32,
    lifetime: f32,
    max_lifetime: f32,
}

pub struct EvolutionStats {
    pub total_creatures: usize,
    pub alive_creatures: usize,
    pub species_count: usize,
    pub total_food: usize,
    pub average_energy: f32,
    pub average_age: f32,
    pub species_populations: HashMap<u32, usize>,
    pub species_avg_energy: HashMap<u32, f32>,
    pub species_avg_age: HashMap<u32, f32>,
    pub history: Vec<StatsSnapshot>,
}

pub struct StatsSnapshot {
    pub time: f32,
    pub species_count: usize,
    pub population: usize,
}

impl EvolutionStats {
    pub fn new() -> Self {
        Self {
            total_creatures: 0,
            alive_creatures: 0,
            species_count: 0,
            total_food: 0,
            average_energy: 0.0,
            average_age: 0.0,
            species_populations: HashMap::new(),
            species_avg_energy: HashMap::new(),
            species_avg_age: HashMap::new(),
            history: Vec::new(),
        }
    }
    
    pub fn update(&mut self, creatures: &[Creature], food_count: usize, time: f32) {
        self.total_creatures = creatures.len();
        self.alive_creatures = creatures.iter().filter(|c| c.is_alive).count();
        self.total_food = food_count;
        
        let alive: Vec<_> = creatures.iter().filter(|c| c.is_alive).collect();
        
        if !alive.is_empty() {
            self.average_energy = alive.iter().map(|c| c.energy).sum::<f32>() / alive.len() as f32;
            self.average_age = alive.iter().map(|c| c.age).sum::<f32>() / alive.len() as f32;
        }
        
        // Статистика по видам
        self.species_populations.clear();
        self.species_avg_energy.clear();
        self.species_avg_age.clear();
        
        let mut species_creatures: HashMap<u32, Vec<&Creature>> = HashMap::new();
        for creature in alive {
            species_creatures.entry(creature.species_id).or_insert_with(Vec::new).push(creature);
        }
        
        self.species_count = species_creatures.len();
        
        for (species_id, species_list) in species_creatures {
            self.species_populations.insert(species_id, species_list.len());
            self.species_avg_energy.insert(
                species_id,
                species_list.iter().map(|c| c.energy).sum::<f32>() / species_list.len() as f32
            );
            self.species_avg_age.insert(
                species_id,
                species_list.iter().map(|c| c.age).sum::<f32>() / species_list.len() as f32
            );
        }
        
        // Сохраняем снимок каждые 0.5 секунды симуляционного времени (независимо от ускорения)
        if self.history.is_empty() || time - self.history.last().unwrap().time >= 0.5 {
            self.history.push(StatsSnapshot {
                time,
                species_count: self.species_count,
                population: self.alive_creatures,
            });
            
            // Ограничиваем историю последними 200 точками (больше точек для более детального графика)
            if self.history.len() > 200 {
                self.history.remove(0);
            }
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 2],
    color: [f32; 3],
}

pub struct SimulationState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    
    creatures: Vec<Creature>,
    food_system: FoodSystem,
    physics_world: PhysicsWorld,
    lifecycle_system: LifecycleSystem,
    biomechanics: Biomechanics,
    species_map: HashMap<u32, Vec<usize>>,
    next_creature_id: u64,
    next_species_id: u32,
    time: f32,
    time_scale: f32,
    slider_dragging: bool,
    mouse_pos: (f32, f32),
    stats: EvolutionStats,
    active_sounds: Vec<ActiveSound>,
    smell_field: SmellField,
    environment: Environment,
}

#[derive(Clone)]
struct Environment {
    light_level: f32, // 0.0 = темно, 1.0 = светло
    noise_level: f32, // 0.0 = тихо, 1.0 = шумно
    temperature: f32, // Температура среды
}

impl SimulationState {
    pub async fn new(window: &Window) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = unsafe {
            instance.create_surface_unsafe(
                wgpu::SurfaceTargetUnsafe::from_window(window).unwrap()
            )
        }.unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    label: None,
                },
                None,
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &config);

        let shader_source = r#"
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.position = vec4<f32>(input.position, 0.0, 1.0);
    output.color = input.color;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(input.color, 1.0);
}
"#;
        
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        let vertex_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        };

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[vertex_buffer_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let mut state = Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            creatures: Vec::new(),
            food_system: FoodSystem::new(),
            physics_world: PhysicsWorld::new(size.width as f32, size.height as f32),
            lifecycle_system: LifecycleSystem::new(),
            biomechanics: Biomechanics::new(),
            species_map: HashMap::new(),
            next_creature_id: 0,
            next_species_id: 0,
            time: 0.0,
            time_scale: 50.0, // Увеличиваем скорость по умолчанию в 50 раз
            slider_dragging: false,
            mouse_pos: (0.0, 0.0),
            stats: EvolutionStats::new(),
            active_sounds: Vec::new(),
            smell_field: SmellField {
                food_smells: Vec::new(),
                creature_smells: Vec::new(),
            },
            environment: Environment {
                light_level: 0.8, // Нормальное освещение
                noise_level: 0.3, // Умеренный шум
                temperature: 20.0, // Нормальная температура
            },
        };

        // Настраиваем rayon для использования всех доступных ядер
        let num_threads = num_cpus::get();
        let _ = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|i| format!("evo-worker-{}", i))
            .build_global();
        println!("✅ Rayon настроен на использование {} потоков", num_threads);
        
        state.initialize_population();
        state
    }

    fn initialize_population(&mut self) {
        let mut rng = rand::thread_rng();
        let initial_count = 50;
        
        for i in 0..initial_count {
            let x = rng.gen_range(100.0..1180.0);
            let y = rng.gen_range(100.0..620.0);
            
            let creature = Creature::new(
                self.next_creature_id,
                self.next_species_id,
                cgmath::Point2::new(x, y),
            );
            
            self.creatures.push(creature);
            self.species_map.entry(self.next_species_id).or_insert_with(Vec::new).push(i);
            self.next_creature_id += 1;
        }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    pub fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_pos = (position.x as f32, position.y as f32);
                if self.slider_dragging {
                    self.update_slider_from_mouse();
                }
                false
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if *button == winit::event::MouseButton::Left {
                    match state {
                        winit::event::ElementState::Pressed => {
                            if self.is_mouse_on_slider() {
                                self.slider_dragging = true;
                                self.update_slider_from_mouse();
                                return true;
                            } else {
                                // Клик по арене - спавним еду
                                self.spawn_food_at_mouse();
                                return true;
                            }
                        }
                        winit::event::ElementState::Released => {
                            self.slider_dragging = false;
                        }
                    }
                }
                false
            }
            _ => false,
        }
    }
    
    fn is_mouse_on_slider(&self) -> bool {
        let slider_x = self.size.width as f32 - 150.0;
        let slider_y = 20.0;
        let slider_width = 120.0;
        let slider_height = 20.0;
        
        self.mouse_pos.0 >= slider_x 
            && self.mouse_pos.0 <= slider_x + slider_width
            && self.mouse_pos.1 >= slider_y 
            && self.mouse_pos.1 <= slider_y + slider_height
    }
    
    fn update_slider_from_mouse(&mut self) {
        let slider_x = self.size.width as f32 - 180.0;
        let slider_width = 150.0;
        let relative_x = (self.mouse_pos.0 - slider_x).max(0.0).min(slider_width);
        // Масштаб от 1x до 500x (увеличили максимальную скорость)
        self.time_scale = 1.0 + (relative_x / slider_width) * 499.0;
    }
    
    fn spawn_food_at_mouse(&mut self) {
        // Координаты мыши уже в пикселях экрана
        let mouse_x = self.mouse_pos.0;
        let mouse_y = self.mouse_pos.1;
        
        // Проверяем, что клик не на ползунке/статистике
        let slider_x = self.size.width as f32 - 180.0;
        let slider_y = 10.0;
        let slider_width = 150.0;
        let slider_height = 25.0;
        let stats_x = 10.0 + 280.0; // Правая граница статистики
        let stats_y = 10.0;
        let stats_height = 200.0; // Высота статистики
        
        let on_slider = mouse_x >= slider_x && mouse_x <= slider_x + slider_width
            && mouse_y >= slider_y && mouse_y <= slider_y + slider_height;
        let on_stats = mouse_x >= 10.0 && mouse_x <= stats_x
            && mouse_y >= stats_y && mouse_y <= stats_y + stats_height;
        
        // Если клик не на UI элементах, спавним еду
        if !on_slider && !on_stats {
            use crate::food::Food;
            use cgmath::Point2;
            use rand::Rng;
            
            let mut rng = rand::thread_rng();
            
            // Спавним несколько кусков еды в месте клика с небольшим разбросом
            for _ in 0..3 {
                let offset_x = rng.gen_range(-20.0..20.0);
                let offset_y = rng.gen_range(-20.0..20.0);
                let food_x = (mouse_x + offset_x).max(10.0).min(self.size.width as f32 - 10.0);
                let food_y = (mouse_y + offset_y).max(10.0).min(self.size.height as f32 - 10.0);
                
                self.food_system.foods.push(Food::new(
                    Point2::new(food_x, food_y),
                    rng.gen_range(40.0..80.0), // Немного больше энергии
                ));
            }
        }
    }

    pub fn size(&self) -> winit::dpi::PhysicalSize<u32> {
        self.size
    }

    pub fn update(&mut self) {
        let dt = 0.016 * self.time_scale; // ~60 FPS, умноженное на масштаб времени
        self.time += dt;
        
        let mut rng = rand::thread_rng();
        
        // Обновляем звуки и запахи
        self.update_sounds_and_smells(dt);
        
        // Обновляем среду (динамическое освещение и шум)
        self.update_environment(dt);
        
        self.food_system.update(dt, self.physics_world.arena_size, &mut rng);
        
        // Собираем информацию о партнерах для всех существ заранее
        let creatures_snapshot: Vec<_> = self.creatures.iter().map(|c| {
            (c.id, c.species_id, c.position, c.is_alive, c.energy, c.age)
        }).collect();
        
        // Подготавливаем данные для параллельной обработки
        let foods_snapshot: Vec<_> = self.food_system.foods.iter()
            .map(|f| (f.position, f.is_eaten, f.energy_value))
            .collect();
        let sounds_snapshot: Vec<_> = self.active_sounds.iter()
            .map(|s| (s.position, s.volume, s.sound_type))
            .collect();
        let smells_snapshot: Vec<_> = self.smell_field.food_smells.iter()
            .map(|s| (s.position, s.intensity, s.smell_type))
            .chain(self.smell_field.creature_smells.iter()
                .map(|s| (s.position, s.intensity, s.smell_type)))
            .collect();
        let environment_snapshot = self.environment.clone();
        let biomechanics_snapshot = &self.biomechanics;
        let lifecycle_snapshot = &self.lifecycle_system;
        let physics_snapshot = &self.physics_world;
        
        // Параллельно обрабатываем существ (вычисляем изменения, затем применяем синхронно)
        let time_snapshot = self.time;
        // Используем with_min_len для лучшей параллелизации при малом количестве существ
        // Параллельно обрабатываем существ с минимальным размером чанка для эффективности
        // Используем меньший with_min_len для лучшей параллелизации
        let creature_updates: Vec<_> = self.creatures.par_iter()
            .with_min_len(5) // Минимум 5 существ на поток для лучшей параллелизации
            .enumerate()
            .filter_map(|(idx, creature)| {
                if !creature.is_alive {
                    return None;
                }
                
                // Копируем данные существа для вычислений
                let mut energy = creature.energy;
                let mut age = creature.age;
                let creature_pos = creature.position;
                let creature_velocity = creature.velocity;
                let creature_id = creature.id;
                let creature_species = creature.species_id;
                
                // Обновляем метаболизм (вручную, так как не можем мутировать)
                energy -= creature.genome.metabolism.base_metabolism * dt;
                
                // Обновляем возраст
                age += dt;
                
                let velocity_magnitude = (creature_velocity.x * creature_velocity.x + creature_velocity.y * creature_velocity.y).sqrt();
                
                // Рассчитываем эффективную силу с учетом биомеханики
                let muscle_force = if !creature.genome.muscles.is_empty() {
                    creature.genome.muscles.iter().map(|m| m.strength).sum::<f32>() / creature.genome.muscles.len() as f32
                } else {
                    1.0
                };
                
                let biomechanical_efficiency = biomechanics_snapshot.calculate_lever_mechanical_advantage(creature);
                
                // Затраты на движение с учетом биомеханики (увеличиваем для более быстрой смерти)
                let base_cost = velocity_magnitude * 0.03; // Увеличили с 0.01 до 0.03
                let movement_cost = base_cost / (1.0 + biomechanical_efficiency * 0.5);
                energy -= movement_cost * dt;
                
                // Дополнительные затраты на органы чувств
                for sensor in &creature.genome.sensors {
                    energy -= sensor.maintenance_cost * dt;
                    // Активные затраты при использовании
                    if sensor.development > 0.5 {
                        energy -= sensor.active_cost * dt;
                    }
                }
                
                if energy <= 0.0 {
                    return Some((idx, creature_pos, creature_velocity, 0.0, age, false, None));
                }
                
                // Находим ближайшую еду (оптимизировано: ограничиваем радиус поиска)
                let mut nearest_food_distance = 200.0; // Ограничиваем радиус поиска
                let mut nearest_food_dx = 0.0;
                let mut nearest_food_dy = 0.0;
                let mut nearest_food_idx = None;
                
                // Проверяем только еду в разумном радиусе (квадрат расстояния для оптимизации)
                let search_radius_sq = 200.0 * 200.0;
                for (i, (food_pos, is_eaten, _)) in foods_snapshot.iter().enumerate() {
                    if !is_eaten {
                        let dx = food_pos.x - creature_pos.x;
                        let dy = food_pos.y - creature_pos.y;
                        let distance_sq = dx * dx + dy * dy;
                        if distance_sq < search_radius_sq {
                            let distance = distance_sq.sqrt();
                            if distance < nearest_food_distance {
                                nearest_food_distance = distance;
                                nearest_food_dx = dx;
                                nearest_food_dy = dy;
                                nearest_food_idx = Some(i);
                            }
                        }
                    }
                }
                
                // Находим ближайшего партнера (оптимизировано: ограничиваем радиус и количество проверок)
                let mut nearest_partner_distance = 150.0; // Ограничиваем радиус поиска
                let mut nearest_partner_dx = 0.0;
                let mut nearest_partner_dy = 0.0;
                
                let can_reproduce = energy >= 50.0 && age >= 2.0;
                if can_reproduce {
                    let search_radius_sq = 150.0 * 150.0;
                    let mut checked = 0;
                    const MAX_PARTNER_CHECKS: usize = 20; // Ограничиваем количество проверок
                    for (other_id, other_species, other_pos, other_alive, other_energy, other_age) in &creatures_snapshot {
                        if checked >= MAX_PARTNER_CHECKS {
                            break; // Ранний выход
                        }
                        if *other_id != creature_id 
                            && *other_alive 
                            && *other_species == creature_species
                            && *other_energy >= 50.0 
                            && *other_age >= 2.0 {
                            let dx = other_pos.x - creature_pos.x;
                            let dy = other_pos.y - creature_pos.y;
                            let distance_sq = dx * dx + dy * dy;
                            if distance_sq < search_radius_sq {
                                checked += 1;
                                let distance = distance_sq.sqrt();
                                if distance < nearest_partner_distance {
                                    nearest_partner_distance = distance;
                                    nearest_partner_dx = dx;
                                    nearest_partner_dy = dy;
                                }
                            }
                        }
                    }
                }
                
                let mut inputs = Vec::new();
                
                let food_proximity = if nearest_food_distance < 200.0 {
                    1.0 / (1.0 + nearest_food_distance / 100.0)
                } else {
                    0.0
                };
                
                let food_direction_x = if nearest_food_distance > 0.0 {
                    nearest_food_dx / nearest_food_distance
                } else {
                    0.0
                };
                let food_direction_y = if nearest_food_distance > 0.0 {
                    nearest_food_dy / nearest_food_distance
                } else {
                    0.0
                };
                
                inputs.push(food_proximity);
                inputs.push(food_direction_x);
                inputs.push(food_direction_y);
                inputs.push(energy / 100.0);
                inputs.push(age / 100.0);
                let vel_mag = (creature_velocity.x * creature_velocity.x + creature_velocity.y * creature_velocity.y).sqrt();
                inputs.push(vel_mag / 5.0);
                
                // Подготавливаем данные для органов чувств (только в радиусе для оптимизации)
                let sensor_range = creature.genome.sensors.iter()
                    .map(|s| s.range)
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or(100.0);
                let max_sensor_range = sensor_range.max(150.0);
                let sensor_range_sq = max_sensor_range * max_sensor_range;
                
                let foods_data: Vec<_> = foods_snapshot.iter()
                    .filter(|(pos, _, _)| {
                        let dx = pos.x - creature_pos.x;
                        let dy = pos.y - creature_pos.y;
                        dx * dx + dy * dy < sensor_range_sq
                    })
                    .map(|(pos, eaten, _)| (*pos, *eaten))
                    .collect();
                let creatures_data: Vec<_> = creatures_snapshot.iter()
                    .filter(|(id, _, pos, alive, _, _)| {
                        if *id == creature_id || !*alive {
                            return false;
                        }
                        let dx = pos.x - creature_pos.x;
                        let dy = pos.y - creature_pos.y;
                        dx * dx + dy * dy < sensor_range_sq
                    })
                    .map(|(_, species, pos, _, _, _)| (*pos, *species))
                    .collect();
                
                // Добавляем данные от органов чувств
                for sensor in &creature.genome.sensors {
                    match sensor.sensor_type {
                        crate::creature::SensorType::Vision => {
                            let effective_range = sensor.range * (0.3 + environment_snapshot.light_level * 0.7);
                            let effective_resolution = sensor.development * (0.5 + environment_snapshot.light_level * 0.5);
                            
                            // Ограничиваем количество лучей для производительности
                            let ray_count = (sensor.development * 6.0).max(4.0).min(8.0) as usize;
                            let vision = VisionSensor {
                                range: effective_range,
                                angle_of_view: std::f32::consts::PI * 2.0,
                                ray_count,
                                resolution: effective_resolution,
                                color_vision: sensor.development > 0.7 && environment_snapshot.light_level > 0.5,
                            };
                            let vision_data = vision.sense_with_objects(
                                creature_pos,
                                0.0,
                                &foods_data,
                                &creatures_data,
                                physics_snapshot.arena_size,
                            );
                            for ray in &vision_data.rays {
                                inputs.push(ray.distance / effective_range);
                            }
                        }
                        crate::creature::SensorType::Hearing => {
                            let effective_sensitivity = sensor.sensitivity * (1.0 - environment_snapshot.noise_level * 0.3);
                            let effective_range = sensor.range * (0.7 + (1.0 - environment_snapshot.noise_level) * 0.3);
                            
                            let hearing = HearingSensor {
                                range: effective_range,
                                sensitivity: effective_sensitivity,
                                frequency_range: (20.0, 20000.0),
                                directionality: sensor.development,
                            };
                            let hearing_data = hearing.sense_with_sounds(creature_pos, &sounds_snapshot);
                            inputs.push(hearing_data.sounds.len() as f32 / 10.0);
                            if !hearing_data.sounds.is_empty() {
                                let avg_distance = hearing_data.sounds.iter()
                                    .map(|s| s.distance)
                                    .sum::<f32>() / hearing_data.sounds.len() as f32;
                                inputs.push(avg_distance / effective_range);
                            } else {
                                inputs.push(1.0);
                            }
                        }
                        crate::creature::SensorType::Touch => {
                            let effective_sensitivity = sensor.sensitivity * (1.0 + (1.0 - environment_snapshot.light_level) * 0.2);
                            let touch = TouchSensor {
                                sensitivity: effective_sensitivity,
                                contact_radius: sensor.range,
                                receptor_count: (sensor.development * 10.0) as usize,
                            };
                            let touch_data = touch.sense(creature_pos);
                            inputs.push(touch_data.contacts.len() as f32 / 5.0);
                        }
                        crate::creature::SensorType::Smell => {
                            let effective_sensitivity = sensor.sensitivity * (1.0 + (1.0 - environment_snapshot.light_level) * 0.5);
                            let effective_range = sensor.range * (1.0 + (1.0 - environment_snapshot.light_level) * 0.3);
                            
                            let smell = SmellSensor {
                                range: effective_range,
                                sensitivity: effective_sensitivity,
                                discrimination: sensor.development,
                            };
                            let smell_data = smell.sense_with_sources(creature_pos, &smells_snapshot);
                            let mut max_concentration: f32 = 0.0;
                            let mut food_smell_dir = 0.0;
                            for s in &smell_data.smells {
                                max_concentration = max_concentration.max(s.concentration);
                                if s.smell_type == SmellType::Food {
                                    food_smell_dir = s.direction;
                                }
                            }
                            inputs.push(max_concentration);
                            inputs.push(food_smell_dir / std::f32::consts::PI);
                        }
                    }
                }
                
                // Вычисляем выходы нейросети
                let mut network = Network::new(&creature.genome.neural_network);
                let outputs = network.forward(&inputs);
                
                // Базовое движение к еде или партнеру
                let mut target_velocity = cgmath::Vector2::new(0.0, 0.0);
                
                if food_proximity > 0.0 {
                    let base_speed = 1.5;
                    target_velocity.x = food_direction_x * base_speed;
                    target_velocity.y = food_direction_y * base_speed;
                } else if nearest_partner_distance < 150.0 && nearest_partner_distance > 0.0 {
                    let base_speed = 1.0;
                    target_velocity.x = (nearest_partner_dx / nearest_partner_distance) * base_speed;
                    target_velocity.y = (nearest_partner_dy / nearest_partner_distance) * base_speed;
                } else {
                    let angle = (creature_id as f32 * 137.508 + time_snapshot * 0.5) % (std::f32::consts::PI * 2.0);
                    let base_speed = 0.5;
                    target_velocity.x = angle.cos() * base_speed;
                    target_velocity.y = angle.sin() * base_speed;
                }
                
                // Комбинируем нейросеть и базовое движение
                let neural_speed = if outputs.len() > 1 { outputs[1].max(0.0) * 2.0 } else { 0.0 };
                let neural_angle = if outputs.len() > 0 { outputs[0] * std::f32::consts::PI * 2.0 } else { 0.0 };
                
                let neural_velocity = cgmath::Vector2::new(
                    neural_angle.cos() * neural_speed,
                    neural_angle.sin() * neural_speed,
                );
                
                // Если есть еда, приоритет базовому движению к еде (оно правильное)
                // Если еды нет, больше полагаемся на нейросеть
                let base_weight = if food_proximity > 0.0 { 0.8 } else { 0.3 };
                let neural_weight = 1.0 - base_weight;
                let mut new_vel = target_velocity * base_weight + neural_velocity * neural_weight;
                
                // Применяем трение
                let friction_coeff = biomechanics_snapshot.friction_coefficient;
                let mag = (new_vel.x * new_vel.x + new_vel.y * new_vel.y).sqrt();
                if mag > 0.0 {
                    let friction_force = friction_coeff * mag;
                    let friction_x = (new_vel.x / mag) * friction_force;
                    let friction_y = (new_vel.y / mag) * friction_force;
                    new_vel.x = new_vel.x - friction_x * dt;
                    new_vel.y = new_vel.y - friction_y * dt;
                    
                    // Останавливаем очень медленное движение
                    if mag < 0.01 {
                        new_vel.x = 0.0;
                        new_vel.y = 0.0;
                    }
                }
                
                // Обновляем позицию
                let mut new_pos = creature_pos;
                new_pos.x += new_vel.x * dt;
                new_pos.y += new_vel.y * dt;
                
                // Проверяем границы
                let radius = 15.0 + creature.genome.body_parts.torso.size * 10.0;
                if new_pos.x < radius {
                    new_pos.x = radius;
                    new_vel.x = 0.0;
                }
                if new_pos.x > physics_snapshot.arena_size.0 - radius {
                    new_pos.x = physics_snapshot.arena_size.0 - radius;
                    new_vel.x = 0.0;
                }
                if new_pos.y < radius {
                    new_pos.y = radius;
                    new_vel.y = 0.0;
                }
                if new_pos.y > physics_snapshot.arena_size.1 - radius {
                    new_pos.y = physics_snapshot.arena_size.1 - radius;
                    new_vel.y = 0.0;
                }
                
                // Пытаемся съесть еду
                let ate_food = nearest_food_distance < 30.0 && nearest_food_idx.is_some();
                
                Some((idx, new_pos, new_vel, energy, age, ate_food, nearest_food_idx))
            })
            .collect();
        
        // Применяем вычисленные изменения к существам (синхронно)
        for (creature_idx, new_pos, new_vel, new_energy, new_age, ate_food, food_idx) in creature_updates {
            if creature_idx < self.creatures.len() {
                let creature = &mut self.creatures[creature_idx];
                creature.position = new_pos;
                creature.velocity = new_vel;
                creature.energy = new_energy;
                creature.age = new_age;
                
                if new_energy <= 0.0 {
                    creature.is_alive = false;
                }
                
                // Обрабатываем поедание еды
                if ate_food {
                    if let Some(idx) = food_idx {
                        if idx < self.food_system.foods.len() {
                            let distance = 30.0; // Уже проверено в параллельном коде
                            if self.lifecycle_system.try_eat(creature, &mut self.food_system.foods[idx], distance) {
                                // Генерируем звук поедания
                                self.active_sounds.push(ActiveSound {
                                    position: self.food_system.foods[idx].position,
                                    sound_type: SoundType::Eating,
                                    volume: 0.5,
                                    lifetime: 0.3,
                                    max_lifetime: 0.3,
                                });
                            }
                        }
                    }
                }
                
                // Генерируем звук при движении
                let velocity_magnitude = (new_vel.x * new_vel.x + new_vel.y * new_vel.y).sqrt();
                if velocity_magnitude > 0.1 {
                    self.active_sounds.push(ActiveSound {
                        position: new_pos,
                        sound_type: SoundType::Movement,
                        volume: velocity_magnitude * 0.1,
                        lifetime: 0.5,
                        max_lifetime: 0.5,
                    });
                }
                
                // Генерируем запах от существ
                self.smell_field.creature_smells.push(SmellSource {
                    position: new_pos,
                    smell_type: if creature.species_id == 0 {
                        SmellType::SameSpecies
                    } else {
                        SmellType::DifferentSpecies
                    },
                    intensity: 1.0,
                    lifetime: 2.0,
                    max_lifetime: 2.0,
                });
            }
        }
        
        // УДАЛЕНО: дублирование генерации звуков и запахов (уже делается выше в параллельном коде)
        
        // Удаляем мертвых существ
        self.creatures.retain(|c| c.is_alive);
        
        // Если слишком много существ, убиваем самых слабых
        const MAX_CREATURES: usize = 20; // Максимальное количество существ
        if self.creatures.len() > MAX_CREATURES {
            // Используем частичную сортировку для производительности
            let to_remove = self.creatures.len() - MAX_CREATURES;
            self.creatures.select_nth_unstable_by(to_remove, |a, b| {
                a.energy.partial_cmp(&b.energy).unwrap_or(std::cmp::Ordering::Equal)
            });
            for i in 0..to_remove {
                if i < self.creatures.len() {
                    self.creatures[i].is_alive = false;
                }
            }
            self.creatures.retain(|c| c.is_alive);
        }
        
        self.update_species_map();
        self.handle_speciation(&mut rng);
        // Вызываем размножение несколько раз в зависимости от ускорения времени
        // При ускорении времени размножение должно происходить чаще
        let reproduction_iterations = (self.time_scale / 10.0).max(1.0).min(10.0) as usize;
        for _ in 0..reproduction_iterations {
            self.handle_reproduction(&mut rng);
        }
        
        // Обновляем статистику
        self.stats.update(&self.creatures, self.food_system.foods.len(), self.time);
        
        // Генерируем запахи от еды (только раз в секунду, чтобы не накапливать слишком много)
        if (self.time * 10.0) as usize % 10 == 0 {
            for food in &self.food_system.foods {
                if !food.is_eaten {
                    self.smell_field.food_smells.push(SmellSource {
                        position: food.position,
                        smell_type: SmellType::Food,
                        intensity: 2.0,
                        lifetime: 10.0,
                        max_lifetime: 10.0,
                    });
                }
            }
        }
    }
    
    fn update_sounds_and_smells(&mut self, dt: f32) {
        // Обновляем звуки
        self.active_sounds.retain_mut(|sound| {
            sound.lifetime -= dt;
            sound.lifetime > 0.0
        });
        
        // Ограничиваем количество звуков для производительности
        const MAX_SOUNDS: usize = 200;
        if self.active_sounds.len() > MAX_SOUNDS {
            self.active_sounds.sort_by(|a, b| a.lifetime.partial_cmp(&b.lifetime).unwrap_or(std::cmp::Ordering::Equal));
            self.active_sounds.truncate(MAX_SOUNDS);
        }
        
        // Обновляем запахи
        self.smell_field.food_smells.retain_mut(|smell| {
            smell.lifetime -= dt;
            smell.intensity *= 0.99; // Постепенное затухание
            smell.lifetime > 0.0 && smell.intensity > 0.1
        });
        
        self.smell_field.creature_smells.retain_mut(|smell| {
            smell.lifetime -= dt;
            smell.intensity *= 0.95; // Быстрее затухает
            smell.lifetime > 0.0 && smell.intensity > 0.1
        });
        
        // Ограничиваем количество запахов для производительности
        const MAX_FOOD_SMELLS: usize = 100;
        const MAX_CREATURE_SMELLS: usize = 200;
        if self.smell_field.food_smells.len() > MAX_FOOD_SMELLS {
            self.smell_field.food_smells.sort_by(|a, b| a.lifetime.partial_cmp(&b.lifetime).unwrap_or(std::cmp::Ordering::Equal));
            self.smell_field.food_smells.truncate(MAX_FOOD_SMELLS);
        }
        if self.smell_field.creature_smells.len() > MAX_CREATURE_SMELLS {
            self.smell_field.creature_smells.sort_by(|a, b| a.lifetime.partial_cmp(&b.lifetime).unwrap_or(std::cmp::Ordering::Equal));
            self.smell_field.creature_smells.truncate(MAX_CREATURE_SMELLS);
        }
    }
    
    fn update_environment(&mut self, _dt: f32) {
        // Динамическое освещение (циклическое изменение)
        self.environment.light_level = 0.5 + 0.3 * (self.time * 0.1).sin();
        
        // Шум зависит от количества активных звуков
        let active_sound_count = self.active_sounds.len();
        self.environment.noise_level = (active_sound_count as f32 / 50.0).min(1.0) * 0.5;
        
        // Температура может варьироваться
        self.environment.temperature = 20.0 + 5.0 * (self.time * 0.05).sin();
    }

    fn handle_speciation(&mut self, _rng: &mut impl Rng) {
        let evolution_system = crate::evolution::EvolutionSystem::new();
        let mut new_species_assignments: HashMap<usize, u32> = HashMap::new();
        let species_map_clone = self.species_map.clone();
        let creatures_genomes: Vec<_> = self.creatures.iter().map(|c| &c.genome).collect();
        
        for (i, creature_genome) in creatures_genomes.iter().enumerate() {
            let mut assigned_species = None;
            
            for (&species_id, indices) in &species_map_clone {
                if let Some(&representative_idx) = indices.first() {
                    if representative_idx < self.creatures.len() {
                        let compatibility = evolution_system.calculate_compatibility(
                            creature_genome,
                            &self.creatures[representative_idx].genome,
                        );
                        
                        if compatibility < evolution_system.compatibility_threshold {
                            assigned_species = Some(species_id);
                            break;
                        }
                    }
                }
            }
            
            if assigned_species.is_none() {
                assigned_species = Some(self.next_species_id);
                self.next_species_id += 1;
            }
            
            if assigned_species != Some(self.creatures[i].species_id) {
                new_species_assignments.insert(i, assigned_species.unwrap());
            }
        }
        
        for (idx, new_species_id) in new_species_assignments {
            self.creatures[idx].species_id = new_species_id;
        }
    }

    fn update_species_map(&mut self) {
        self.species_map.clear();
        for (i, creature) in self.creatures.iter().enumerate() {
            self.species_map.entry(creature.species_id).or_insert_with(Vec::new).push(i);
        }
    }

    fn handle_reproduction(&mut self, rng: &mut impl Rng) {
        // Используем time_scale для масштабирования вероятности размножения
        let time_scale = self.time_scale;
        // Ограничиваем размножение, если уже много существ
        const MAX_CREATURES: usize = 500;
        if self.creatures.len() >= MAX_CREATURES {
            return;
        }
        
        let mut new_creatures = Vec::new();
        let evolution_system = crate::evolution::EvolutionSystem::new();
        
        // Собираем всех потенциальных родителей (живые, могут размножаться)
        let mut all_potential_parents: Vec<(usize, u32)> = Vec::new();
        for (i, creature) in self.creatures.iter().enumerate() {
            if creature.is_alive 
                && self.lifecycle_system.can_reproduce(creature)
                && creature.energy >= 50.0 {
                all_potential_parents.push((i, creature.species_id));
            }
        }
        
        if all_potential_parents.len() < 2 {
            return;
        }
        
        // Перемешиваем потенциальных родителей
        for i in 0..all_potential_parents.len() {
            let j = rng.gen_range(0..all_potential_parents.len());
            all_potential_parents.swap(i, j);
        }
        
        // Пробуем создать пары для размножения
        for i in 0..(all_potential_parents.len() - 1) {
            let (parent1_idx, parent1_species) = all_potential_parents[i];
            let (parent2_idx, parent2_species) = all_potential_parents[i + 1];
            
            // Пропускаем, если это одно и то же существо
            if parent1_idx == parent2_idx {
                continue;
            }
            
            let parent1_pos = self.creatures[parent1_idx].position;
            let parent2_pos = self.creatures[parent2_idx].position;
            
            let dx = parent1_pos.x - parent2_pos.x;
            let dy = parent1_pos.y - parent2_pos.y;
            let distance = (dx * dx + dy * dy).sqrt();
            
            if distance < 40.0 {
                // Вычисляем совместимость геномов
                let compatibility = evolution_system.calculate_compatibility(
                    &self.creatures[parent1_idx].genome,
                    &self.creatures[parent2_idx].genome,
                );
                
                // Вероятность размножения зависит от:
                // 1. Ускорения времени
                // 2. Совместимости геномов (более совместимые = выше шанс)
                // 3. Одинаковый вид = выше шанс, разные виды = ниже, но возможно
                let base_chance = 0.1 * (time_scale / 10.0).min(1.0);
                let compatibility_bonus = if compatibility < evolution_system.compatibility_threshold {
                    1.0 // Очень совместимые (один вид или близкие виды)
                } else if compatibility < evolution_system.compatibility_threshold * 2.0 {
                    0.5 // Умеренно совместимые (далекие, но возможные)
                } else {
                    0.1 // Мало совместимые (очень разные виды, но возможно)
                };
                
                let same_species_bonus = if parent1_species == parent2_species { 1.0 } else { 0.7 };
                let reproduction_chance = base_chance * compatibility_bonus * same_species_bonus;
                
                if rng.gen::<f32>() < reproduction_chance {
                    self.creatures[parent1_idx].energy -= self.lifecycle_system.reproduction_energy_cost;
                    self.creatures[parent2_idx].energy -= self.lifecycle_system.reproduction_energy_cost;
                    
                    let parent1_genome = self.creatures[parent1_idx].genome.clone();
                    let parent2_genome = self.creatures[parent2_idx].genome.clone();
                    
                    let mut child_genome = evolution_system.crossover(&parent1_genome, &parent2_genome, rng);
                    evolution_system.mutate(&mut child_genome, rng);
                    
                    let child_position = cgmath::Point2::new(
                        (parent1_pos.x + parent2_pos.x) / 2.0 + rng.gen_range(-5.0..5.0),
                        (parent1_pos.y + parent2_pos.y) / 2.0 + rng.gen_range(-5.0..5.0),
                    );
                    
                    // Определяем вид потомка: если родители одного вида - тот же вид,
                    // иначе создаем новый вид или выбираем вид более совместимого родителя
                    let child_species = if parent1_species == parent2_species {
                        parent1_species
                    } else {
                        // Межвидовое скрещивание - создаем новый вид или наследуем от более "сильного" родителя
                        if compatibility < evolution_system.compatibility_threshold * 1.5 {
                            // Достаточно совместимы - наследуем вид от родителя с большей энергией
                            if self.creatures[parent1_idx].energy > self.creatures[parent2_idx].energy {
                                parent1_species
                            } else {
                                parent2_species
                            }
                        } else {
                            // Слишком разные - создаем новый вид
                            let new_species = self.next_species_id;
                            self.next_species_id += 1;
                            new_species
                        }
                    };
                    
                    // Звук размножения
                    self.active_sounds.push(ActiveSound {
                        position: child_position,
                        sound_type: SoundType::Reproduction,
                        volume: 0.8,
                        lifetime: 0.5,
                        max_lifetime: 0.5,
                    });
                    
                    let mut child = Creature::new(
                        self.next_creature_id,
                        child_species,
                        child_position,
                    );
                    child.genome = child_genome;
                    child.energy = 100.0;
                    
                    new_creatures.push(child);
                    self.next_creature_id += 1;
                }
            }
        }
        
        self.creatures.extend(new_creatures);
    }

    fn collect_sensor_inputs(&self, creature: &Creature) -> Vec<f32> {
        let mut inputs = Vec::new();
        
        // Находим ближайшую еду для всех существ (даже без зрения)
        let mut nearest_food_distance = 1000.0;
        let mut nearest_food_dx = 0.0;
        let mut nearest_food_dy = 0.0;
        
        for food in &self.food_system.foods {
            if !food.is_eaten {
                let dx = food.position.x - creature.position.x;
                let dy = food.position.y - creature.position.y;
                let distance = (dx * dx + dy * dy).sqrt();
                if distance < nearest_food_distance {
                    nearest_food_distance = distance;
                    nearest_food_dx = dx;
                    nearest_food_dy = dy;
                }
            }
        }
        
        // Нормализуем расстояние до еды (0-1, где 1 = очень близко)
        let food_proximity = if nearest_food_distance < 1000.0 {
            1.0 / (1.0 + nearest_food_distance / 100.0)
        } else {
            0.0
        };
        
        // Направление к еде (нормализованное)
        let food_direction_x = if nearest_food_distance > 0.0 {
            nearest_food_dx / nearest_food_distance
        } else {
            0.0
        };
        let food_direction_y = if nearest_food_distance > 0.0 {
            nearest_food_dy / nearest_food_distance
        } else {
            0.0
        };
        
        // Добавляем информацию о еде как базовые входы
        inputs.push(food_proximity);
        inputs.push(food_direction_x);
        inputs.push(food_direction_y);
        
        // Подготавливаем данные для органов чувств
        let foods_data: Vec<_> = self.food_system.foods.iter()
            .map(|f| (f.position, f.is_eaten))
            .collect();
        let creatures_data: Vec<_> = self.creatures.iter()
            .filter(|c| c.id != creature.id && c.is_alive)
            .map(|c| (c.position, c.species_id))
            .collect();
        let sounds_data: Vec<_> = self.active_sounds.iter()
            .map(|s| (s.position, s.volume, s.sound_type))
            .collect();
        let smell_sources: Vec<_> = self.smell_field.food_smells.iter()
            .map(|s| (s.position, s.intensity, s.smell_type))
            .chain(self.smell_field.creature_smells.iter()
                .map(|s| (s.position, s.intensity, s.smell_type)))
            .collect();
        
        for sensor in &creature.genome.sensors {
            match sensor.sensor_type {
                crate::creature::SensorType::Vision => {
                    // Адаптация зрения к освещению
                    let effective_range = sensor.range * (0.3 + self.environment.light_level * 0.7);
                    let effective_resolution = sensor.development * (0.5 + self.environment.light_level * 0.5);
                    
                    let vision = VisionSensor {
                        range: effective_range,
                        angle_of_view: std::f32::consts::PI * 2.0,
                        ray_count: 8,
                        resolution: effective_resolution,
                        color_vision: sensor.development > 0.7 && self.environment.light_level > 0.5,
                    };
                    let vision_data = vision.sense_with_objects(
                        creature.position,
                        0.0,
                        &foods_data,
                        &creatures_data,
                        self.physics_world.arena_size,
                    );
                    for ray in &vision_data.rays {
                        inputs.push(ray.distance / effective_range);
                    }
                }
                crate::creature::SensorType::Hearing => {
                    // Адаптация слуха к шуму
                    let effective_sensitivity = sensor.sensitivity * (1.0 - self.environment.noise_level * 0.3);
                    let effective_range = sensor.range * (0.7 + (1.0 - self.environment.noise_level) * 0.3);
                    
                    let hearing = HearingSensor {
                        range: effective_range,
                        sensitivity: effective_sensitivity,
                        frequency_range: (20.0, 20000.0),
                        directionality: sensor.development,
                    };
                    let hearing_data = hearing.sense_with_sounds(creature.position, &sounds_data);
                    inputs.push(hearing_data.sounds.len() as f32 / 10.0);
                    if !hearing_data.sounds.is_empty() {
                        let avg_distance = hearing_data.sounds.iter()
                            .map(|s| s.distance)
                            .sum::<f32>() / hearing_data.sounds.len() as f32;
                        inputs.push(avg_distance / effective_range);
                    } else {
                        inputs.push(1.0);
                    }
                }
                crate::creature::SensorType::Touch => {
                    // Осязание улучшается в темноте
                    let effective_sensitivity = sensor.sensitivity * (1.0 + (1.0 - self.environment.light_level) * 0.2);
                    let touch = TouchSensor {
                        sensitivity: effective_sensitivity,
                        contact_radius: sensor.range,
                        receptor_count: (sensor.development * 10.0) as usize,
                    };
                    let touch_data = touch.sense(creature.position);
                    inputs.push(touch_data.contacts.len() as f32 / 5.0);
                }
                crate::creature::SensorType::Smell => {
                    // Обоняние улучшается в темноте
                    let effective_sensitivity = sensor.sensitivity * (1.0 + (1.0 - self.environment.light_level) * 0.5);
                    let effective_range = sensor.range * (1.0 + (1.0 - self.environment.light_level) * 0.3);
                    
                    let smell = SmellSensor {
                        range: effective_range,
                        sensitivity: effective_sensitivity,
                        discrimination: sensor.development,
                    };
                    let smell_data = smell.sense_with_sources(creature.position, &smell_sources);
                    let mut max_concentration: f32 = 0.0;
                    let mut food_smell_dir = 0.0;
                    for s in &smell_data.smells {
                        max_concentration = max_concentration.max(s.concentration);
                        if s.smell_type == SmellType::Food {
                            food_smell_dir = s.direction;
                        }
                    }
                    inputs.push(max_concentration);
                    inputs.push(food_smell_dir / std::f32::consts::PI);
                }
            }
        }
        
        inputs.push(creature.energy / 100.0);
        inputs.push(creature.age / 100.0);
        let vel_mag = (creature.velocity.x * creature.velocity.x + creature.velocity.y * creature.velocity.y).sqrt();
        inputs.push(vel_mag / 5.0);
        
        inputs
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Собираем все вершины
        let mut all_vertices = Vec::new();
        let mut offsets = Vec::new();
        
        // Вершины существ (только живые, ограничиваем для производительности)
        let creatures_to_render: Vec<_> = self.creatures.iter()
            .filter(|c| c.is_alive)
            .take(300) // Ограничиваем рендеринг до 300 существ для производительности
            .collect();
        
        for creature in creatures_to_render {
            let start = all_vertices.len();
            all_vertices.extend(self.create_creature_vertices(creature));
            offsets.push((start, all_vertices.len() - start));
        }
        
        // Вершины еды
        for food in &self.food_system.foods {
            if food.is_eaten {
                continue;
            }
            let start = all_vertices.len();
            all_vertices.extend(self.create_food_vertices(food));
            offsets.push((start, all_vertices.len() - start));
        }
        
        let vertex_buffer = if !all_vertices.is_empty() {
            Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(&all_vertices),
                usage: wgpu::BufferUsages::VERTEX,
            }))
        } else {
            None
        };
        
        // Вершины ползунка
        let slider_vertices = self.create_slider_vertices();
        let slider_buffer = if !slider_vertices.is_empty() {
            Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Slider Vertex Buffer"),
                contents: bytemuck::cast_slice(&slider_vertices),
                usage: wgpu::BufferUsages::VERTEX,
            }))
        } else {
            None
        };
        
        // Вершины статистики
        let stats_vertices = self.create_stats_text_vertices();
        let stats_buffer = if !stats_vertices.is_empty() {
            Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Stats Vertex Buffer"),
                contents: bytemuck::cast_slice(&stats_vertices),
                usage: wgpu::BufferUsages::VERTEX,
            }))
        } else {
            None
        };
        
        // Вершины графика эволюции
        let graph_vertices = self.create_evolution_graph_vertices();
        let graph_vertices_len = graph_vertices.len();
        let graph_buffer = if !graph_vertices.is_empty() {
            Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Graph Vertex Buffer"),
                contents: bytemuck::cast_slice(&graph_vertices),
                usage: wgpu::BufferUsages::VERTEX,
            }))
        } else {
            None
        };

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.1,
                            b: 0.1,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            
            if let Some(ref vertex_buffer) = vertex_buffer {
                render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                
                // Отрисовываем все объекты
                for (start, count) in offsets {
                    render_pass.draw(start as u32..(start + count) as u32, 0..1);
                }
            }
            
            // Отрисовка статистики
            if let Some(ref stats_buffer) = stats_buffer {
                render_pass.set_vertex_buffer(0, stats_buffer.slice(..));
                render_pass.draw(0..stats_vertices.len() as u32, 0..1);
            }
            
            // Отрисовка графика эволюции
            if let Some(ref graph_buffer) = graph_buffer {
                render_pass.set_vertex_buffer(0, graph_buffer.slice(..));
                render_pass.draw(0..graph_vertices_len as u32, 0..1);
            }
            
            // Отрисовка ползунка скорости времени
            if let Some(ref slider_buffer) = slider_buffer {
                render_pass.set_vertex_buffer(0, slider_buffer.slice(..));
                render_pass.draw(0..slider_vertices.len() as u32, 0..1);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    fn create_creature_vertices(&self, creature: &Creature) -> Vec<Vertex> {
        let mut vertices = Vec::new();
        
        // Размер существа зависит от размера туловища
        let base_radius = 15.0 + creature.genome.body_parts.torso.size * 10.0;
        let radius = base_radius;
        let segments = 20; // Больше сегментов для более гладкого круга
        
        // Цвет по виду (простая хеш-функция для генерации цвета)
        let hue = (creature.species_id as f32 * 137.508) % 1.0;
        let mut color = self.hsv_to_rgb(hue, 0.8, 0.9);
        
        // Интенсивность цвета зависит от энергии (более яркие = больше энергии)
        let energy_factor = creature.energy / 100.0;
        color[0] = color[0] * 0.4 + color[0] * energy_factor * 0.6;
        color[1] = color[1] * 0.4 + color[1] * energy_factor * 0.6;
        color[2] = color[2] * 0.4 + color[2] * energy_factor * 0.6;
        
        // Добавляем обводку для лучшей видимости
        let outline_color = [color[0] * 0.5, color[1] * 0.5, color[2] * 0.5];
        
        // Нормализуем позицию в координаты экрана (-1 до 1)
        let x = (creature.position.x / self.size.width as f32) * 2.0 - 1.0;
        let y = 1.0 - (creature.position.y / self.size.height as f32) * 2.0;
        
        // Создаем туловище (форма зависит от shape параметра)
        let radius_norm = radius / (self.size.width.min(self.size.height) as f32);
        let shape_factor = creature.genome.body_parts.torso.shape;
        
        // Обводка туловища (немного больше)
        let outline_radius = radius_norm * 1.1;
        let outline_radius_x = outline_radius * (1.0 + shape_factor * 0.5);
        let outline_radius_y = outline_radius * (1.0 - shape_factor * 0.3);
        
        for i in 0..segments {
            let angle1 = (i as f32 / segments as f32) * std::f32::consts::PI * 2.0;
            let angle2 = ((i + 1) as f32 / segments as f32) * std::f32::consts::PI * 2.0;
            
            // Форма туловища: от круга (shape=0) до овала (shape=1)
            let radius_x = radius_norm * (1.0 + shape_factor * 0.5);
            let radius_y = radius_norm * (1.0 - shape_factor * 0.3);
            
            // Обводка (внешний круг)
            vertices.push(Vertex {
                position: [x, y],
                color: outline_color,
            });
            vertices.push(Vertex {
                position: [
                    x + angle1.cos() * outline_radius_x,
                    y + angle1.sin() * outline_radius_y,
                ],
                color: outline_color,
            });
            vertices.push(Vertex {
                position: [
                    x + angle2.cos() * outline_radius_x,
                    y + angle2.sin() * outline_radius_y,
                ],
                color: outline_color,
            });
            
            // Основной круг
            vertices.push(Vertex {
                position: [x, y],
                color: color,
            });
            vertices.push(Vertex {
                position: [
                    x + angle1.cos() * radius_x,
                    y + angle1.sin() * radius_y,
                ],
                color: color,
            });
            vertices.push(Vertex {
                position: [
                    x + angle2.cos() * radius_x,
                    y + angle2.sin() * radius_y,
                ],
                color: color,
            });
        }
        
        // Отрисовка ног
        let leg_count = creature.genome.body_parts.legs.len();
        if leg_count > 0 {
            let leg_spacing = std::f32::consts::PI * 2.0 / leg_count as f32;
            for (i, leg) in creature.genome.body_parts.legs.iter().enumerate() {
                let leg_angle = (i as f32 * leg_spacing) - std::f32::consts::PI / 2.0;
                let leg_start_x = x + leg_angle.cos() * radius_norm;
                let leg_start_y = y + leg_angle.sin() * radius_norm;
                
                let mut current_x = leg_start_x;
                let mut current_y = leg_start_y;
                let mut current_angle = leg_angle;
                
                for segment in &leg.segments {
                    let segment_length = segment.length * 0.02; // Масштабируем
                    let segment_end_x = current_x + current_angle.cos() * segment_length;
                    let segment_end_y = current_y + current_angle.sin() * segment_length;
                    
                    // Рисуем сегмент ноги
                    let segment_width = segment.width * 0.01;
                    let perp_angle = current_angle + std::f32::consts::PI / 2.0;
                    
                    let p1_x = current_x + perp_angle.cos() * segment_width;
                    let p1_y = current_y + perp_angle.sin() * segment_width;
                    let p2_x = current_x - perp_angle.cos() * segment_width;
                    let p2_y = current_y - perp_angle.sin() * segment_width;
                    let p3_x = segment_end_x + perp_angle.cos() * segment_width;
                    let p3_y = segment_end_y + perp_angle.sin() * segment_width;
                    let p4_x = segment_end_x - perp_angle.cos() * segment_width;
                    let p4_y = segment_end_y - perp_angle.sin() * segment_width;
                    
                    // Два треугольника для сегмента
                    vertices.push(Vertex { position: [p1_x, p1_y], color });
                    vertices.push(Vertex { position: [p2_x, p2_y], color });
                    vertices.push(Vertex { position: [p3_x, p3_y], color });
                    vertices.push(Vertex { position: [p2_x, p2_y], color });
                    vertices.push(Vertex { position: [p4_x, p4_y], color });
                    vertices.push(Vertex { position: [p3_x, p3_y], color });
                    
                    current_x = segment_end_x;
                    current_y = segment_end_y;
                    // Небольшой изгиб в суставе
                    current_angle += 0.1;
                }
            }
        }
        
        // Отрисовка рук (аналогично ногам)
        let arm_count = creature.genome.body_parts.arms.len();
        if arm_count > 0 {
            let arm_spacing = std::f32::consts::PI * 2.0 / arm_count as f32;
            for (i, arm) in creature.genome.body_parts.arms.iter().enumerate() {
                let arm_angle = (i as f32 * arm_spacing) + std::f32::consts::PI / 2.0;
                let arm_start_x = x + arm_angle.cos() * radius_norm;
                let arm_start_y = y + arm_angle.sin() * radius_norm;
                
                let mut current_x = arm_start_x;
                let mut current_y = arm_start_y;
                let mut current_angle = arm_angle;
                
                for segment in &arm.segments {
                    let segment_length = segment.length * 0.015;
                    let segment_end_x = current_x + current_angle.cos() * segment_length;
                    let segment_end_y = current_y + current_angle.sin() * segment_length;
                    
                    let segment_width = segment.width * 0.008;
                    let perp_angle = current_angle + std::f32::consts::PI / 2.0;
                    
                    let p1_x = current_x + perp_angle.cos() * segment_width;
                    let p1_y = current_y + perp_angle.sin() * segment_width;
                    let p2_x = current_x - perp_angle.cos() * segment_width;
                    let p2_y = current_y - perp_angle.sin() * segment_width;
                    let p3_x = segment_end_x + perp_angle.cos() * segment_width;
                    let p3_y = segment_end_y + perp_angle.sin() * segment_width;
                    let p4_x = segment_end_x - perp_angle.cos() * segment_width;
                    let p4_y = segment_end_y - perp_angle.sin() * segment_width;
                    
                    vertices.push(Vertex { position: [p1_x, p1_y], color });
                    vertices.push(Vertex { position: [p2_x, p2_y], color });
                    vertices.push(Vertex { position: [p3_x, p3_y], color });
                    vertices.push(Vertex { position: [p2_x, p2_y], color });
                    vertices.push(Vertex { position: [p4_x, p4_y], color });
                    vertices.push(Vertex { position: [p3_x, p3_y], color });
                    
                    current_x = segment_end_x;
                    current_y = segment_end_y;
                    current_angle += 0.1;
                }
            }
        }
        
        // Визуализация органов чувств (опционально, для отладки)
        // Радиус зрения
        for sensor in &creature.genome.sensors {
            if sensor.sensor_type == crate::creature::SensorType::Vision && sensor.development > 0.3 {
                let vision_radius = sensor.range / (self.size.width.min(self.size.height) as f32);
                let vision_color = [1.0, 1.0, 0.0]; // Желтый для зрения
                let vision_alpha = sensor.development * 0.3;
                
                // Рисуем круг зрения (только контур)
                for i in 0..32 {
                    let angle1 = (i as f32 / 32.0) * std::f32::consts::PI * 2.0;
                    let angle2 = ((i + 1) as f32 / 32.0) * std::f32::consts::PI * 2.0;
                    
                    let x1 = x + angle1.cos() * vision_radius;
                    let y1 = y + angle1.sin() * vision_radius;
                    let x2 = x + angle2.cos() * vision_radius;
                    let y2 = y + angle2.sin() * vision_radius;
                    
                    // Тонкая линия
                    let line_width = 0.002;
                    vertices.push(Vertex {
                        position: [x1 - line_width, y1],
                        color: [vision_color[0] * vision_alpha, vision_color[1] * vision_alpha, vision_color[2] * vision_alpha],
                    });
                    vertices.push(Vertex {
                        position: [x1 + line_width, y1],
                        color: [vision_color[0] * vision_alpha, vision_color[1] * vision_alpha, vision_color[2] * vision_alpha],
                    });
                    vertices.push(Vertex {
                        position: [x2, y2],
                        color: [vision_color[0] * vision_alpha, vision_color[1] * vision_alpha, vision_color[2] * vision_alpha],
                    });
                }
            }
        }
        
        vertices
    }

    fn create_food_vertices(&self, food: &Food) -> Vec<Vertex> {
        let mut vertices = Vec::new();
        let radius = 10.0;
        let segments = 12;
        
        let color = [0.0, 1.0, 0.0]; // Зеленый цвет для еды
        
        // Нормализуем позицию в координаты экрана (-1 до 1)
        let x = (food.position.x / self.size.width as f32) * 2.0 - 1.0;
        let y = 1.0 - (food.position.y / self.size.height as f32) * 2.0;
        
        // Создаем круг из треугольников
        for i in 0..segments {
            let angle1 = (i as f32 / segments as f32) * std::f32::consts::PI * 2.0;
            let angle2 = ((i + 1) as f32 / segments as f32) * std::f32::consts::PI * 2.0;
            
            let radius_norm = radius / (self.size.width.min(self.size.height) as f32);
            
            // Центр
            vertices.push(Vertex {
                position: [x, y],
                color: color,
            });
            // Первая точка на окружности
            vertices.push(Vertex {
                position: [
                    x + angle1.cos() * radius_norm,
                    y + angle1.sin() * radius_norm,
                ],
                color: color,
            });
            // Вторая точка на окружности
            vertices.push(Vertex {
                position: [
                    x + angle2.cos() * radius_norm,
                    y + angle2.sin() * radius_norm,
                ],
                color: color,
            });
        }
        
        vertices
    }

    fn hsv_to_rgb(&self, h: f32, s: f32, v: f32) -> [f32; 3] {
        let c = v * s;
        let x = c * (1.0 - ((h * 6.0) % 2.0 - 1.0).abs());
        let m = v - c;
        
        let (r, g, b) = if h < 1.0/6.0 {
            (c, x, 0.0)
        } else if h < 2.0/6.0 {
            (x, c, 0.0)
        } else if h < 3.0/6.0 {
            (0.0, c, x)
        } else if h < 4.0/6.0 {
            (0.0, x, c)
        } else if h < 5.0/6.0 {
            (x, 0.0, c)
        } else {
            (c, 0.0, x)
        };
        
        [r + m, g + m, b + m]
    }

    fn create_slider_vertices(&self) -> Vec<Vertex> {
        let mut vertices = Vec::new();
        
        // Позиция ползунка в правом верхнем углу
        let slider_x = self.size.width as f32 - 150.0;
        let slider_y = 20.0;
        let slider_width = 120.0;
        let slider_height = 20.0;
        
        // Нормализуем координаты в -1..1
        let x1 = (slider_x / self.size.width as f32) * 2.0 - 1.0;
        let x2 = ((slider_x + slider_width) / self.size.width as f32) * 2.0 - 1.0;
        let y1 = 1.0 - (slider_y / self.size.height as f32) * 2.0;
        let y2 = 1.0 - ((slider_y + slider_height) / self.size.height as f32) * 2.0;
        
        // Фон ползунка (темно-серый)
        let bg_color = [0.3, 0.3, 0.3];
        vertices.push(Vertex { position: [x1, y1], color: bg_color });
        vertices.push(Vertex { position: [x2, y1], color: bg_color });
        vertices.push(Vertex { position: [x1, y2], color: bg_color });
        vertices.push(Vertex { position: [x2, y1], color: bg_color });
        vertices.push(Vertex { position: [x2, y2], color: bg_color });
        vertices.push(Vertex { position: [x1, y2], color: bg_color });
        
        // Индикатор текущего значения (зеленый)
        let indicator_pos = ((self.time_scale - 1.0) / 499.0).min(1.0).max(0.0); // 0..1 для диапазона 1-500
        let indicator_x = x1 + (x2 - x1) * indicator_pos;
        let indicator_width = 0.01;
        let indicator_color = [0.0, 1.0, 0.0];
        
        vertices.push(Vertex { position: [indicator_x - indicator_width, y1], color: indicator_color });
        vertices.push(Vertex { position: [indicator_x + indicator_width, y1], color: indicator_color });
        vertices.push(Vertex { position: [indicator_x - indicator_width, y2], color: indicator_color });
        vertices.push(Vertex { position: [indicator_x + indicator_width, y1], color: indicator_color });
        vertices.push(Vertex { position: [indicator_x + indicator_width, y2], color: indicator_color });
        vertices.push(Vertex { position: [indicator_x - indicator_width, y2], color: indicator_color });
        
        vertices
    }

    fn create_stats_text_vertices(&self) -> Vec<Vertex> {
        // Улучшенная визуализация статистики с подписями и легендой
        let mut vertices = Vec::new();
        
        let stats_x = 10.0;
        let stats_y = 10.0;
        let bar_width = 280.0;
        let bar_height = 20.0;
        let spacing = 28.0;
        let legend_size = 12.0; // Размер цветных квадратиков легенды
        
        // Нормализуем координаты
        let x1 = (stats_x / self.size.width as f32) * 2.0 - 1.0;
        let x2 = ((stats_x + bar_width) / self.size.width as f32) * 2.0 - 1.0;
        
        // Фон для статистики (темный с рамкой)
        let bg_x1 = x1 - 0.025;
        let bg_x2 = x2 + 0.025;
        let bg_y1 = 1.0 - ((stats_y - 8.0) / self.size.height as f32) * 2.0;
        let bg_y2 = 1.0 - ((stats_y + spacing * 5.5 + 8.0) / self.size.height as f32) * 2.0;
        let bg_color = [0.1, 0.1, 0.1];
        vertices.push(Vertex { position: [bg_x1, bg_y1], color: bg_color });
        vertices.push(Vertex { position: [bg_x2, bg_y1], color: bg_color });
        vertices.push(Vertex { position: [bg_x1, bg_y2], color: bg_color });
        vertices.push(Vertex { position: [bg_x2, bg_y1], color: bg_color });
        vertices.push(Vertex { position: [bg_x2, bg_y2], color: bg_color });
        vertices.push(Vertex { position: [bg_x1, bg_y2], color: bg_color });
        
        // Рамка вокруг статистики
        let border_color = [0.5, 0.5, 0.5];
        let border_width = 0.003;
        // Верх
        vertices.push(Vertex { position: [bg_x1, bg_y1], color: border_color });
        vertices.push(Vertex { position: [bg_x2, bg_y1], color: border_color });
        vertices.push(Vertex { position: [bg_x1, bg_y1 + border_width], color: border_color });
        vertices.push(Vertex { position: [bg_x2, bg_y1], color: border_color });
        vertices.push(Vertex { position: [bg_x2, bg_y1 + border_width], color: border_color });
        vertices.push(Vertex { position: [bg_x1, bg_y1 + border_width], color: border_color });
        // Низ
        vertices.push(Vertex { position: [bg_x1, bg_y2 - border_width], color: border_color });
        vertices.push(Vertex { position: [bg_x2, bg_y2 - border_width], color: border_color });
        vertices.push(Vertex { position: [bg_x1, bg_y2], color: border_color });
        vertices.push(Vertex { position: [bg_x2, bg_y2 - border_width], color: border_color });
        vertices.push(Vertex { position: [bg_x2, bg_y2], color: border_color });
        vertices.push(Vertex { position: [bg_x1, bg_y2], color: border_color });
        // Лево
        vertices.push(Vertex { position: [bg_x1, bg_y1], color: border_color });
        vertices.push(Vertex { position: [bg_x1 + border_width, bg_y1], color: border_color });
        vertices.push(Vertex { position: [bg_x1, bg_y2], color: border_color });
        vertices.push(Vertex { position: [bg_x1 + border_width, bg_y1], color: border_color });
        vertices.push(Vertex { position: [bg_x1 + border_width, bg_y2], color: border_color });
        vertices.push(Vertex { position: [bg_x1, bg_y2], color: border_color });
        // Право
        vertices.push(Vertex { position: [bg_x2 - border_width, bg_y1], color: border_color });
        vertices.push(Vertex { position: [bg_x2, bg_y1], color: border_color });
        vertices.push(Vertex { position: [bg_x2 - border_width, bg_y2], color: border_color });
        vertices.push(Vertex { position: [bg_x2, bg_y1], color: border_color });
        vertices.push(Vertex { position: [bg_x2, bg_y2], color: border_color });
        vertices.push(Vertex { position: [bg_x2 - border_width, bg_y2], color: border_color });
        
        // Легенда (цветные квадратики слева от полос)
        let legend_x = x1 - 0.04;
        let legend_size_norm = legend_size / self.size.width as f32;
        
        // Видов (красный)
        let mut y_offset = 0.0;
        let y1 = 1.0 - ((stats_y + y_offset) / self.size.height as f32) * 2.0;
        let y2 = 1.0 - ((stats_y + y_offset + bar_height) / self.size.height as f32) * 2.0;
        let y_center = (y1 + y2) / 2.0;
        let species_ratio = (self.stats.species_count as f32 / 30.0).min(1.0);
        let species_x2 = x1 + (x2 - x1) * species_ratio;
        let species_color = [1.0, 0.3, 0.3];
        
        // Легенда (квадратик)
        let legend_y1 = y_center - legend_size_norm;
        let legend_y2 = y_center + legend_size_norm;
        vertices.push(Vertex { position: [legend_x, legend_y1], color: species_color });
        vertices.push(Vertex { position: [legend_x + legend_size_norm * 2.0, legend_y1], color: species_color });
        vertices.push(Vertex { position: [legend_x, legend_y2], color: species_color });
        vertices.push(Vertex { position: [legend_x + legend_size_norm * 2.0, legend_y1], color: species_color });
        vertices.push(Vertex { position: [legend_x + legend_size_norm * 2.0, legend_y2], color: species_color });
        vertices.push(Vertex { position: [legend_x, legend_y2], color: species_color });
        
        // Полоса
        vertices.push(Vertex { position: [x1, y1], color: species_color });
        vertices.push(Vertex { position: [species_x2, y1], color: species_color });
        vertices.push(Vertex { position: [x1, y2], color: species_color });
        vertices.push(Vertex { position: [species_x2, y1], color: species_color });
        vertices.push(Vertex { position: [species_x2, y2], color: species_color });
        vertices.push(Vertex { position: [x1, y2], color: species_color });
        
        // Популяция (синий)
        y_offset += spacing;
        let y1 = 1.0 - ((stats_y + y_offset) / self.size.height as f32) * 2.0;
        let y2 = 1.0 - ((stats_y + y_offset + bar_height) / self.size.height as f32) * 2.0;
        let y_center = (y1 + y2) / 2.0;
        let pop_ratio = (self.stats.alive_creatures as f32 / 300.0).min(1.0);
        let pop_x2 = x1 + (x2 - x1) * pop_ratio;
        let pop_color = [0.3, 0.3, 1.0];
        
        // Легенда
        let legend_y1 = y_center - legend_size_norm;
        let legend_y2 = y_center + legend_size_norm;
        vertices.push(Vertex { position: [legend_x, legend_y1], color: pop_color });
        vertices.push(Vertex { position: [legend_x + legend_size_norm * 2.0, legend_y1], color: pop_color });
        vertices.push(Vertex { position: [legend_x, legend_y2], color: pop_color });
        vertices.push(Vertex { position: [legend_x + legend_size_norm * 2.0, legend_y1], color: pop_color });
        vertices.push(Vertex { position: [legend_x + legend_size_norm * 2.0, legend_y2], color: pop_color });
        vertices.push(Vertex { position: [legend_x, legend_y2], color: pop_color });
        
        // Полоса
        vertices.push(Vertex { position: [x1, y1], color: pop_color });
        vertices.push(Vertex { position: [pop_x2, y1], color: pop_color });
        vertices.push(Vertex { position: [x1, y2], color: pop_color });
        vertices.push(Vertex { position: [pop_x2, y1], color: pop_color });
        vertices.push(Vertex { position: [pop_x2, y2], color: pop_color });
        vertices.push(Vertex { position: [x1, y2], color: pop_color });
        
        // Средняя энергия (зеленый)
        y_offset += spacing;
        let y1 = 1.0 - ((stats_y + y_offset) / self.size.height as f32) * 2.0;
        let y2 = 1.0 - ((stats_y + y_offset + bar_height) / self.size.height as f32) * 2.0;
        let y_center = (y1 + y2) / 2.0;
        let energy_ratio = (self.stats.average_energy / 100.0).min(1.0);
        let energy_x2 = x1 + (x2 - x1) * energy_ratio;
        let energy_color = [0.3, 1.0, 0.3];
        
        // Легенда
        let legend_y1 = y_center - legend_size_norm;
        let legend_y2 = y_center + legend_size_norm;
        vertices.push(Vertex { position: [legend_x, legend_y1], color: energy_color });
        vertices.push(Vertex { position: [legend_x + legend_size_norm * 2.0, legend_y1], color: energy_color });
        vertices.push(Vertex { position: [legend_x, legend_y2], color: energy_color });
        vertices.push(Vertex { position: [legend_x + legend_size_norm * 2.0, legend_y1], color: energy_color });
        vertices.push(Vertex { position: [legend_x + legend_size_norm * 2.0, legend_y2], color: energy_color });
        vertices.push(Vertex { position: [legend_x, legend_y2], color: energy_color });
        
        // Полоса
        vertices.push(Vertex { position: [x1, y1], color: energy_color });
        vertices.push(Vertex { position: [energy_x2, y1], color: energy_color });
        vertices.push(Vertex { position: [x1, y2], color: energy_color });
        vertices.push(Vertex { position: [energy_x2, y1], color: energy_color });
        vertices.push(Vertex { position: [energy_x2, y2], color: energy_color });
        vertices.push(Vertex { position: [x1, y2], color: energy_color });
        
        // Еда (желтый)
        y_offset += spacing;
        let y1 = 1.0 - ((stats_y + y_offset) / self.size.height as f32) * 2.0;
        let y2 = 1.0 - ((stats_y + y_offset + bar_height) / self.size.height as f32) * 2.0;
        let y_center = (y1 + y2) / 2.0;
        let food_ratio = (self.stats.total_food as f32 / 50.0).min(1.0);
        let food_x2 = x1 + (x2 - x1) * food_ratio;
        let food_color = [1.0, 1.0, 0.3];
        
        // Легенда
        let legend_y1 = y_center - legend_size_norm;
        let legend_y2 = y_center + legend_size_norm;
        vertices.push(Vertex { position: [legend_x, legend_y1], color: food_color });
        vertices.push(Vertex { position: [legend_x + legend_size_norm * 2.0, legend_y1], color: food_color });
        vertices.push(Vertex { position: [legend_x, legend_y2], color: food_color });
        vertices.push(Vertex { position: [legend_x + legend_size_norm * 2.0, legend_y1], color: food_color });
        vertices.push(Vertex { position: [legend_x + legend_size_norm * 2.0, legend_y2], color: food_color });
        vertices.push(Vertex { position: [legend_x, legend_y2], color: food_color });
        
        // Полоса
        vertices.push(Vertex { position: [x1, y1], color: food_color });
        vertices.push(Vertex { position: [food_x2, y1], color: food_color });
        vertices.push(Vertex { position: [x1, y2], color: food_color });
        vertices.push(Vertex { position: [food_x2, y1], color: food_color });
        vertices.push(Vertex { position: [food_x2, y2], color: food_color });
        vertices.push(Vertex { position: [x1, y2], color: food_color });
        
        // Скорость времени (белый)
        y_offset += spacing;
        let y1 = 1.0 - ((stats_y + y_offset) / self.size.height as f32) * 2.0;
        let y2 = 1.0 - ((stats_y + y_offset + bar_height) / self.size.height as f32) * 2.0;
        let y_center = (y1 + y2) / 2.0;
        let time_ratio = ((self.time_scale - 1.0) / 499.0).min(1.0); // Нормализуем для диапазона 1-500
        let time_x2 = x1 + (x2 - x1) * time_ratio;
        let time_color = [1.0, 1.0, 1.0];
        
        // Легенда
        let legend_y1 = y_center - legend_size_norm;
        let legend_y2 = y_center + legend_size_norm;
        vertices.push(Vertex { position: [legend_x, legend_y1], color: time_color });
        vertices.push(Vertex { position: [legend_x + legend_size_norm * 2.0, legend_y1], color: time_color });
        vertices.push(Vertex { position: [legend_x, legend_y2], color: time_color });
        vertices.push(Vertex { position: [legend_x + legend_size_norm * 2.0, legend_y1], color: time_color });
        vertices.push(Vertex { position: [legend_x + legend_size_norm * 2.0, legend_y2], color: time_color });
        vertices.push(Vertex { position: [legend_x, legend_y2], color: time_color });
        
        // Полоса
        vertices.push(Vertex { position: [x1, y1], color: time_color });
        vertices.push(Vertex { position: [time_x2, y1], color: time_color });
        vertices.push(Vertex { position: [x1, y2], color: time_color });
        vertices.push(Vertex { position: [time_x2, y1], color: time_color });
        vertices.push(Vertex { position: [time_x2, y2], color: time_color });
        vertices.push(Vertex { position: [x1, y2], color: time_color });
        
        vertices
    }

    fn create_evolution_graph_vertices(&self) -> Vec<Vertex> {
        let mut vertices = Vec::new();
        
        if self.stats.history.len() < 2 {
            return vertices;
        }
        
        // График в правом нижнем углу
        let graph_x = self.size.width as f32 - 320.0;
        let graph_y = self.size.height as f32 - 220.0;
        let graph_width = 300.0;
        let graph_height = 200.0;
        
        // Нормализуем координаты
        let x1 = (graph_x / self.size.width as f32) * 2.0 - 1.0;
        let x2 = ((graph_x + graph_width) / self.size.width as f32) * 2.0 - 1.0;
        let y1 = 1.0 - ((graph_y + graph_height) / self.size.height as f32) * 2.0;
        let y2 = 1.0 - (graph_y / self.size.height as f32) * 2.0;
        
        // Фон графика (темный)
        let bg_color = [0.15, 0.15, 0.15];
        vertices.push(Vertex { position: [x1, y1], color: bg_color });
        vertices.push(Vertex { position: [x2, y1], color: bg_color });
        vertices.push(Vertex { position: [x1, y2], color: bg_color });
        vertices.push(Vertex { position: [x2, y1], color: bg_color });
        vertices.push(Vertex { position: [x2, y2], color: bg_color });
        vertices.push(Vertex { position: [x1, y2], color: bg_color });
        
        // Сетка графика (светло-серые линии)
        let grid_color = [0.3, 0.3, 0.3];
        let grid_line_width = 0.001;
        // Горизонтальные линии (5 линий)
        for i in 0..=5 {
            let grid_y = y1 + (y2 - y1) * (i as f32 / 5.0);
            vertices.push(Vertex { position: [x1, grid_y], color: grid_color });
            vertices.push(Vertex { position: [x2, grid_y], color: grid_color });
            vertices.push(Vertex { position: [x1, grid_y + grid_line_width], color: grid_color });
            vertices.push(Vertex { position: [x2, grid_y], color: grid_color });
            vertices.push(Vertex { position: [x2, grid_y + grid_line_width], color: grid_color });
            vertices.push(Vertex { position: [x1, grid_y + grid_line_width], color: grid_color });
        }
        // Вертикальные линии (10 линий)
        for i in 0..=10 {
            let grid_x = x1 + (x2 - x1) * (i as f32 / 10.0);
            vertices.push(Vertex { position: [grid_x, y1], color: grid_color });
            vertices.push(Vertex { position: [grid_x, y2], color: grid_color });
            vertices.push(Vertex { position: [grid_x + grid_line_width, y1], color: grid_color });
            vertices.push(Vertex { position: [grid_x, y2], color: grid_color });
            vertices.push(Vertex { position: [grid_x + grid_line_width, y2], color: grid_color });
            vertices.push(Vertex { position: [grid_x + grid_line_width, y1], color: grid_color });
        }
        
        // Находим максимумы для нормализации
        let max_species = self.stats.history.iter().map(|s| s.species_count).max().unwrap_or(1).max(1);
        let max_pop = self.stats.history.iter().map(|s| s.population).max().unwrap_or(1).max(1);
        
        // График количества видов (красный, более толстая линия)
        let species_color = [1.0, 0.2, 0.2];
        for i in 0..(self.stats.history.len() - 1) {
            let x1_val = x1 + (i as f32 / (self.stats.history.len() - 1) as f32) * (x2 - x1);
            let x2_val = x1 + ((i + 1) as f32 / (self.stats.history.len() - 1) as f32) * (x2 - x1);
            // Нормализуем значения: 0 соответствует y1, max_species соответствует y2
            let y1_val = y1 + (self.stats.history[i].species_count as f32 / max_species as f32) * (y2 - y1);
            let y2_val = y1 + (self.stats.history[i + 1].species_count as f32 / max_species as f32) * (y2 - y1);
            
            // Более толстая линия для лучшей видимости
            let line_width = 0.005;
            let perp_angle = (y2_val - y1_val).atan2(x2_val - x1_val) + std::f32::consts::PI / 2.0;
            let dx = perp_angle.cos() * line_width;
            let dy = perp_angle.sin() * line_width;
            
            vertices.push(Vertex { position: [x1_val + dx, y1_val + dy], color: species_color });
            vertices.push(Vertex { position: [x1_val - dx, y1_val - dy], color: species_color });
            vertices.push(Vertex { position: [x2_val + dx, y2_val + dy], color: species_color });
            vertices.push(Vertex { position: [x1_val - dx, y1_val - dy], color: species_color });
            vertices.push(Vertex { position: [x2_val - dx, y2_val - dy], color: species_color });
            vertices.push(Vertex { position: [x2_val + dx, y2_val + dy], color: species_color });
        }
        
        // График популяции (синий, более толстая линия)
        let pop_color = [0.2, 0.2, 1.0];
        for i in 0..(self.stats.history.len() - 1) {
            let x1_val = x1 + (i as f32 / (self.stats.history.len() - 1) as f32) * (x2 - x1);
            let x2_val = x1 + ((i + 1) as f32 / (self.stats.history.len() - 1) as f32) * (x2 - x1);
            let y1_val = y1 + (self.stats.history[i].population as f32 / max_pop as f32) * (y2 - y1);
            let y2_val = y1 + (self.stats.history[i + 1].population as f32 / max_pop as f32) * (y2 - y1);
            
            let line_width = 0.005;
            let perp_angle = (y2_val - y1_val).atan2(x2_val - x1_val) + std::f32::consts::PI / 2.0;
            let dx = perp_angle.cos() * line_width;
            let dy = perp_angle.sin() * line_width;
            
            vertices.push(Vertex { position: [x1_val + dx, y1_val + dy], color: pop_color });
            vertices.push(Vertex { position: [x1_val - dx, y1_val - dy], color: pop_color });
            vertices.push(Vertex { position: [x2_val + dx, y2_val + dy], color: pop_color });
            vertices.push(Vertex { position: [x1_val - dx, y1_val - dy], color: pop_color });
            vertices.push(Vertex { position: [x2_val - dx, y2_val - dy], color: pop_color });
            vertices.push(Vertex { position: [x2_val + dx, y2_val + dy], color: pop_color });
        }
        
        // Граница графика (белая рамка)
        let border_color = [0.9, 0.9, 0.9];
        let border_width = 0.003;
        // Верх
        vertices.push(Vertex { position: [x1, y1], color: border_color });
        vertices.push(Vertex { position: [x2, y1], color: border_color });
        vertices.push(Vertex { position: [x1, y1 + border_width], color: border_color });
        vertices.push(Vertex { position: [x2, y1], color: border_color });
        vertices.push(Vertex { position: [x2, y1 + border_width], color: border_color });
        vertices.push(Vertex { position: [x1, y1 + border_width], color: border_color });
        // Низ
        vertices.push(Vertex { position: [x1, y2 - border_width], color: border_color });
        vertices.push(Vertex { position: [x2, y2 - border_width], color: border_color });
        vertices.push(Vertex { position: [x1, y2], color: border_color });
        vertices.push(Vertex { position: [x2, y2 - border_width], color: border_color });
        vertices.push(Vertex { position: [x2, y2], color: border_color });
        vertices.push(Vertex { position: [x1, y2], color: border_color });
        // Лево
        vertices.push(Vertex { position: [x1, y1], color: border_color });
        vertices.push(Vertex { position: [x1 + border_width, y1], color: border_color });
        vertices.push(Vertex { position: [x1, y2], color: border_color });
        vertices.push(Vertex { position: [x1 + border_width, y1], color: border_color });
        vertices.push(Vertex { position: [x1 + border_width, y2], color: border_color });
        vertices.push(Vertex { position: [x1, y2], color: border_color });
        // Право
        vertices.push(Vertex { position: [x2 - border_width, y1], color: border_color });
        vertices.push(Vertex { position: [x2, y1], color: border_color });
        vertices.push(Vertex { position: [x2 - border_width, y2], color: border_color });
        vertices.push(Vertex { position: [x2, y1], color: border_color });
        vertices.push(Vertex { position: [x2, y2], color: border_color });
        vertices.push(Vertex { position: [x2 - border_width, y2], color: border_color });
        
        vertices
    }
}

