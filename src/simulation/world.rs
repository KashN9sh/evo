use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

pub fn spawn_ground(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    // Размеры пола
    let ground_size = 100.0;
    let ground_height = 0.1;
    
    // Создаем визуализацию пола (большой плоский куб)
    // Создаем простой меш куба через вершины
    let mut mesh = Mesh::new(bevy::render::mesh::PrimitiveTopology::TriangleList);
    
    let half_size = ground_size / 2.0;
    let half_height = ground_height / 2.0;
    
    // Вершины куба (8 вершин)
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_POSITION,
        vec![
            // Нижняя грань
            [-half_size, -half_height, -half_size],
            [half_size, -half_height, -half_size],
            [half_size, -half_height, half_size],
            [-half_size, -half_height, half_size],
            // Верхняя грань
            [-half_size, half_height, -half_size],
            [half_size, half_height, -half_size],
            [half_size, half_height, half_size],
            [-half_size, half_height, half_size],
        ],
    );
    
    // Индексы для треугольников
    mesh.set_indices(Some(bevy::render::mesh::Indices::U32(vec![
        // Нижняя грань
        0, 1, 2, 0, 2, 3,
        // Верхняя грань
        4, 6, 5, 4, 7, 6,
        // Боковые грани
        0, 4, 5, 0, 5, 1,
        1, 5, 6, 1, 6, 2,
        2, 6, 7, 2, 7, 3,
        3, 7, 4, 3, 4, 0,
    ])));
    
    // Добавляем нормали (все направлены вверх для верхней грани)
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_NORMAL,
        vec![
            [0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, -1.0, 0.0], // Нижняя
            [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], // Верхняя
        ],
    );
    
    // Добавляем UV координаты для текстуры
    // Масштабируем UV для создания нужного размера клеток на полу
    let uv_scale = ground_size / 10.0; // Клетки будут повторяться каждые 10 единиц
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_UV_0,
        vec![
            // Нижняя грань (не видна, но нужны для корректного меша)
            [0.0, 0.0], [uv_scale, 0.0], [uv_scale, uv_scale], [0.0, uv_scale],
            // Верхняя грань (видимая) - повторяем паттерн
            [0.0, 0.0], [uv_scale, 0.0], [uv_scale, uv_scale], [0.0, uv_scale],
        ],
    );
    
    let ground_mesh = meshes.add(mesh);
    
    // Создаем текстуру с клетчатым паттерном
    let checkerboard_size = 512;
    let cell_size = 32; // Размер одной клетки в пикселях
    let mut image_data = Vec::new();
    
    for y in 0..checkerboard_size {
        for x in 0..checkerboard_size {
            let cell_x = (x / cell_size) % 2;
            let cell_y = (y / cell_size) % 2;
            let is_blue = (cell_x + cell_y) % 2 == 0;
            
            if is_blue {
                // Синяя клетка
                image_data.extend_from_slice(&[0, 100, 200, 255]); // RGBA
            } else {
                // Светло-синяя/белая клетка
                image_data.extend_from_slice(&[150, 200, 255, 255]); // RGBA
            }
        }
    }
    
    let image = Image::new(
        bevy::render::render_resource::Extent3d {
            width: checkerboard_size,
            height: checkerboard_size,
            depth_or_array_layers: 1,
        },
        bevy::render::render_resource::TextureDimension::D2,
        image_data,
        bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb,
    );
    
    let texture_handle = images.add(image);
    
    let ground_material = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        base_color_texture: Some(texture_handle),
        perceptual_roughness: 0.8,
        metallic: 0.0,
        ..default()
    });
    
    // Создаем пол с физикой и визуализацией
    // Позиция пола: верхняя поверхность на y = 0.0
    // Центр пола должен быть на y = ground_height / 2.0, чтобы верх был на 0.0
    let ground_y = ground_height / 2.0;
    
    commands.spawn((
        RigidBody::Fixed,
        Collider::cuboid(ground_size / 2.0, ground_height / 2.0, ground_size / 2.0),
        PbrBundle {
            mesh: ground_mesh,
            material: ground_material,
            transform: Transform::from_translation(Vec3::new(0.0, ground_y, 0.0)),
            ..default()
        },
    ));
}

