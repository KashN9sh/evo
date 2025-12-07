use bevy::prelude::*;

#[derive(Resource)]
pub struct SimulationStats {
    pub generation: u32,
    pub best_fitness: f32,
    pub average_fitness: f32,
    pub distance_to_target: f32,
}

impl Default for SimulationStats {
    fn default() -> Self {
        Self {
            generation: 0,
            best_fitness: 0.0,
            average_fitness: 0.0,
            distance_to_target: f32::MAX,
        }
    }
}

pub fn setup_ui(mut commands: Commands) {
    commands
        .spawn(NodeBundle {
            style: Style {
                position_type: PositionType::Absolute,
                top: Val::Px(10.0),
                left: Val::Px(10.0),
                ..default()
            },
            background_color: Color::rgba(0.0, 0.0, 0.0, 0.5).into(),
            ..default()
        })
        .with_children(|parent| {
            parent.spawn(TextBundle::from_section(
                "Evo",
                TextStyle {
                    font_size: 20.0,
                    color: Color::WHITE,
                    ..default()
                },
            ));
        });
}

pub fn update_ui_stats(stats: Res<SimulationStats>, mut query: Query<&mut Text>) {
    if let Ok(mut text) = query.get_single_mut() {
        text.sections[0].value = format!(
            "Gen: {}\nBest fit: {:.2}\nAw fit: {:.2}\nDistance to target: {:.2}",
            stats.generation, stats.best_fitness, stats.average_fitness, stats.distance_to_target
        );
    }
}
