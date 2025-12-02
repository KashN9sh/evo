use winit::{
    event::{Event, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowAttributes},
};

mod creature;
mod neural;
mod sensors;
mod metabolism;
mod biomechanics;
mod evolution;
mod renderer;
mod physics;
mod food;
mod lifecycle;
mod simulation;

fn main() {
    env_logger::init();
    
    let event_loop = EventLoop::new().unwrap();
    let mut app_handler = AppHandler {
        state: None,
        window: None,
    };

    event_loop.run_app(&mut app_handler).unwrap();
}

struct AppHandler {
    state: Option<simulation::SimulationState>,
    window: Option<Window>,
}

impl winit::application::ApplicationHandler for AppHandler {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window_attributes = WindowAttributes::default()
                .with_title("Эволюционная симуляция")
                .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0));
            let win = event_loop.create_window(window_attributes).unwrap();
            let state = pollster::block_on(simulation::SimulationState::new(&win));
            self.window = Some(win);
            self.state = Some(state);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: winit::window::WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(physical_size) => {
                if let Some(ref mut state) = self.state {
                    state.resize(physical_size);
                }
            }
            WindowEvent::RedrawRequested => {
                if let (Some(ref mut state), Some(ref win)) = (self.state.as_mut(), self.window.as_ref()) {
                    state.update();
                    match state.render() {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost) => {
                            let size = state.size();
                            state.resize(size);
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                        Err(e) => eprintln!("{:?}", e),
                    }
                    win.request_redraw();
                }
            }
            _ => {
                // Передаем события в state для обработки ползунка
                if let (Some(ref mut state), Some(ref win)) = (self.state.as_mut(), self.window.as_ref()) {
                    if state.input(&event) {
                        win.request_redraw();
                    }
                }
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(ref win) = self.window {
            win.request_redraw();
        }
    }
}

