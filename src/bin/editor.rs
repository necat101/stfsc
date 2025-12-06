use eframe::egui;
use std::net::TcpStream;
use std::io::Write;
use stfsc_engine::world::SceneUpdate;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "STFSC Editor",
        options,
        Box::new(|_cc| Box::new(MyApp::new())),
    )
}

enum AppCommand {
    Connect(String),
    Send(SceneUpdate),
}

enum AppEvent {
    Connected,
    ConnectionError(String),
    SendError(String),
    StatusUpdate(String),
}

struct MyApp {
    ip: String,
    status: String,
    command_tx: Sender<AppCommand>,
    event_rx: Receiver<AppEvent>,
    is_connected: bool,
}

impl MyApp {
    fn new() -> Self {
        let (command_tx, command_rx) = channel::<AppCommand>();
        let (event_tx, event_rx) = channel::<AppEvent>();

        thread::spawn(move || {
            let mut stream: Option<TcpStream> = None;

            while let Ok(cmd) = command_rx.recv() {
                match cmd {
                    AppCommand::Connect(ip) => {
                        let _ = event_tx.send(AppEvent::StatusUpdate(format!("Attempting to connect to {}...", ip)));
                        
                        // Parse IP and port
                        let addr_result = if ip.contains(':') {
                            ip.parse()
                        } else {
                            format!("{}:8080", ip).parse()
                        };

                        match addr_result {
                            Ok(addr) => {
                                match TcpStream::connect_timeout(&addr, std::time::Duration::from_secs(5)) {
                                    Ok(s) => {
                                        stream = Some(s);
                                        let _ = event_tx.send(AppEvent::Connected);
                                    }
                                    Err(e) => {
                                        let _ = event_tx.send(AppEvent::ConnectionError(format!("Failed to connect: {}", e)));
                                    }
                                }
                            }
                            Err(e) => {
                                let _ = event_tx.send(AppEvent::ConnectionError(format!("Invalid IP address: {}", e)));
                            }
                        }
                    }
                    AppCommand::Send(update) => {
                        if let Some(s) = &mut stream {
                            let bytes = bincode::serialize(&update).unwrap();
                            let len = bytes.len() as u32;
                            
                            let mut success = true;
                            if let Err(e) = s.write_all(&len.to_le_bytes()) {
                                let _ = event_tx.send(AppEvent::SendError(e.to_string()));
                                success = false;
                            }
                            if success {
                                if let Err(e) = s.write_all(&bytes) {
                                    let _ = event_tx.send(AppEvent::SendError(e.to_string()));
                                }
                            }
                        } else {
                            let _ = event_tx.send(AppEvent::SendError("Not connected".to_string()));
                        }
                    }
                }
            }
        });

        Self {
            ip: "127.0.0.1:8080".to_owned(),
            status: "Disconnected".to_owned(),
            command_tx,
            event_rx,
            is_connected: false,
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Poll for events from background thread
        while let Ok(event) = self.event_rx.try_recv() {
            match event {
                AppEvent::Connected => {
                    self.status = "Connected".to_owned();
                    self.is_connected = true;
                }
                AppEvent::ConnectionError(e) => {
                    self.status = format!("Connection Error: {}", e);
                    self.is_connected = false;
                }
                AppEvent::SendError(e) => {
                    self.status = format!("Send Error: {}", e);
                }
                AppEvent::StatusUpdate(msg) => {
                    self.status = msg;
                }
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("STFSC Editor");
            
            ui.horizontal(|ui| {
                ui.label("Quest IP:");
                ui.text_edit_singleline(&mut self.ip);
                if ui.button("Connect").clicked() {
                    self.status = "Connecting...".to_owned();
                    let _ = self.command_tx.send(AppCommand::Connect(self.ip.clone()));
                }
            });
            
            ui.label(&self.status);
            
            if self.is_connected {
                if ui.button("Spawn Cube").clicked() {
                    let update = SceneUpdate::Spawn {
                        id: rand::random(),
                        position: [0.0, 0.0, -2.0],
                        color: [1.0, 1.0, 1.0],
                    };
                    let _ = self.command_tx.send(AppCommand::Send(update));
                }
            }
        });
        
        // Request repaint to ensure we poll events frequently
        ctx.request_repaint();
    }
}
