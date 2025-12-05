use hecs::World;
use tokio::runtime::Runtime;
use tokio::sync::mpsc;
use log::info;

pub struct GameWorld {
    pub ecs: World,
    pub runtime: Runtime,
    pub chunk_receiver: mpsc::Receiver<ChunkData>,
    pub chunk_sender: mpsc::Sender<ChunkData>,
}

pub struct ChunkData {
    pub id: u32,
    // Mesh data, physics data, etc.
}

impl GameWorld {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel(10);
        Self {
            ecs: World::new(),
            runtime: Runtime::new().unwrap(),
            chunk_receiver: rx,
            chunk_sender: tx,
        }
    }

    pub fn update_streaming(&mut self) {
        // Check for loaded chunks
        while let Ok(chunk) = self.chunk_receiver.try_recv() {
            info!("Chunk {} loaded", chunk.id);
            // Add to ECS
            // self.ecs.spawn((chunk.mesh, chunk.rigid_body));
        }

        // Simulate requesting a chunk (e.g. based on player position)
        // In a real game, this would check distance and only request if not loaded
        // For demo, we just spawn a task occasionally or once
        
        // Example: Spawn a task to load chunk 1
        // self.request_chunk(1);
    }

    pub fn request_chunk(&self, chunk_id: u32) {
        let tx = self.chunk_sender.clone();
        self.runtime.spawn(async move {
            // Simulate IO
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            
            // "Load" data
            let chunk = ChunkData { id: chunk_id };
            
            // Send back
            let _ = tx.send(chunk).await;
        });
    }
}
