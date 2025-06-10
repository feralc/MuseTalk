# MuseTalk gRPC Service

This is a real-time lip sync service that uses MuseTalk to generate video frames from audio input. The service is implemented as a gRPC server that can be integrated with WebRTC applications.

## Prerequisites

- Python 3.8+
- Go 1.16+
- CUDA-capable GPU (recommended)
- FFmpeg

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements-grpc.txt
```

2. Install Go dependencies:
```bash
go get google.golang.org/grpc
go get google.golang.org/protobuf
```

3. Generate gRPC code:
```bash
# Generate Python code
python -m grpc_tools.protoc -I./proto --python_out=. --grpc_python_out=. ./proto/lipsync.proto

# Generate Go code
protoc --go_out=. --go_opt=paths=source_relative \
    --go-grpc_out=. --go-grpc_opt=paths=source_relative \
    ./proto/lipsync.proto
```

## Configuration

1. Update the configuration file at `configs/grpc_server.yaml` with your settings:
   - Set the correct paths to your model files
   - Configure the avatar ID and other parameters
   - Adjust the gRPC port if needed

2. Make sure you have the required model files:
   - MuseTalk model files in `./models/musetalk/`
   - Whisper model in `./models/whisper/`

## Running the Service

1. Start the gRPC server:
```bash
python server/grpc_server.py --config configs/grpc_server.yaml
```

2. The server will start listening on the configured port (default: 50051)

## Integration with WebRTC

To integrate with your Go WebRTC server:

1. Create a gRPC client in your Go code:
```go
conn, err := grpc.Dial("localhost:50051", grpc.WithTransportCredentials(insecure.NewCredentials()))
if err != nil {
    log.Fatalf("Failed to connect: %v", err)
}
defer conn.Close()

client := pb.NewLipSyncServiceClient(conn)
```

2. When receiving audio from WebRTC:
```go
// Create stream
stream, err := client.StreamAudioToVideo(context.Background())
if err != nil {
    log.Fatalf("Failed to create stream: %v", err)
}

// Send audio chunks
chunk := &pb.AudioChunk{
    AudioData:  audioData,
    SampleRate: 16000,
    Channels:   1,
}
if err := stream.Send(chunk); err != nil {
    log.Printf("Failed to send audio chunk: %v", err)
}

// Receive video frames
response, err := stream.Recv()
if err != nil {
    log.Printf("Error receiving frame: %v", err)
}
// Use response.FrameData as your video frame
```

## Testing

You can test the service using the provided Go client:

```bash
go run client/go/main.go --server localhost:50051 --audio path/to/audio.wav --output output_dir
```

## Performance Considerations

1. The service is optimized for real-time processing but requires a GPU for best performance
2. Adjust the batch size and other parameters in the configuration file based on your hardware
3. Consider using a connection pool if you need to handle multiple clients
4. Monitor memory usage as video frames are generated

## Troubleshooting

1. If you get CUDA out of memory errors:
   - Reduce the batch size
   - Use a smaller model
   - Process fewer frames simultaneously

2. If you experience high latency:
   - Check your GPU utilization
   - Adjust the audio chunk size
   - Consider using a more powerful GPU

3. If frames are missing or delayed:
   - Check network connectivity
   - Monitor server CPU and memory usage
   - Adjust buffer sizes in the configuration 