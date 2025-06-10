package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"time"

	pb "github.com/yourusername/musetalk/proto" // Update this import path
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func main() {
	serverAddr := flag.String("server", "localhost:50051", "The server address in the format of host:port")
	audioFile := flag.String("audio", "", "Path to the audio file to stream")
	outputDir := flag.String("output", "output", "Directory to save video frames")
	flag.Parse()

	// Create output directory if it doesn't exist
	if err := os.MkdirAll(*outputDir, 0755); err != nil {
		log.Fatalf("Failed to create output directory: %v", err)
	}

	// Set up connection to server
	conn, err := grpc.Dial(*serverAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("Failed to connect: %v", err)
	}
	defer conn.Close()

	// Create client
	client := pb.NewLipSyncServiceClient(conn)

	// Open audio file
	audioData, err := os.ReadFile(*audioFile)
	if err != nil {
		log.Fatalf("Failed to read audio file: %v", err)
	}

	// Create stream
	stream, err := client.StreamAudioToVideo(context.Background())
	if err != nil {
		log.Fatalf("Failed to create stream: %v", err)
	}

	// Create a channel to signal when we're done sending
	done := make(chan bool)

	// Start goroutine to receive video frames
	go func() {
		frameCount := 0
		for {
			response, err := stream.Recv()
			if err == io.EOF {
				done <- true
				return
			}
			if err != nil {
				log.Printf("Error receiving frame: %v", err)
				done <- true
				return
			}

			// Save frame to file
			framePath := fmt.Sprintf("%s/frame_%04d.jpg", *outputDir, frameCount)
			if err := os.WriteFile(framePath, response.FrameData, 0644); err != nil {
				log.Printf("Error saving frame: %v", err)
			}
			frameCount++
		}
	}()

	// Send audio chunks
	chunkSize := 16000 // 1 second of audio at 16kHz
	for i := 0; i < len(audioData); i += chunkSize {
		end := i + chunkSize
		if end > len(audioData) {
			end = len(audioData)
		}

		chunk := &pb.AudioChunk{
			AudioData:  audioData[i:end],
			SampleRate: 16000,
			Channels:   1,
		}

		if err := stream.Send(chunk); err != nil {
			log.Fatalf("Failed to send audio chunk: %v", err)
		}

		// Add a small delay to simulate real-time streaming
		time.Sleep(time.Second)
	}

	// Close the send direction of the stream
	if err := stream.CloseSend(); err != nil {
		log.Fatalf("Failed to close stream: %v", err)
	}

	// Wait for receiving goroutine to finish
	<-done
}
