# Running the Multilingual Voice Chat in Docker

This guide explains how to run the Multilingual Voice Chat application in a Docker container, which resolves any Python version or dependency issues.

## Prerequisites

1. Docker installed on your system
   - [Install Docker](https://docs.docker.com/get-docker/)

2. For GPU acceleration: NVIDIA Container Toolkit (for GPU support)
   - [Install NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

## Quick Start

The easiest way to run the application is using the provided script:

```bash
chmod +x run_docker.sh
./run_docker.sh
```

This script will:
1. Check if Docker is installed
2. Detect if NVIDIA GPU support is available
3. Build and start the container
4. Make the application available at http://localhost:7860

## Manual Setup

If you prefer to run the commands manually:

### Building the Docker Image

```bash
docker build -t voice-chat .
```

### Running with GPU Support

```bash
docker run --gpus all -p 7860:7860 -v $(pwd)/models:/app/models -v $(pwd)/user_profiles.json:/app/user_profiles.json voice-chat
```

### Running without GPU (CPU Only)

```bash
docker run -p 7860:7860 -v $(pwd)/models:/app/models -v $(pwd)/user_profiles.json:/app/user_profiles.json voice-chat
```

## Using Docker Compose

You can also use Docker Compose to run the application:

### With GPU Support

```bash
docker-compose up
```

### Without GPU (CPU Only)

You'll need to modify the `docker-compose.yml` file first to remove the GPU-specific sections:

1. Remove or comment out the `deploy` section with the NVIDIA device
2. Remove the `NVIDIA_VISIBLE_DEVICES` environment variable

Then run:

```bash
docker-compose up
```

## Accessing the Application

Once the container is running, access the application at:

http://localhost:7860

## Persistent Storage

The following directories/files are mounted as volumes for persistent storage:

- `./models`: Caches downloaded AI models
- `./user_profiles.json`: Stores user session data

## Troubleshooting

### GPU Issues

If you encounter GPU-related errors:

1. Ensure NVIDIA drivers are installed and up-to-date
2. Verify NVIDIA Container Toolkit is properly installed
3. Run `nvidia-smi` to check if your GPU is recognized by the system
4. Try running in CPU-only mode by removing GPU requirements

### Port Conflicts

If port 7860 is already in use, you can change it in the `docker-compose.yml` file or use:

```bash
docker run -p <your_port>:7860 voice-chat
```