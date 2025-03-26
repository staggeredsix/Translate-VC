
# Build and run the Docker container for the Multilingual Voice Chat application

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if NVIDIA Container Toolkit is installed (for GPU support)
if docker info | grep -q "Runtimes: nvidia"; then
    echo "NVIDIA Container Toolkit detected - GPU support enabled"
    HAS_NVIDIA=true
else
    echo "Note: NVIDIA Container Toolkit not detected - running with CPU only"
    HAS_NVIDIA=false
    
    # Modify docker-compose.yml to remove GPU requirements
    sed -i.bak '/deploy:/,/capabilities: \[gpu\]/d' docker-compose.yml
    sed -i.bak '/NVIDIA_VISIBLE_DEVICES/d' docker-compose.yml
fi

# Build and run using docker-compose
echo "Building and starting container..."
docker-compose up --build

# Restore original docker-compose.yml if modified
if [ "$HAS_NVIDIA" = false ] && [ -f "docker-compose.yml.bak" ]; then
    mv docker-compose.yml.bak docker-compose.yml
fi

echo "Container stopped. To restart, run: docker-compose up"