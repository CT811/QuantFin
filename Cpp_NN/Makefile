# Define the build directory
BUILD_DIR = build

# Default target
install: $(BUILD_DIR)
	@echo "Building the project..."
	cd $(BUILD_DIR) && cmake .. && make && make install

# Create the build directory if it doesn't exist
$(BUILD_DIR):
	@echo "Creating build directory..."
	mkdir -p $(BUILD_DIR)

