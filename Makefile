# Build configuration
BUILD_DIR = build
CMAKE_BUILD_TYPE = Debug
CMAKE_INSTALL_PREFIX = /usr/local

# Default target
all: build

# Create build directory and configure with CMake, then build
build:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. \
		-DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
		-DCMAKE_INSTALL_PREFIX=$(CMAKE_INSTALL_PREFIX)
	$(MAKE) -C $(BUILD_DIR) -j$(shell nproc)

# Build and run tests
test: build
	cd $(BUILD_DIR)
	ctest --verbose

# Build only tests
build-tests: build
	$(MAKE) -C $(BUILD_DIR) device_test tensor_test parameter_test module_test linear_layer_test

# Clean build directory
clean:
	rm -rf $(BUILD_DIR)

# Install the built project
install: build
	$(MAKE) -C $(BUILD_DIR) install

# Reconfigure and rebuild
rebuild: clean build

# Run a specific test
run-test-%: build
	cd $(BUILD_DIR) && ./tests/$*

# Debug build
debug: CMAKE_BUILD_TYPE=Debug
debug: clean build

# Force debug build (explicit)
debug-force:
	rm -rf $(BUILD_DIR)
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. \
		-DCMAKE_BUILD_TYPE=Debug \
		-DCMAKE_INSTALL_PREFIX=$(CMAKE_INSTALL_PREFIX) \
		-DCMAKE_CXX_FLAGS="-g -O0 -DDEBUG" \
		-DCMAKE_CUDA_FLAGS="-g -G -O0"
	$(MAKE) -C $(BUILD_DIR) -j$(shell nproc)

# Debug a specific test
debug-test-%: debug
	cd $(BUILD_DIR)/tests && gdb --args ./$* --gtest_filter='*'

# Debug a specific test case
debug-test-case: debug
	@echo "Usage: make debug-test-case TEST=tensor_test CASE=TensorTest.ViewReshape"
	cd $(BUILD_DIR)/tests && gdb --args ./$(TEST) --gtest_filter=$(CASE)

.PHONY: all build clean install rebuild test build-tests run-test-% debug