#include <core/device.hpp>
#include <gtest/gtest.h>
#include <set>
#include <vector>

using namespace surgengine;

// Basic Device functionality tests
class DeviceTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}
};

// Constructor tests
class DeviceConstructorTest : public DeviceTest {};

TEST_F(DeviceConstructorTest, DefaultConstructorCreatesCPU) {
  Device device;

  EXPECT_EQ(device.type, DeviceType::CPU);
  EXPECT_EQ(device.rank, 0);
  EXPECT_TRUE(device.is_cpu());
  EXPECT_FALSE(device.is_cuda());
}

TEST_F(DeviceConstructorTest, ExplicitConstructorCPU) {
  Device device(DeviceType::CPU, 5);

  EXPECT_EQ(device.type, DeviceType::CPU);
  EXPECT_EQ(device.rank, 5);
  EXPECT_TRUE(device.is_cpu());
  EXPECT_FALSE(device.is_cuda());
}

TEST_F(DeviceConstructorTest, ExplicitConstructorCUDA) {
  Device device(DeviceType::CUDA, 3);

  EXPECT_EQ(device.type, DeviceType::CUDA);
  EXPECT_EQ(device.rank, 3);
  EXPECT_FALSE(device.is_cpu());
  EXPECT_TRUE(device.is_cuda());
}

TEST_F(DeviceConstructorTest, ExplicitConstructorCUDADefaultRank) {
  Device device(DeviceType::CUDA);

  EXPECT_EQ(device.type, DeviceType::CUDA);
  EXPECT_EQ(device.rank, 0);
  EXPECT_FALSE(device.is_cpu());
  EXPECT_TRUE(device.is_cuda());
}

// Factory method tests
class DeviceFactoryTest : public DeviceTest {};

TEST_F(DeviceFactoryTest, CPUFactoryMethod) {
  Device device = Device::cpu();

  EXPECT_EQ(device.type, DeviceType::CPU);
  EXPECT_EQ(device.rank, 0);
  EXPECT_TRUE(device.is_cpu());
  EXPECT_FALSE(device.is_cuda());
}

TEST_F(DeviceFactoryTest, CUDAFactoryMethodDefaultRank) {
  Device device = Device::cuda();

  EXPECT_EQ(device.type, DeviceType::CUDA);
  EXPECT_EQ(device.rank, 0);
  EXPECT_FALSE(device.is_cpu());
  EXPECT_TRUE(device.is_cuda());
}

TEST_F(DeviceFactoryTest, CUDAFactoryMethodWithRank) {
  Device device = Device::cuda(7);

  EXPECT_EQ(device.type, DeviceType::CUDA);
  EXPECT_EQ(device.rank, 7);
  EXPECT_FALSE(device.is_cpu());
  EXPECT_TRUE(device.is_cuda());
}

TEST_F(DeviceFactoryTest, CUDAFactoryMethodNegativeRank) {
  Device device = Device::cuda(-1);

  EXPECT_EQ(device.type, DeviceType::CUDA);
  EXPECT_EQ(device.rank, -1);
  EXPECT_FALSE(device.is_cpu());
  EXPECT_TRUE(device.is_cuda());
}

// Type checking tests
class DeviceTypeTest : public DeviceTest {};

TEST_F(DeviceTypeTest, CPUTypeChecks) {
  Device cpu_device = Device::cpu();

  EXPECT_TRUE(cpu_device.is_cpu());
  EXPECT_FALSE(cpu_device.is_cuda());
}

TEST_F(DeviceTypeTest, CUDATypeChecks) {
  Device cuda_device = Device::cuda(2);

  EXPECT_FALSE(cuda_device.is_cpu());
  EXPECT_TRUE(cuda_device.is_cuda());
}

TEST_F(DeviceTypeTest, MultipleDevicesTypeChecks) {
  std::vector<Device> devices = {Device::cpu(), Device::cuda(0),
                                 Device::cuda(1), Device(DeviceType::CPU, 10)};

  EXPECT_TRUE(devices[0].is_cpu());
  EXPECT_FALSE(devices[0].is_cuda());

  EXPECT_FALSE(devices[1].is_cpu());
  EXPECT_TRUE(devices[1].is_cuda());

  EXPECT_FALSE(devices[2].is_cpu());
  EXPECT_TRUE(devices[2].is_cuda());

  EXPECT_TRUE(devices[3].is_cpu());
  EXPECT_FALSE(devices[3].is_cuda());
}

// Equality operator tests
class DeviceEqualityTest : public DeviceTest {};

TEST_F(DeviceEqualityTest, SameDevicesAreEqual) {
  Device device1 = Device::cpu();
  Device device2 = Device::cpu();

  EXPECT_TRUE(device1 == device2);
  EXPECT_FALSE(device1 != device2);
}

TEST_F(DeviceEqualityTest, SameCUDADevicesAreEqual) {
  Device device1 = Device::cuda(3);
  Device device2 = Device::cuda(3);

  EXPECT_TRUE(device1 == device2);
  EXPECT_FALSE(device1 != device2);
}

TEST_F(DeviceEqualityTest, DifferentTypesAreNotEqual) {
  Device cpu_device = Device::cpu();
  Device cuda_device = Device::cuda(0);

  EXPECT_FALSE(cpu_device == cuda_device);
  EXPECT_TRUE(cpu_device != cuda_device);
}

TEST_F(DeviceEqualityTest, DifferentRanksAreNotEqual) {
  Device cuda_device1 = Device::cuda(0);
  Device cuda_device2 = Device::cuda(1);

  EXPECT_FALSE(cuda_device1 == cuda_device2);
  EXPECT_TRUE(cuda_device1 != cuda_device2);
}

TEST_F(DeviceEqualityTest, CPUWithDifferentRanksAreNotEqual) {
  Device cpu_device1(DeviceType::CPU, 0);
  Device cpu_device2(DeviceType::CPU, 1);

  EXPECT_FALSE(cpu_device1 == cpu_device2);
  EXPECT_TRUE(cpu_device1 != cpu_device2);
}

TEST_F(DeviceEqualityTest, SelfEqualityWorks) {
  Device device = Device::cuda(5);

  EXPECT_TRUE(device == device);
  EXPECT_FALSE(device != device);
}

// Edge case tests
class DeviceEdgeCaseTest : public DeviceTest {};

TEST_F(DeviceEdgeCaseTest, ZeroRankDevices) {
  Device cpu_zero = Device::cpu();
  Device cuda_zero = Device::cuda(0);

  EXPECT_EQ(cpu_zero.rank, 0);
  EXPECT_EQ(cuda_zero.rank, 0);
  EXPECT_FALSE(cpu_zero == cuda_zero); // Different types
}

TEST_F(DeviceEdgeCaseTest, LargeRankValues) {
  Device large_rank_device = Device::cuda(99999);

  EXPECT_EQ(large_rank_device.rank, 99999);
  EXPECT_TRUE(large_rank_device.is_cuda());
  EXPECT_FALSE(large_rank_device.is_cpu());
}

TEST_F(DeviceEdgeCaseTest, NegativeRankValues) {
  Device negative_rank_cpu(DeviceType::CPU, -5);
  Device negative_rank_cuda = Device::cuda(-10);

  EXPECT_EQ(negative_rank_cpu.rank, -5);
  EXPECT_EQ(negative_rank_cuda.rank, -10);
  EXPECT_TRUE(negative_rank_cpu.is_cpu());
  EXPECT_TRUE(negative_rank_cuda.is_cuda());
}

// Collection compatibility tests
class DeviceCollectionTest : public DeviceTest {};

TEST_F(DeviceCollectionTest, VectorOfDevices) {
  std::vector<Device> devices = {Device::cpu(), Device::cuda(0),
                                 Device::cuda(1), Device::cuda(2)};

  EXPECT_EQ(devices.size(), 4);
  EXPECT_TRUE(devices[0].is_cpu());
  EXPECT_TRUE(devices[1].is_cuda());
  EXPECT_EQ(devices[2].rank, 1);
  EXPECT_EQ(devices[3].rank, 2);
}

TEST_F(DeviceCollectionTest, SetOfDevices) {
  std::set<Device> device_set;

  // Note: This test assumes operator< is defined for Device
  // If not implemented, you might need to provide a custom comparator
  Device cpu1 = Device::cpu();
  Device cpu2 = Device::cpu();
  Device cuda1 = Device::cuda(0);
  Device cuda2 = Device::cuda(1);

  // For this test to work properly, you'd need to implement operator< for
  // Device We'll test basic insertion and equality instead
  std::vector<Device> unique_devices;

  auto add_if_unique = [&](const Device &d) {
    for (const auto &existing : unique_devices) {
      if (existing == d)
        return;
    }
    unique_devices.push_back(d);
  };

  add_if_unique(cpu1);
  add_if_unique(cpu2); // Should not be added (duplicate)
  add_if_unique(cuda1);
  add_if_unique(cuda2);

  EXPECT_EQ(unique_devices.size(), 3); // cpu, cuda(0), cuda(1)
}

// Copy and assignment tests
class DeviceCopyTest : public DeviceTest {};

TEST_F(DeviceCopyTest, CopyConstructor) {
  Device original = Device::cuda(5);
  Device copy(original);

  EXPECT_EQ(copy.type, original.type);
  EXPECT_EQ(copy.rank, original.rank);
  EXPECT_TRUE(copy == original);
}

TEST_F(DeviceCopyTest, AssignmentOperator) {
  Device original = Device::cuda(7);
  Device assigned = Device::cpu(); // Start with different device

  assigned = original;

  EXPECT_EQ(assigned.type, original.type);
  EXPECT_EQ(assigned.rank, original.rank);
  EXPECT_TRUE(assigned == original);
}

TEST_F(DeviceCopyTest, SelfAssignment) {
  Device device = Device::cuda(3);
  Device *device_ptr = &device;

  device = *device_ptr; // Self assignment

  EXPECT_EQ(device.type, DeviceType::CUDA);
  EXPECT_EQ(device.rank, 3);
  EXPECT_TRUE(device.is_cuda());
}

// Comprehensive scenario tests
class DeviceScenarioTest : public DeviceTest {};

TEST_F(DeviceScenarioTest, MultiGPUScenario) {
  std::vector<Device> gpu_cluster;

  // Create a cluster of 8 GPUs
  for (int i = 0; i < 8; ++i) {
    gpu_cluster.push_back(Device::cuda(i));
  }

  // Verify all devices
  for (int i = 0; i < 8; ++i) {
    EXPECT_TRUE(gpu_cluster[i].is_cuda());
    EXPECT_EQ(gpu_cluster[i].rank, i);
    EXPECT_FALSE(gpu_cluster[i].is_cpu());
  }

  // Test uniqueness
  for (int i = 0; i < 8; ++i) {
    for (int j = i + 1; j < 8; ++j) {
      EXPECT_NE(gpu_cluster[i], gpu_cluster[j]);
    }
  }
}

TEST_F(DeviceScenarioTest, HeterogeneousSystem) {
  std::vector<Device> system_devices = {
      Device::cpu(),             // CPU
      Device::cuda(0),           // Primary GPU
      Device::cuda(1),           // Secondary GPU
      Device(DeviceType::CPU, 1) // Another CPU representation
  };

  // Count devices by type
  int cpu_count = 0, cuda_count = 0;
  for (const auto &device : system_devices) {
    if (device.is_cpu())
      cpu_count++;
    if (device.is_cuda())
      cuda_count++;
  }

  EXPECT_EQ(cpu_count, 2);
  EXPECT_EQ(cuda_count, 2);

  // Verify specific devices
  EXPECT_TRUE(system_devices[0] == Device::cpu());
  EXPECT_TRUE(system_devices[1] == Device::cuda(0));
  EXPECT_TRUE(system_devices[2] == Device::cuda(1));
  EXPECT_FALSE(system_devices[0] == system_devices[3]); // Different CPU ranks
}