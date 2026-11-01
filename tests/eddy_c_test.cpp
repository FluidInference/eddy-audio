#include "eddy/eddy_c.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <string_view>

namespace {

bool expect(bool condition, std::string_view message) {
  if (condition) return true;
  std::cerr << "FAILED: " << message << "\n";
  return false;
}

bool is_zeroed(const EddyParakeetComponentDevices& d) {
  return !d.preprocessor && !d.encoder && !d.decoder && !d.joint;
}

bool test_null_model_is_rejected() {
  EddyParakeetComponentDevices devices{};
  char* error = nullptr;
  const auto status = eddy_parakeet_get_component_devices(nullptr, &devices, &error);
  bool ok = expect(status == EDDY_ERROR_INVALID_ARGUMENT, "NULL model returns EDDY_ERROR_INVALID_ARGUMENT");
  ok &= expect(error != nullptr, "NULL model sets error_message");
  ok &= expect(is_zeroed(devices), "NULL model leaves devices zeroed");
  eddy_free_string(error);
  return ok;
}

bool test_null_devices_is_rejected() {
  int not_a_model = 0;
  char* error = nullptr;
  const auto status = eddy_parakeet_get_component_devices(&not_a_model, nullptr, &error);
  bool ok = expect(status == EDDY_ERROR_INVALID_ARGUMENT, "NULL devices returns EDDY_ERROR_INVALID_ARGUMENT");
  ok &= expect(error != nullptr, "NULL devices sets error_message");
  eddy_free_string(error);
  return ok;
}

bool test_free_is_safe_on_null_and_repeat() {
  eddy_parakeet_free_component_devices(nullptr);
  EddyParakeetComponentDevices devices{};
  eddy_parakeet_free_component_devices(&devices);
  eddy_parakeet_free_component_devices(&devices);
  return expect(is_zeroed(devices), "free leaves a zeroed struct zeroed");
}

// Opt-in: EDDY_TEST_PARAKEET_MODEL selects "parakeet-v2" or "parakeet-v3" from the Eddy cache;
// EDDY_TEST_DEVICE (default CPU) and EDDY_TEST_EXPECT_DEVICE (default: the requested device, CPU
// for AUTO) set the request and the device encoder, decoder and joint must report.
bool test_cached_model_reports_component_devices() {
  const char* model_name = std::getenv("EDDY_TEST_PARAKEET_MODEL");
  if (!model_name || !*model_name) {
    std::cout << "SKIP: set EDDY_TEST_PARAKEET_MODEL to run the cached-model check\n";
    return true;
  }
  const char* device = std::getenv("EDDY_TEST_DEVICE");
  if (!device || !*device) device = "CPU";
  const char* expected = std::getenv("EDDY_TEST_EXPECT_DEVICE");
  const std::string expect_device = expected && *expected ? expected : (std::string(device) == "AUTO" ? "CPU" : device);

  EddyParakeetConfig config{};
  config.device = device;
  config.model_dir = "cache";
  config.blank_token_id = std::string(model_name) == "parakeet-v3" ? 8192 : 1024;
  char* error = nullptr;
  EddyParakeetModel model = eddy_parakeet_create(config, &error);
  if (!expect(model != nullptr, std::string("create failed: ") + (error ? error : ""))) {
    eddy_free_string(error);
    return false;
  }

  EddyParakeetComponentDevices devices{};
  const auto status = eddy_parakeet_get_component_devices(model, &devices, &error);
  bool ok = expect(status == EDDY_OK, std::string("get_component_devices failed: ") + (error ? error : ""));
  if (status == EDDY_OK) {
    std::cout << "preprocessor=" << devices.preprocessor << " encoder=" << devices.encoder
              << " decoder=" << devices.decoder << " joint=" << devices.joint << "\n";
    ok &= expect(std::strcmp(devices.preprocessor, "CPU") == 0, "preprocessor reports CPU");
    ok &= expect(devices.encoder == expect_device, "encoder reports " + expect_device);
    ok &= expect(devices.decoder == expect_device, "decoder reports " + expect_device);
    ok &= expect(devices.joint == expect_device, "joint reports " + expect_device);
  }
  eddy_parakeet_free_component_devices(&devices);
  ok &= expect(is_zeroed(devices), "free zeroes a filled struct");
  eddy_parakeet_free_component_devices(&devices);
  eddy_free_string(error);
  eddy_parakeet_destroy(model);
  return ok;
}

}  // namespace

int main() {
  bool ok = true;
  ok &= test_null_model_is_rejected();
  ok &= test_null_devices_is_rejected();
  ok &= test_free_is_safe_on_null_and_repeat();
  ok &= test_cached_model_reports_component_devices();
  return ok ? 0 : 1;
}
