#include "eddy/eddy_c.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
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
  // Never dereferenced: the NULL devices check returns before the handle is used.
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

// No models needed: a model directory without the graphs is accepted by create, so the
// getter's compile fails and must report it with the struct zeroed, on a retry too.
bool test_missing_models_report_load_failure() {
  const auto missing = std::filesystem::temp_directory_path() / "eddy_c_test_missing_models";
  std::filesystem::remove_all(missing);
#if !defined(_WIN32)
  // Keep the OpenVINO cache directory create sets up out of the user's cache.
  if (!std::getenv("EDDY_TEST_PARAKEET_MODEL")) {
    const auto cache = std::filesystem::temp_directory_path() / "eddy_c_test_cache";
    setenv("XDG_CACHE_HOME", cache.string().c_str(), 1);
  }
#endif
  const std::string missing_dir = missing.string();
  EddyParakeetConfig config{};
  config.device = "CPU";
  config.model_dir = missing_dir.c_str();
  char* error = nullptr;
  EddyParakeetModel model = eddy_parakeet_create(config, &error);
  if (!expect(model != nullptr, std::string("create failed: ") + (error ? error : ""))) {
    eddy_free_string(error);
    return false;
  }
  bool ok = true;
  for (int attempt = 0; attempt < 2; ++attempt) {
    EddyParakeetComponentDevices devices{};
    devices.encoder = new char[1]{};
    char* stale = devices.encoder;
    const auto status = eddy_parakeet_get_component_devices(model, &devices, &error);
    delete[] stale;
    ok &= expect(status == EDDY_ERROR_MODEL_LOAD_FAILED, "missing models return EDDY_ERROR_MODEL_LOAD_FAILED");
    ok &= expect(error != nullptr, "missing models set error_message");
    ok &= expect(is_zeroed(devices), "missing models leave devices zeroed");
    eddy_free_string(error);
    error = nullptr;
  }
  eddy_parakeet_destroy(model);
  return ok;
}

// Opt-in: EDDY_TEST_PARAKEET_MODEL selects "parakeet-v2" or "parakeet-v3" from the Eddy cache;
// EDDY_TEST_DEVICE (default CPU) and EDDY_TEST_EXPECT_DEVICE (default: the requested device) set
// the request and the device encoder, decoder and joint must report. AUTO without an expected
// device only requires a non-empty report, since AUTO's value is a snapshot.
bool test_cached_model_reports_component_devices() {
  const char* model_name = std::getenv("EDDY_TEST_PARAKEET_MODEL");
  if (!model_name || !*model_name) {
    std::cout << "SKIP: set EDDY_TEST_PARAKEET_MODEL to run the cached-model check\n";
    return true;
  }
  const char* device = std::getenv("EDDY_TEST_DEVICE");
  if (!device || !*device) device = "CPU";
  const char* expected = std::getenv("EDDY_TEST_EXPECT_DEVICE");
  const std::string expect_device = expected && *expected ? expected : (std::string(device) == "AUTO" ? "" : device);
  const char* preproc_env = std::getenv("EDDY_PREPROC_DEVICE");
  const std::string expect_preproc = preproc_env && *preproc_env ? preproc_env : "CPU";

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
    ok &= expect(devices.preprocessor == expect_preproc, "preprocessor reports " + expect_preproc);
    for (const char* graph : {devices.encoder, devices.decoder, devices.joint}) {
      ok &= expect_device.empty() ? expect(*graph != '\0', "graph reports a device")
                                  : expect(graph == expect_device, "graph reports " + expect_device);
    }
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
  ok &= test_missing_models_report_load_failure();
  ok &= test_cached_model_reports_component_devices();
  return ok ? 0 : 1;
}
