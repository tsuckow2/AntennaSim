#include "mfem.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace {
constexpr uint32_t kAbiVersion = 1;
constexpr uint32_t kEndianSentinel = 0x12345678;
constexpr uint32_t kHeaderBytes = 128;
constexpr uint32_t kResponseBytes = 104;

uint32_t read_u32(const uint8_t *data, uint32_t offset) {
  uint32_t value;
  std::memcpy(&value, data + offset, sizeof(value));
  return value;
}

void write_u32(uint8_t *data, uint32_t offset, uint32_t value) {
  std::memcpy(data + offset, &value, sizeof(value));
}
}

extern "C" {

void *mfem_solver_alloc(uint32_t size_bytes) {
  return std::malloc(size_bytes);
}

void mfem_solver_free(void *ptr) {
  std::free(ptr);
}

void mfem_solver_set_cancelled(int) {}

int mfem_solver_run(const void *request, uint32_t request_length,
                   void *response_ptr_out, void *response_len_out) {
  if (request_length < kHeaderBytes) return 1;
  const auto *data = static_cast<const uint8_t *>(request);
  if (read_u32(data, 0) != kAbiVersion) return 3;
  if (read_u32(data, 4) != kEndianSentinel) return 1;

  const uint32_t vertex_count = read_u32(data, 24);
  const uint32_t tet_count = read_u32(data, 28);
  const uint32_t vertices_offset = read_u32(data, 48);
  const uint32_t vertices_length = read_u32(data, 84);
  const uint32_t tets_offset = read_u32(data, 52);
  const uint32_t tets_length = read_u32(data, 88);
  if (vertices_length != vertex_count * 3u * 8u ||
      tets_length != tet_count * 4u * 4u ||
      vertices_offset < kHeaderBytes || tets_offset < kHeaderBytes ||
      vertices_offset + vertices_length > request_length ||
      tets_offset + tets_length > request_length) return 2;

  const char *message = "MFEM production solver ABI loaded; frequency solve is not implemented";
  const uint32_t message_length = static_cast<uint32_t>(std::strlen(message));
  auto *response = static_cast<uint8_t *>(std::malloc(kResponseBytes + message_length));
  if (!response) return 6;
  std::memset(response, 0, kResponseBytes + message_length);
  write_u32(response, 0, kAbiVersion);
  write_u32(response, 4, 6); // solver_failure until the physical solve is implemented.
  write_u32(response, 96, kResponseBytes);
  write_u32(response, 100, message_length);
  std::memcpy(response + kResponseBytes, message, message_length);

  const uint32_t response_pointer = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(response));
  const uint32_t response_length = kResponseBytes + message_length;
  std::memcpy(response_ptr_out, &response_pointer, sizeof(response_pointer));
  std::memcpy(response_len_out, &response_length, sizeof(response_length));
  return 6;
}

}