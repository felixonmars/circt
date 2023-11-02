//===- DpiEntryPoints.cpp - ESI cosim DPI calls -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Cosim DPI function implementations. Mostly C-C++ gaskets to the C++
// RpcServer.
//
// These function signatures were generated by an HW simulator (see dpi.h) so
// we don't change them to be more rational here. The resulting code gets
// dynamically linked in and I'm concerned about maintaining binary
// compatibility with all the simulators.
//
//===----------------------------------------------------------------------===//

#include "cosim/Server.h"
#include "cosim/dpi.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>

using namespace esi::cosim;

/// If non-null, log to this file. Protected by 'serverMutex`.
static FILE *logFile;
static RpcServer *server = nullptr;
static std::vector<uint8_t> manifest;
static std::mutex serverMutex;

// ---- Helper functions ----

/// Emit the contents of 'msg' to the log file in hex.
static void log(char *epId, bool toClient, const Endpoint::BlobPtr &msg) {
  std::lock_guard<std::mutex> g(serverMutex);
  if (!logFile)
    return;

  fprintf(logFile, "[ep: %50s to: %4s]", epId, toClient ? "host" : "sim");
  size_t msgSize = msg->size();
  for (size_t i = 0; i < msgSize; ++i) {
    auto b = (*msg)[i];
    // Separate 32-bit words.
    if (i % 4 == 0 && i > 0)
      fprintf(logFile, " ");
    // Separate 64-bit words (capnp word size)
    if (i % 8 == 0 && i > 0)
      fprintf(logFile, "  ");
    fprintf(logFile, " %02x", b);
  }
  fprintf(logFile, "\n");
  fflush(logFile);
}

/// Get the TCP port on which to listen. If the port isn't specified via an
/// environment variable, return 0 to allow automatic selection.
static int findPort() {
  const char *portEnv = getenv("COSIM_PORT");
  if (portEnv == nullptr) {
    printf("[COSIM] RPC server port not found. Letting CapnpRPC select one\n");
    return 0;
  }
  printf("[COSIM] Opening RPC server on port %s\n", portEnv);
  return std::strtoull(portEnv, nullptr, 10);
}

/// Check that an array is an array of bytes and has some size.
// NOLINTNEXTLINE(misc-misplaced-const)
static int validateSvOpenArray(const svOpenArrayHandle data,
                               int expectedElemSize) {
  if (svDimensions(data) != 1) {
    printf("DPI-C: ERROR passed array argument that doesn't have expected 1D "
           "dimensions\n");
    return -1;
  }
  if (svGetArrayPtr(data) == NULL) {
    printf("DPI-C: ERROR passed array argument that doesn't have C layout "
           "(ptr==NULL)\n");
    return -2;
  }
  int totalBytes = svSizeOfArray(data);
  if (totalBytes == 0) {
    printf("DPI-C: ERROR passed array argument that doesn't have C layout "
           "(total_bytes==0)\n");
    return -3;
  }
  int numElems = svSize(data, 1);
  int elemSize = numElems == 0 ? 0 : (totalBytes / numElems);
  if (numElems * expectedElemSize != totalBytes) {
    printf("DPI-C: ERROR: passed array argument that doesn't have expected "
           "element-size: expected=%d actual=%d numElems=%d totalBytes=%d\n",
           expectedElemSize, elemSize, numElems, totalBytes);
    return -4;
  }
  return 0;
}

// ---- Traditional cosim DPI entry points ----

// Register simulated device endpoints.
// - return 0 on success, non-zero on failure (duplicate EP registered).
DPI int sv2cCosimserverEpRegister(char *endpointId, char *fromHostTypeId,
                                  int fromHostTypeSize, char *toHostTypeId,
                                  int toHostTypeSize) {
  // Ensure the server has been constructed.
  sv2cCosimserverInit();
  // Then register with it.
  if (server->endpoints.registerEndpoint(endpointId, fromHostTypeId,
                                         fromHostTypeSize, toHostTypeId,
                                         toHostTypeSize))
    return 0;
  return -1;
}

// Attempt to recieve data from a client.
//   - Returns negative when call failed (e.g. EP not registered).
//   - If no message, return 0 with dataSize == 0.
//   - Assumes buffer is large enough to contain entire message. Fails if not
//     large enough. (In the future, will add support for getting the message
//     into a fixed-size buffer over multiple calls.)
DPI int sv2cCosimserverEpTryGet(char *endpointId,
                                // NOLINTNEXTLINE(misc-misplaced-const)
                                const svOpenArrayHandle data,
                                unsigned int *dataSize) {
  if (server == nullptr)
    return -1;

  Endpoint *ep = server->endpoints[endpointId];
  if (!ep) {
    fprintf(stderr, "Endpoint not found in registry!\n");
    return -4;
  }

  Endpoint::BlobPtr msg;
  // Poll for a message.
  if (!ep->getMessageToSim(msg)) {
    // No message.
    *dataSize = 0;
    return 0;
  }
  // Do the validation only if there's a message available. Since the
  // simulator is going to poll up to every tick and there's not going to be
  // a message most of the time, this is important for performance.

  log(endpointId, false, msg);

  if (validateSvOpenArray(data, sizeof(int8_t)) != 0) {
    printf("ERROR: DPI-func=%s line=%d event=invalid-sv-array\n", __func__,
           __LINE__);
    return -2;
  }

  // Detect or verify size of buffer.
  if (*dataSize == ~0u) {
    *dataSize = svSizeOfArray(data);
  } else if (*dataSize > (unsigned)svSizeOfArray(data)) {
    printf("ERROR: DPI-func=%s line %d event=invalid-size (max %d)\n", __func__,
           __LINE__, (unsigned)svSizeOfArray(data));
    return -3;
  }
  // Verify it'll fit.
  size_t msgSize = msg->size();
  if (msgSize > *dataSize) {
    printf("ERROR: Message size too big to fit in HW buffer\n");
    return -5;
  }

  // Copy the message data.
  size_t i;
  for (i = 0; i < msgSize; ++i) {
    auto b = (*msg)[i];
    *(char *)svGetArrElemPtr1(data, i) = b;
  }
  // Zero out the rest of the buffer.
  for (; i < *dataSize; ++i) {
    *(char *)svGetArrElemPtr1(data, i) = 0;
  }
  // Set the output data size.
  *dataSize = msg->size();
  return 0;
}

// Attempt to send data to a client.
// - return 0 on success, negative on failure (unregistered EP).
// - if dataSize is negative, attempt to dynamically determine the size of
//   'data'.
DPI int sv2cCosimserverEpTryPut(char *endpointId,
                                // NOLINTNEXTLINE(misc-misplaced-const)
                                const svOpenArrayHandle data, int dataSize) {
  if (server == nullptr)
    return -1;

  if (validateSvOpenArray(data, sizeof(int8_t)) != 0) {
    printf("ERROR: DPI-func=%s line=%d event=invalid-sv-array\n", __func__,
           __LINE__);
    return -2;
  }

  // Detect or verify size.
  if (dataSize < 0) {
    dataSize = svSizeOfArray(data);
  } else if (dataSize > svSizeOfArray(data)) { // not enough data
    printf("ERROR: DPI-func=%s line %d event=invalid-size limit %d array %d\n",
           __func__, __LINE__, dataSize, svSizeOfArray(data));
    return -3;
  }

  Endpoint::BlobPtr blob = std::make_unique<Endpoint::Blob>(dataSize);
  // Copy the message data into 'blob'.
  for (int i = 0; i < dataSize; ++i) {
    (*blob)[i] = *(char *)svGetArrElemPtr1(data, i);
  }
  // Queue the blob.
  Endpoint *ep = server->endpoints[endpointId];
  if (!ep) {
    fprintf(stderr, "Endpoint not found in registry!\n");
    return -4;
  }
  log(endpointId, true, blob);
  ep->pushMessageToClient(std::move(blob));
  return 0;
}

// Teardown cosimserver (disconnects from primary server port, stops connections
// from active clients).
DPI void sv2cCosimserverFinish() {
  std::lock_guard<std::mutex> g(serverMutex);
  printf("[cosim] Tearing down RPC server.\n");
  if (server != nullptr) {
    server->stop();
    server = nullptr;

    fclose(logFile);
    logFile = nullptr;
  }
}

// Start cosimserver (spawns server for HW-initiated work, listens for
// connections from new SW-clients).
DPI int sv2cCosimserverInit() {
  std::lock_guard<std::mutex> g(serverMutex);
  if (server == nullptr) {
    // Open log file if requested.
    const char *logFN = getenv("COSIM_DEBUG_FILE");
    if (logFN != nullptr) {
      printf("[cosim] Opening debug log: %s\n", logFN);
      logFile = fopen(logFN, "w");
    }

    // Find the port and run.
    printf("[cosim] Starting RPC server.\n");
    server = new RpcServer();
    server->run(findPort());
  }
  return 0;
}

// ---- Manifest DPI entry points ----

DPI void
sv2cCosimserverSetManifest(const svOpenArrayHandle compressedManifest) {
  sv2cCosimserverInit();

  if (validateSvOpenArray(compressedManifest, sizeof(int8_t)) != 0) {
    printf("ERROR: DPI-func=%s line=%d event=invalid-sv-array\n", __func__,
           __LINE__);
    return;
  }

  // Copy the message data into 'blob'.
  int size = svSizeOfArray(compressedManifest);
  std::vector<uint8_t> blob(size);
  for (int i = 0; i < size; ++i) {
    blob[i] = *(char *)svGetArrElemPtr1(compressedManifest, i);
  }
  server->setManifest(blob);
}

// ---- Low-level cosim DPI entry points ----

static bool mmioRegistered = false;
DPI int sv2cCosimserverMMIORegister() {
  if (mmioRegistered) {
    printf("ERROR: DPI MMIO master already registered!");
    return -1;
  }
  sv2cCosimserverInit();
  mmioRegistered = true;
  return 0;
}

DPI int sv2cCosimserverMMIOReadTryGet(uint32_t *address) {
  assert(server);
  std::optional<int> reqAddress = server->lowLevelBridge.readReqs.pop();
  if (!reqAddress.has_value())
    return -1;
  *address = reqAddress.value();
  return 0;
}

DPI void sv2cCosimserverMMIOReadRespond(uint32_t data, char error) {
  assert(server);
  server->lowLevelBridge.readResps.push(data, error);
}

DPI void sv2cCosimserverMMIOWriteRespond(char error) {
  assert(server);
  server->lowLevelBridge.writeResps.push(error);
}

DPI int sv2cCosimserverMMIOWriteTryGet(uint32_t *address, uint32_t *data) {
  assert(server);
  auto req = server->lowLevelBridge.writeReqs.pop();
  if (!req.has_value())
    return -1;
  *address = req.value().first;
  *data = req.value().second;
  return 0;
}
