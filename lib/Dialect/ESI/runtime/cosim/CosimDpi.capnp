##===- CosimDpi.capnp - ESI cosim RPC schema ------------------*- CAPNP -*-===//
##
## Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
##===----------------------------------------------------------------------===//
##
## The ESI cosimulation RPC Cap'nProto schema. Documentation is in
## docs/ESI/cosim.md. TL;DR: Run the simulation, then connect to its RPC server
## with a client generated by the Cap'nProto implementation for your language of
## choice! (https://capnproto.org/otherlang.html)
##
##===----------------------------------------------------------------------===//

@0x9fd65fec6e2d2779;

# The primary interface exposed by an ESI cosim simulation.
interface CosimDpiServer @0xe3d7f70c7065c46a {
  # List all the registered endpoints.
  list @0 () -> (ifaces :List(EsiDpiInterfaceDesc));
  # Open one of them. Specify both the send and recv data types if want type
  # safety and your language supports it.
  open @1 (iface :EsiDpiInterfaceDesc) -> (endpoint :EsiDpiEndpoint);

  # Get the zlib-compressed JSON system manifest.
  getCompressedManifest @2 () -> (version :Int32, compressedManifest :Data);

  # Create a low level interface into the simulation.
  openLowLevel @3 () -> (lowLevel :EsiLowLevel);
}

# Description of a registered endpoint.
struct EsiDpiInterfaceDesc @0xd2584d2506f01c8c {
  # Capn'Proto ID of the struct type being sent _to_ the simulator.
  fromHostType @0 :Text;
  # Capn'Proto ID of the struct type being sent _from_ the simulator.
  toHostType @1 :Text;
  # Numerical identifier of the endpoint. Defined in the design.
  endpointID @2 :Text;
}

# Interactions with an open endpoint. Optionally typed.
interface EsiDpiEndpoint @0xfb0a36bf859be47b {
  # Send a message to the endpoint.
  sendFromHost @0 (msg :Data);
  # Recieve a message from the endpoint. Non-blocking.
  recvToHost @1 () -> (hasData :Bool, resp :Data);
  # Close the connect to this endpoint.
  close @2 ();
}

interface EsiHostMemory @0xb566da0118690d14 {
  write @0 (address :UInt64, data :Data) -> ();
  read @1 (address :UInt64, size :UInt64) -> (data :Data);
}

# A low level interface simply provides MMIO and host memory access. In all
# cases, hardware errors become exceptions.
interface EsiLowLevel @0xae716100ef82f6d6 {
  # Write to an MMIO register.
  writeMMIO @0 (address :UInt32, data :UInt32) -> ();
  # Read from an MMIO register.
  readMMIO  @1 (address :UInt32) -> (data :UInt32);

  # Register an interface for the simulation to use to access host memory.
  registerHostMemory @2 (mem :EsiHostMemory) -> ();
}
