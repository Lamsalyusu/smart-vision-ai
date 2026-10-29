# SmartVision Protobuf Contracts

These files define the versioned wire contracts in the `smartvision.v1` package.
They are the canonical schemas for frame input, detections, 3D tracks, scene
graphs, and camera telemetry.

## Validate

With `protoc` installed, compile the complete schema set from the repository
root:

```sh
protoc -I proto --descriptor_set_out=/tmp/smartvision-contracts.pb \
  --include_imports \
  proto/frame.proto \
  proto/detection.proto \
  proto/tracking3d.proto \
  proto/scenegraph.proto \
  proto/camera_telemetry.proto
```

The descriptor output is temporary; the `.proto` files remain the source of
truth. Client code generation and integration with the existing TypeScript and
Python runtime types are follow-up work, not generated artifacts in this
initial schema slice.