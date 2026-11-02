export type GpuBackend = "webgpu" | "webgl2" | "webgl" | "none";

interface DetectionScope {
  gpu?: { requestAdapter?: unknown };
  WebGL2RenderingContext?: unknown;
  WebGLRenderingContext?: unknown;
  document?: {
    createElement?: (tagName: string) => {
      getContext?: (contextId: string) => unknown;
    };
  };
}

export function detectGpuBackend(scope: DetectionScope = globalThis): GpuBackend {
  if (scope.gpu && typeof scope.gpu.requestAdapter === "function") {
    return "webgpu";
  }
  const canvas = scope.document?.createElement?.("canvas");
  if (!canvas || typeof canvas.getContext !== "function") {
    return "none";
  }
  if (canvas.getContext("webgl2")) {
    return "webgl2";
  }
  if (canvas.getContext("webgl")) {
    return "webgl";
  }
  return "none";
}
