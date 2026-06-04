import { Detection, Spatial3D } from "./types";

export class CameraIntrinsics {
  public width: number;
  public height: number;
  public fovRad: number;
  public fx: number;
  public fy: number;
  public cx: number;
  public cy: number;

  constructor(fovDegrees = 60.0, width = 640, height = 480) {
    this.width = width;
    this.height = height;
    this.fovRad = (fovDegrees * Math.PI) / 180.0;
    this.fx = width / 2.0 / Math.tan(this.fovRad / 2.0);
    this.fy = this.fx;
    this.cx = width / 2.0;
    this.cy = height / 2.0;
  }

  updateResolution(width: number, height: number): void {
    if (width !== this.width || height !== this.height) {
      this.width = width;
      this.height = height;
      this.fx = width / 2.0 / Math.tan(this.fovRad / 2.0);
      this.fy = this.fx;
      this.cx = width / 2.0;
      this.cy = height / 2.0;
    }
  }

  backProject(u: number, v: number, zMetric: number): [number, number, number] {
    const x = ((u - this.cx) * zMetric) / this.fx;
    const y = -((v - this.cy) * zMetric) / this.fy; // inverted Y so camera frame points upwards
    return [Math.round(x * 100) / 100, Math.round(y * 100) / 100, Math.round(zMetric * 100) / 100];
  }
}

export class Spatial3DFusion {
  public intrinsics: CameraIntrinsics;
  public minDepthM: number;
  public maxDepthM: number;
  public cameraHeightM: number;
  public cameraPitchRad: number;

  constructor(
    fovDegrees = 60.0,
    minDepthM = 0.35,
    maxDepthM = 4.5,
    cameraHeightM = 1.2,
    cameraPitchDeg = 0.0
  ) {
    this.intrinsics = new CameraIntrinsics(fovDegrees);
    this.minDepthM = minDepthM;
    this.maxDepthM = maxDepthM;
    this.cameraHeightM = cameraHeightM;
    this.cameraPitchRad = (cameraPitchDeg * Math.PI) / 180.0;
  }

  estimateSpatialMetrics(
    detections: Detection[],
    imageWidth = 640,
    imageHeight = 480
  ): Detection[] {
    this.intrinsics.updateResolution(imageWidth, imageHeight);

    return detections.map((det) => {
      const [x1, y1, x2, y2] = det.bbox;
      const u = (x1 + x2) / 2.0;
      const v = (y1 + y2) / 2.0;

      // Class prior depth heuristic combined with bounding box scale
      const boxW = Math.max(1, x2 - x1);
      const boxH = Math.max(1, y2 - y1);
      const boxAreaRatio = (boxW * boxH) / (imageWidth * imageHeight);

      // Estimate metric Z from perspective projection
      // An object filling 50% of the frame is ~0.6m away, 1% of the frame is ~3.5m away
      const normalizedArea = Math.min(1.0, Math.max(0.005, boxAreaRatio));
      const zMetric = Math.min(
        this.maxDepthM,
        Math.max(
          this.minDepthM,
          0.45 / Math.sqrt(normalizedArea)
        )
      );

      const [x, y, z] = this.intrinsics.backProject(u, v, zMetric);
      const distance = Math.round(Math.sqrt(x * x + y * y + z * z) * 100) / 100;

      // Metric object dimensions
      const widthMetric = Math.round(((boxW * zMetric) / this.intrinsics.fx) * 100) / 100;
      const heightMetric = Math.round(((boxH * zMetric) / this.intrinsics.fy) * 100) / 100;

      // Ground plane elevation
      const elevation =
        Math.round(
          (this.cameraHeightM + y * Math.cos(this.cameraPitchRad) - z * Math.sin(this.cameraPitchRad)) *
            100
        ) / 100;

      const spatial: Spatial3D = {
        x,
        y,
        z,
        distance_m: distance,
        depth_m: z,
        width_m: widthMetric,
        height_m: heightMetric,
        elevation_m: elevation,
      };

      return {
        ...det,
        spatial_3d: spatial,
        spatial,
      };
    });
  }
}

export const spatialFuser = new Spatial3DFusion();
