import { Detection } from "./types";

export function calculateIoU(
  bbox1: [number, number, number, number],
  bbox2: [number, number, number, number]
): number {
  const x1 = Math.max(bbox1[0], bbox2[0]);
  const y1 = Math.max(bbox1[1], bbox2[1]);
  const x2 = Math.min(bbox1[2], bbox2[2]);
  const y2 = Math.min(bbox1[3], bbox2[3]);

  const interW = Math.max(0, x2 - x1);
  const interH = Math.max(0, y2 - y1);
  const intersection = interW * interH;

  const area1 = Math.max(0, bbox1[2] - bbox1[0]) * Math.max(0, bbox1[3] - bbox1[1]);
  const area2 = Math.max(0, bbox2[2] - bbox2[0]) * Math.max(0, bbox2[3] - bbox2[1]);
  const union = area1 + area2 - intersection;

  return union > 0 ? intersection / union : 0;
}

export function bboxToZ(bbox: [number, number, number, number]): [number, number, number, number] {
  const w = Math.max(1e-4, bbox[2] - bbox[0]);
  const h = Math.max(1e-4, bbox[3] - bbox[1]);
  const u = bbox[0] + w / 2.0;
  const v = bbox[1] + h / 2.0;
  const s = w * h;
  const r = w / h;
  return [u, v, s, r];
}

export function xToBbox(
  u: number,
  v: number,
  s: number,
  r: number
): [number, number, number, number] {
  if (s <= 0 || r <= 0) return [0, 0, 0, 0];
  const w = Math.sqrt(s * r);
  const h = s / Math.max(1e-4, w);
  return [u - w / 2.0, v - h / 2.0, u + w / 2.0, v + h / 2.0];
}

export enum TrackState {
  TENTATIVE = 1,
  CONFIRMED = 2,
  DELETED = 3,
}

let globalTrackerCount = 0;

export class KalmanBoxTracker {
  public id: number;
  public className: string;
  public confidence: number;
  public state: TrackState;
  public timeSinceUpdate: number;
  public hits: number;
  public hitStreak: number;
  public age: number;

  // State: [u, v, s, r, u_dot, v_dot, s_dot]
  public u: number;
  public v: number;
  public s: number;
  public r: number;
  public uDot: number;
  public vDot: number;
  public sDot: number;

  constructor(bbox: [number, number, number, number], className: string, confidence: number) {
    globalTrackerCount += 1;
    this.id = globalTrackerCount;
    this.className = className;
    this.confidence = confidence;
    this.state = TrackState.TENTATIVE;
    this.timeSinceUpdate = 0;
    this.hits = 1;
    this.hitStreak = 1;
    this.age = 0;

    const [u, v, s, r] = bboxToZ(bbox);
    this.u = u;
    this.v = v;
    this.s = s;
    this.r = r;
    this.uDot = 0;
    this.vDot = 0;
    this.sDot = 0;
  }

  predict(): [number, number, number, number] {
    // Constant velocity motion update
    this.u += this.uDot;
    this.v += this.vDot;
    this.s = Math.max(1e-4, this.s + this.sDot);
    this.age += 1;

    if (this.timeSinceUpdate > 0) {
      this.hitStreak = 0;
    }
    this.timeSinceUpdate += 1;

    return xToBbox(this.u, this.v, this.s, this.r);
  }

  update(bbox: [number, number, number, number], confidence: number): void {
    this.timeSinceUpdate = 0;
    this.hits += 1;
    this.hitStreak += 1;
    this.confidence = confidence;

    const [measU, measV, measS, measR] = bboxToZ(bbox);

    // Kalman gain approximation (alpha-beta filter smoothing)
    const alpha = 0.7;
    const beta = 0.3;

    const resU = measU - this.u;
    const resV = measV - this.v;
    const resS = measS - this.s;

    this.u += alpha * resU;
    this.v += alpha * resV;
    this.s = Math.max(1e-4, this.s + alpha * resS);
    this.r = measR;

    this.uDot += beta * resU;
    this.vDot += beta * resV;
    this.sDot += beta * resS;

    if (this.state === TrackState.TENTATIVE && this.hits >= 2) {
      this.state = TrackState.CONFIRMED;
    }
  }

  getCurrentBbox(): [number, number, number, number] {
    return xToBbox(this.u, this.v, this.s, this.r);
  }

  getVelocity(): [number, number] {
    return [Math.round(this.uDot * 100) / 100, Math.round(this.vDot * 100) / 100];
  }
}

export class SortTracker {
  private maxAge: number;
  private minHits: number;
  private iouThreshold: number;
  private trackers: KalmanBoxTracker[];

  constructor(maxAge = 15, minHits = 2, iouThreshold = 0.3) {
    this.maxAge = maxAge;
    this.minHits = minHits;
    this.iouThreshold = iouThreshold;
    this.trackers = [];
  }

  update(detections: Detection[]): Detection[] {
    // 1. Predict new locations of existing trackers
    for (const trk of this.trackers) {
      trk.predict();
    }

    const matchedIndices: Array<[number, number]> = [];
    const unmatchedDetections = new Set<number>(detections.map((_, i) => i));
    const unmatchedTrackers = new Set<number>(this.trackers.map((_, i) => i));

    if (this.trackers.length > 0 && detections.length > 0) {
      // Calculate IoU cost matrix
      const iouMatrix: number[][] = [];
      for (let i = 0; i < detections.length; i++) {
        iouMatrix[i] = [];
        for (let j = 0; j < this.trackers.length; j++) {
          const trkBbox = this.trackers[j].getCurrentBbox();
          const detBbox = detections[i].bbox;
          // Class matching bonus
          const classMatch = detections[i].class === this.trackers[j].className ? 1.0 : 0.0;
          const iou = calculateIoU(detBbox, trkBbox);
          iouMatrix[i][j] = classMatch > 0 ? iou : 0;
        }
      }

      // Greedy bipartite matching
      const usedDets = new Set<number>();
      const usedTrks = new Set<number>();

      const pairs: Array<{ detIdx: number; trkIdx: number; iou: number }> = [];
      for (let i = 0; i < detections.length; i++) {
        for (let j = 0; j < this.trackers.length; j++) {
          if (iouMatrix[i][j] >= this.iouThreshold) {
            pairs.push({ detIdx: i, trkIdx: j, iou: iouMatrix[i][j] });
          }
        }
      }

      pairs.sort((a, b) => b.iou - a.iou);

      for (const pair of pairs) {
        if (!usedDets.has(pair.detIdx) && !usedTrks.has(pair.trkIdx)) {
          matchedIndices.push([pair.detIdx, pair.trkIdx]);
          usedDets.add(pair.detIdx);
          usedTrks.add(pair.trkIdx);
          unmatchedDetections.delete(pair.detIdx);
          unmatchedTrackers.delete(pair.trkIdx);
        }
      }
    }

    // 2. Update matched trackers
    for (const [detIdx, trkIdx] of matchedIndices) {
      const det = detections[detIdx];
      this.trackers[trkIdx].update(det.bbox, det.confidence);
    }

    // 3. Create new trackers for unmatched detections
    for (const detIdx of unmatchedDetections) {
      const det = detections[detIdx];
      const newTrk = new KalmanBoxTracker(det.bbox, det.class as string, det.confidence);
      this.trackers.push(newTrk);
    }

    // 4. Collect active tracked detections
    const outputDetections: Detection[] = [];
    const activeTrackers: KalmanBoxTracker[] = [];

    for (const trk of this.trackers) {
      if (
        trk.timeSinceUpdate <= this.maxAge &&
        (trk.hits >= this.minHits || trk.hitStreak >= 1)
      ) {
        activeTrackers.push(trk);

        // Find associated detection if updated on this frame
        if (trk.timeSinceUpdate === 0) {
          const rawDet = detections.find((d) => d.class === trk.className);
          const bbox = trk.getCurrentBbox();
          outputDetections.push({
            class: trk.className,
            confidence: trk.confidence,
            bbox: [
              Math.round(bbox[0]),
              Math.round(bbox[1]),
              Math.round(bbox[2]),
              Math.round(bbox[3]),
            ],
            track_id: trk.id,
            velocity: trk.getVelocity(),
            spatial_3d: rawDet?.spatial_3d,
            spatial: rawDet?.spatial,
          });
        }
      }
    }

    this.trackers = activeTrackers;
    return outputDetections;
  }
}
