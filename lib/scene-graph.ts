import { Detection, SceneGraph, SceneGraphEdge, SceneGraphNode } from "./types";

export class SpatialSceneGraph {
  public proximityThresholdM: number;

  constructor(proximityThresholdM = 0.8) {
    this.proximityThresholdM = proximityThresholdM;
  }

  private classifyRelation(
    classA: string,
    classB: string,
    deltaY: number,
    dist: number
  ): string {
    const pair = new Set([classA, classB]);

    if (pair.has("human_face")) {
      if (pair.has("sunglasses") || pair.has("helmet")) return "wearing";
      if (pair.has("mobile_phone") || pair.has("knife") || pair.has("water_bottle") || pair.has("pen"))
        return "holding";
      return "interacting_with";
    }

    if (pair.has("chair") && deltaY > 0.1) return "supporting";
    if (dist < 0.25) return "in_contact_with";
    return "adjacent_to";
  }

  build(trackedDetections: Detection[]): SceneGraph {
    const nodes: SceneGraphNode[] = [];

    for (let idx = 0; idx < trackedDetections.length; idx++) {
      const det = trackedDetections[idx];
      const trackId = det.track_id || idx + 1;

      const node: SceneGraphNode = {
        track_id: trackId,
        class: det.class,
        confidence: Math.round(det.confidence * 100) / 100,
        bbox: det.bbox,
        spatial: det.spatial_3d || det.spatial,
        velocity: det.velocity,
      };

      nodes.push(node);
    }

    const edges: SceneGraphEdge[] = [];
    const n = nodes.length;
    const nodeDegrees = new Array(n).fill(0);

    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const n1 = nodes[i];
        const n2 = nodes[j];

        const s1 = n1.spatial;
        const s2 = n2.spatial;

        if (s1 && s2) {
          // 3D Metric Spatial Relationship
          const dx = s1.x - s2.x;
          const dy = s1.y - s2.y;
          const dz = s1.z - s2.z;
          const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

          if (dist <= this.proximityThresholdM) {
            const predicate = this.classifyRelation(n1.class, n2.class, dy, dist);
            const relX = dx < -0.1 ? "left_of" : dx > 0.1 ? "right_of" : "aligned_x";
            const relY = dy > 0.1 ? "above" : dy < -0.1 ? "below" : "same_level";
            const relZ = dz < -0.1 ? "in_front_of" : dz > 0.1 ? "behind" : "same_depth";

            edges.push({
              source_id: n1.track_id,
              source_class: n1.class,
              target_id: n2.track_id,
              target_class: n2.class,
              relation: predicate,
              directional: `${relX}, ${relY}, ${relZ}`,
              distance_m: Math.round(dist * 100) / 100,
            });

            nodeDegrees[i] += 1;
            nodeDegrees[j] += 1;
          }
        } else {
          // 2D bounding box fallback
          const b1 = n1.bbox;
          const b2 = n2.bbox;
          const cx1 = (b1[0] + b1[2]) / 2.0;
          const cy1 = (b1[1] + b1[3]) / 2.0;
          const cx2 = (b2[0] + b2[2]) / 2.0;
          const cy2 = (b2[1] + b2[3]) / 2.0;

          const interX = Math.max(0, Math.min(b1[2], b2[2]) - Math.max(b1[0], b2[0]));
          const interY = Math.max(0, Math.min(b1[3], b2[3]) - Math.max(b1[1], b2[1]));
          const intersection = interX * interY;

          const charSize = Math.max(1.0, (b1[2] - b1[0] + b2[2] - b2[0]) / 2.0);
          const dist2D = Math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) / charSize;

          if (intersection > 0 || dist2D <= 1.5) {
            let predicate = intersection > 0 ? "overlapping_with" : "adjacent_to";
            if (n1.class === "human_face" || n2.class === "human_face") {
              predicate = "interacting_with";
            }

            const relX = cx1 < cx2 - 20 ? "left_of" : cx1 > cx2 + 20 ? "right_of" : "aligned_x";
            const relY = cy1 < cy2 - 20 ? "above" : cy1 > cy2 + 20 ? "below" : "same_level";

            edges.push({
              source_id: n1.track_id,
              source_class: n1.class,
              target_id: n2.track_id,
              target_class: n2.class,
              relation: predicate,
              directional: `${relX}, ${relY}`,
            });

            nodeDegrees[i] += 1;
            nodeDegrees[j] += 1;
          }
        }
      }
    }

    const possibleEdges = (n * (n - 1)) / 2;
    const density = possibleEdges > 0 ? Math.round((edges.length / possibleEdges) * 100) / 100 : 0;

    let maxDeg = -1;
    let centralNode: string | null = null;
    for (let i = 0; i < n; i++) {
      if (nodeDegrees[i] > maxDeg && nodeDegrees[i] > 0) {
        maxDeg = nodeDegrees[i];
        centralNode = `${nodes[i].class} (ID #${nodes[i].track_id})`;
      }
    }

    // Natural Language Summary Prompt
    const summaryLines: string[] = [];
    if (nodes.length === 0) {
      summaryLines.push("No objects currently perceived in scene.");
    } else {
      summaryLines.push(`Scene contains ${nodes.length} perceived object(s):`);
      nodes.forEach((node) => {
        const spatialStr = node.spatial
          ? `at (X: ${node.spatial.x}m, Y: ${node.spatial.y}m, Z: ${node.spatial.z}m, Distance: ${node.spatial.distance_m}m)`
          : "";
        summaryLines.push(`- Track #${node.track_id}: ${node.class.replace("_", " ")} (${Math.round(node.confidence * 100)}% conf) ${spatialStr}`);
      });

      if (edges.length > 0) {
        summaryLines.push("\nSpatial Interactions:");
        edges.forEach((edge) => {
          const distStr = edge.distance_m !== undefined ? ` [${edge.distance_m}m apart]` : "";
          summaryLines.push(`- ${edge.source_class.replace("_", " ")} is ${edge.relation.replace("_", " ")} ${edge.target_class.replace("_", " ")} (${edge.directional})${distStr}`);
        });
      }
    }

    return {
      nodes,
      edges,
      summary_prompt: summaryLines.join("\n"),
      graph_metrics: {
        total_nodes: nodes.length,
        total_edges: edges.length,
        density,
        most_central_node: centralNode,
      },
    };
  }
}

export const sceneGraphBuilder = new SpatialSceneGraph();
