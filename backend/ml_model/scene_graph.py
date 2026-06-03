import numpy as np


class SpatialSceneGraph:
    """Constructs a relational spatial scene graph from multi-object tracked detections."""

    def __init__(self, proximity_threshold_m=0.6):
        self.proximity_threshold_m = proximity_threshold_m

    def build(self, tracked_detections):
        """
        Construct nodes and relational spatial edges.
        tracked_detections: list of dicts with 'track_id', 'class', 'bbox', 'spatial_3d', 'velocity'
        Returns: dict with 'nodes', 'edges', 'summary_prompt', and 'graph_metrics'
        """
        nodes = []
        for idx, det in enumerate(tracked_detections):
            # Guarantee unique track/node identifier even when raw detections lack tracking IDs
            track_id = det.get("track_id")
            if track_id is None or track_id == 0:
                track_id = idx + 1

            node = {
                "track_id": track_id,
                "class": det.get("class", "unknown"),
                "confidence": round(float(det.get("confidence", 0.0)), 2),
                "bbox": det.get("bbox", [0, 0, 0, 0]),
            }

            spatial_data = det.get("spatial_3d") or det.get("spatial")
            if spatial_data:
                node["spatial"] = spatial_data

            if "velocity" in det:
                node["velocity"] = det["velocity"]

            nodes.append(node)

        edges = []
        n = len(nodes)
        node_degrees = [0] * n

        for i in range(n):
            for j in range(i + 1, n):
                n1 = nodes[i]
                n2 = nodes[j]

                c1 = self._get_coords_3d(n1.get("spatial"))
                c2 = self._get_coords_3d(n2.get("spatial"))

                if c1 is not None and c2 is not None:
                    # Metric 3D spatial evaluation
                    x1, y1, z1 = c1
                    x2, y2, z2 = c2
                    dx = x1 - x2
                    dy = y1 - y2
                    dz = z1 - z2
                    dist = float(np.sqrt(dx**2 + dy**2 + dz**2))

                    if dist <= self.proximity_threshold_m:
                        predicate = self._classify_relation(n1["class"], n2["class"], dy, dist)
                        rel_x = "left_of" if dx < -0.1 else ("right_of" if dx > 0.1 else "aligned_x")
                        rel_y = "above" if dy > 0.1 else ("below" if dy < -0.1 else "same_level")
                        rel_z = "in_front_of" if dz < -0.1 else ("behind" if dz > 0.1 else "same_depth")

                        edges.append({
                            "source_id": n1["track_id"],
                            "source_class": n1["class"],
                            "target_id": n2["track_id"],
                            "target_class": n2["class"],
                            "relation": predicate,
                            "directional": f"{rel_x}, {rel_y}, {rel_z}",
                            "distance_m": round(dist, 2),
                        })
                        node_degrees[i] += 1
                        node_degrees[j] += 1

                else:
                    # 2D bounding box fallback when 3D depth is not available
                    b1 = n1["bbox"]
                    b2 = n2["bbox"]
                    cx1, cy1 = (b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0
                    cx2, cy2 = (b2[0] + b2[2]) / 2.0, (b2[1] + b2[3]) / 2.0

                    inter_x = max(0.0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
                    inter_y = max(0.0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
                    intersection = inter_x * inter_y

                    # Normalize 2D centroid distance by mean bounding scale
                    char_size = max(1.0, (b1[2] - b1[0] + b2[2] - b2[0]) / 2.0)
                    dist_2d = float(np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)) / char_size

                    if intersection > 0 or dist_2d <= 1.5:
                        predicate = "overlapping_with" if intersection > 0 else "adjacent_to"
                        if "human_face" in (n1["class"], n2["class"]):
                            predicate = "interacting_with"

                        rel_x = "left_of" if cx1 < cx2 - 20 else ("right_of" if cx1 > cx2 + 20 else "aligned_x")
                        rel_y = "above" if cy1 < cy2 - 20 else ("below" if cy1 > cy2 + 20 else "same_level")

                        edges.append({
                            "source_id": n1["track_id"],
                            "source_class": n1["class"],
                            "target_id": n2["track_id"],
                            "target_class": n2["class"],
                            "relation": predicate,
                            "directional": f"{rel_x}, {rel_y}",
                            "distance_m": round(dist_2d, 2),
                        })
                        node_degrees[i] += 1
                        node_degrees[j] += 1

        # Calculate topological node degree centrality
        for i, n_item in enumerate(nodes):
            n_item["degree_centrality"] = (
                round(float(node_degrees[i]) / float(n - 1), 2) if n > 1 else 0.0
            )

        summary_prompt = self._generate_prompt_context(nodes, edges)

        graph_metrics = {
            "node_count": n,
            "edge_count": len(edges),
            "density": round(2.0 * len(edges) / (n * (n - 1)), 2) if n > 1 else 0.0,
            "max_degree": max(node_degrees) if node_degrees else 0,
        }

        return {
            "nodes": nodes,
            "edges": edges,
            "summary_prompt": summary_prompt,
            "metrics": graph_metrics,
        }

    @staticmethod
    def _get_coords_3d(spatial):
        """Safely extract (x, y, z) metric coordinates from spatial dictionaries."""
        if not isinstance(spatial, dict):
            return None
        try:
            x = spatial.get("x_m") if spatial.get("x_m") is not None else spatial.get("x")
            y = spatial.get("y_m") if spatial.get("y_m") is not None else spatial.get("y")
            z = spatial.get("z_m") if spatial.get("z_m") is not None else spatial.get("z")
            if x is not None and y is not None and z is not None:
                return float(x), float(y), float(z)
        except (ValueError, TypeError):
            return None
        return None

    def _classify_relation(self, cls1, cls2, dy, dist_3d):
        """Domain heuristic to assign semantic predicates between detected classes."""
        pair = {cls1, cls2}
        if "human_face" in pair:
            if "sunglasses" in pair or "helmet" in pair:
                return "worn_on"
            return "interacting_with"
        if "chair" in pair and dy > 0.05:
            return "placed_on"
        return "adjacent_to"

    def _generate_prompt_context(self, nodes, edges):
        if not nodes:
            return "No objects detected in the current camera frame."

        descriptions = []
        for n in nodes:
            desc = f"Track #{n['track_id']} ({n['class']})"
            if "spatial" in n:
                desc += f" at {n['spatial']['distance_m']}m distance"
            if "velocity" in n and (abs(n["velocity"][0]) > 2 or abs(n["velocity"][1]) > 2):
                desc += " (moving)"
            descriptions.append(desc)

        summary = "Detected entities: " + "; ".join(descriptions) + "."

        if edges:
            rel_descriptions = [
                f"{e['source_class']} (#{e['source_id']}) is {e['relation']} {e['target_class']} (#{e['target_id']}) (~{e['distance_m']}m apart)"
                for e in edges
            ]
            summary += " Spatial relationships: " + "; ".join(rel_descriptions) + "."

        return summary


scene_graph_builder = SpatialSceneGraph()
