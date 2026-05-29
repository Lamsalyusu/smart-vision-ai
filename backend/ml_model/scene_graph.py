import numpy as np


class SpatialSceneGraph:
    """Builds a semantic spatial-temporal scene graph from tracked detections."""

    def __init__(self, proximity_threshold_m=0.6):
        self.proximity_threshold_m = proximity_threshold_m

    def build(self, tracked_detections):
        """
        Construct nodes and relational edges for detected and tracked objects.

        tracked_detections: list of dicts with 'track_id', 'class', 'bbox', 'spatial_3d', 'velocity'
        Returns: dict with 'nodes', 'edges', and 'summary_prompt'
        """
        nodes = []
        for det in tracked_detections:
            node = {
                "track_id": det.get("track_id", 0),
                "class": det.get("class", "unknown"),
                "confidence": round(float(det.get("confidence", 0.0)), 2),
                "bbox": det.get("bbox", [0, 0, 0, 0]),
            }

            if "spatial_3d" in det:
                node["spatial"] = det["spatial_3d"]

            if "velocity" in det:
                node["velocity"] = det["velocity"]

            nodes.append(node)

        edges = []
        n = len(nodes)

        for i in range(n):
            for j in range(i + 1, n):
                n1 = nodes[i]
                n2 = nodes[j]

                # 3D Euclidean distance between objects if available
                spatial1 = n1.get("spatial")
                spatial2 = n2.get("spatial")

                if spatial1 and spatial2:
                    dx = spatial1["x_m"] - spatial2["x_m"]
                    dy = spatial1["y_m"] - spatial2["y_m"]
                    dz = spatial1["z_m"] - spatial2["z_m"]
                    dist_3d = float(np.sqrt(dx**2 + dy**2 + dz**2))

                    if dist_3d <= self.proximity_threshold_m:
                        predicate = "adjacent_to"
                        # Interaction heuristic: knife/phone/bottle near human face
                        if "human_face" in (n1["class"], n2["class"]):
                            predicate = "interacting_with"

                        edges.append({
                            "source_id": n1["track_id"],
                            "source_class": n1["class"],
                            "target_id": n2["track_id"],
                            "target_class": n2["class"],
                            "relation": predicate,
                            "distance_m": round(dist_3d, 2),
                        })

        summary_prompt = self._generate_prompt_context(nodes, edges)

        return {
            "nodes": nodes,
            "edges": edges,
            "summary_prompt": summary_prompt,
        }

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
