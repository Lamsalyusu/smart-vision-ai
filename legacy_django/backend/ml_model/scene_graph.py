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
        nodes = [self._build_node(index, detection) for index, detection in enumerate(tracked_detections)]
        edges, node_degrees = self._build_edges(nodes)
        n = len(nodes)
        self._add_degree_centrality(nodes, node_degrees)
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
    def _build_node(index, detection):
        track_id = detection.get("track_id")
        if track_id is None or track_id == 0:
            track_id = index + 1
        node = {
            "track_id": track_id,
            "class": detection.get("class", "unknown"),
            "confidence": round(float(detection.get("confidence", 0.0)), 2),
            "bbox": detection.get("bbox", [0, 0, 0, 0]),
        }
        spatial_data = detection.get("spatial_3d") or detection.get("spatial")
        if spatial_data:
            node["spatial"] = spatial_data
        if "velocity" in detection:
            node["velocity"] = detection["velocity"]
        return node

    def _build_edges(self, nodes):
        edges = []
        node_degrees = [0] * len(nodes)
        for first_index in range(len(nodes)):
            for second_index in range(first_index + 1, len(nodes)):
                edge = self._build_edge(nodes[first_index], nodes[second_index])
                if edge is None:
                    continue
                edges.append(edge)
                node_degrees[first_index] += 1
                node_degrees[second_index] += 1
        return edges, node_degrees

    def _build_edge(self, first, second):
        first_coords = self._get_coords_3d(first.get("spatial"))
        second_coords = self._get_coords_3d(second.get("spatial"))
        if first_coords is not None and second_coords is not None:
            return self._build_3d_edge(first, second, first_coords, second_coords)
        return self._build_2d_edge(first, second)

    def _build_3d_edge(self, first, second, first_coords, second_coords):
        x1, y1, z1 = first_coords
        x2, y2, z2 = second_coords
        dx, dy, dz = x1 - x2, y1 - y2, z1 - z2
        distance = float(np.sqrt(dx**2 + dy**2 + dz**2))
        if distance > self.proximity_threshold_m:
            return None
        relation = self._classify_relation(first["class"], second["class"], dy, distance)
        direction_x = "left_of" if dx < -0.1 else ("right_of" if dx > 0.1 else "aligned_x")
        direction_y = "above" if dy > 0.1 else ("below" if dy < -0.1 else "same_level")
        direction_z = "in_front_of" if dz < -0.1 else ("behind" if dz > 0.1 else "same_depth")
        return {
            "source_id": first["track_id"],
            "source_class": first["class"],
            "target_id": second["track_id"],
            "target_class": second["class"],
            "relation": relation,
            "directional": f"{direction_x}, {direction_y}, {direction_z}",
            "distance_m": round(distance, 2),
        }

    def _build_2d_edge(self, first, second):
        first_box, second_box = first["bbox"], second["bbox"]
        center_x1 = (first_box[0] + first_box[2]) / 2.0
        center_y1 = (first_box[1] + first_box[3]) / 2.0
        center_x2 = (second_box[0] + second_box[2]) / 2.0
        center_y2 = (second_box[1] + second_box[3]) / 2.0
        intersection_x = max(0.0, min(first_box[2], second_box[2]) - max(first_box[0], second_box[0]))
        intersection_y = max(0.0, min(first_box[3], second_box[3]) - max(first_box[1], second_box[1]))
        intersection = intersection_x * intersection_y
        char_size = max(1.0, (first_box[2] - first_box[0] + second_box[2] - second_box[0]) / 2.0)
        distance = float(np.sqrt((center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2)) / char_size
        if intersection == 0 and distance > 1.5:
            return None
        relation = "overlapping_with" if intersection > 0 else "adjacent_to"
        if "human_face" in (first["class"], second["class"]):
            relation = "interacting_with"
        direction_x = "left_of" if center_x1 < center_x2 - 20 else ("right_of" if center_x1 > center_x2 + 20 else "aligned_x")
        direction_y = "above" if center_y1 < center_y2 - 20 else ("below" if center_y1 > center_y2 + 20 else "same_level")
        return {
            "source_id": first["track_id"],
            "source_class": first["class"],
            "target_id": second["track_id"],
            "target_class": second["class"],
            "relation": relation,
            "directional": f"{direction_x}, {direction_y}",
            "distance_m": round(distance, 2),
        }

    @staticmethod
    def _add_degree_centrality(nodes, node_degrees):
        node_count = len(nodes)
        for index, node in enumerate(nodes):
            node["degree_centrality"] = (
                round(float(node_degrees[index]) / float(node_count - 1), 2)
                if node_count > 1
                else 0.0
            )

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
