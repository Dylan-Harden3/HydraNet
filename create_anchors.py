import argparse
import json
import os
from typing import Tuple
from random import randint
from collections import Counter
from tqdm import tqdm

class Box:
    def __init__(self, x: dict):
        self.x1 = x["x1"]
        self.x2 = x["x2"]
        self.y1 = x["y1"]
        self.y2 = x["y2"]
    def __repr__(self):
        return f"{int(self.x1), int(self.y1), int(self.x2), int(self.y2)}"

def iou(box1: Box, box2: Box) -> float:
    x1, x2 = max(box1.x1, box2.x1), min(box1.x2, box2.x2)
    y1, y2 = max(box1.y1, box2.y1), min(box1.y2, box2.y2)

    if y1 >= y2 or x1 >= x2:
        return 0.0

    intersection = (x2-x1) * (y2-y1)
    x1, x2 = min(box1.x1, box2.x1), max(box1.x2, box2.x2)
    y1, y2 = min(box1.y1, box2.y1), max(box1.y2, box2.y2)

    union = (x2-x1) * (y2-y1)
    return intersection / union

def parse_boxes(dataset_path: str) -> list[dict]:
    base_dir = dataset_path
    files = os.listdir(base_dir)
    boxes = []
    for path in files:
        if path.endswith(".json"):
            dataset = json.load(open(os.path.join(base_dir, path), "r"))
            for img in dataset:
                if "labels" in img:
                    boxes.extend([label["box2d"] for label in img["labels"]])
    boxes = [Box(x) for x in boxes]

    # we just want the box sizes so normalize to start at 0, 0
    for i in range(len(boxes)):
        boxes[i].x2 = boxes[i].x2 - boxes[i].x1
        boxes[i].y2 = boxes[i].y2 - boxes[i].y1
        boxes[i].x1 = 0.0
        boxes[i].y1 = 0.0
    return boxes

def get_anchors(boxes: list[Box], k: int, max_iters: int) -> list[Tuple[int, int]]:
    centroids = set()
    while len(centroids) < k:
        centroids.add(randint(0, len(boxes)))
    centroids = {cluster: boxes[boxix] for cluster, boxix in enumerate(centroids)}
    
    clusters = [i for i in range(len(boxes))]
    for _ in tqdm(range(max_iters)):
        for i in range(len(boxes)):
            min_dist, cluster = float("-inf"), None
            for centroid, box in centroids.items():
                if iou(box, boxes[i]) > min_dist:
                    min_dist = iou(box, boxes[i])
                    cluster = centroid
            clusters[i] = cluster
        next_centroids = {}
        counts = Counter(clusters)
        for centroid in counts:
            next_centroids[centroid] = Box({
                "x1": sum(boxes[ix].x1 for ix in range(len(boxes)) if clusters[ix] == centroid) / counts[centroid],
                "x2": sum(boxes[ix].x2 for ix in range(len(boxes)) if clusters[ix] == centroid) / counts[centroid],
                "y1": sum(boxes[ix].y1 for ix in range(len(boxes)) if clusters[ix] == centroid) / counts[centroid],
                "y2": sum(boxes[ix].y2 for ix in range(len(boxes)) if clusters[ix] == centroid) / counts[centroid],
            })
        centroids = next_centroids
    return centroids
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--k", type=int, default=9)
    parser.add_argument("--max_iters", type=int, default=50)
    args = parser.parse_args()
    boxes = parse_boxes(args.dataset_path)
    anchors = get_anchors(boxes, k=args.k, max_iters=args.max_iters)
    print([v for v in anchors.values()])