| Run | Input px | Metric grid | Device | Parameters | IoU | Dice | Precision | Recall | Boundary F1 | Latency ms |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| Pilot (128px) | 128 | original | mps | 14359105 | 0.8403 | 0.9026 | 0.8981 | 0.9304 | 0.0092 | 13.6180 |
| Selected (384px + flip) | 384 | original | mps | 14359105 | 0.9676 | 0.9805 | 0.9796 | 0.9865 | 0.3291 | 62.5704 |
