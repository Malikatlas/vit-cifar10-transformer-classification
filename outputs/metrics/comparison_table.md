| model_name | best_validation_accuracy | test_accuracy | precision | recall | f1_score | total_training_time | peak_gpu_memory_mb | inference_images_per_second |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ViT from scratch | 0.7638 | 0.7536 | 0.7530 | 0.7536 | 0.7513 | 410.5695 | 498.6870 | 788.4147 |
| Hybrid CNN + MLP | 0.9040 | 0.8992 | 0.9014 | 0.8992 | 0.8997 | 315.2951 | 725.2598 | 3656.9879 |
| Pretrained ResNet18 transfer learning | 0.8658 | 0.8598 | 0.8599 | 0.8598 | 0.8595 | 898.7038 | 1547.7998 | 372.4158 |
