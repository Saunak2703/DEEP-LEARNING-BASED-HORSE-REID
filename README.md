###### \# Deep Learning-Based Horse Re-Identification

###### 

###### MSc Artificial Intelligence dissertation project, University of Surrey (2026).

###### Author: Saunak Ghosh

###### 

###### \## Overview

###### 

###### Horse re-identification is the task of recognising the same individual horse across different images. Given a query image, the system ranks a gallery of images so that images of the same horse come first. This project studies the task on racing footage, where horses are partly hidden by jockeys, often have similar coats and move quickly. To the best of the author's knowledge, no public horse re-identification dataset exists, so a dataset was built as part of this work.

###### 

###### \## What this project covers

###### 

###### 1\. Dataset construction: 29,658 images of 21 horse identities from racing video.

###### 2\. A benchmark of three backbones and three loss configurations at two batch sizes.

###### 3\. Experiments on augmentation and background segmentation.

###### 4\. Grad-CAM analysis of what the models focus on.

###### 

###### \## Dataset

###### 

###### \- Horses were detected in racing footage with YOLOv8, then filtered by size and Laplacian blur and resized to 224 x 224.

###### \- MegaDescriptor-S-224 embeddings (used for clustering only), UMAP and HDBSCAN grouped the crops into identities, which were then manually verified.

###### \- 21 identities and 29,658 images from 3 racing video clips.

###### \- Split by identity, with no overlap between training and test horses:

###### &#x20; - Train: 12 identities, 22,984 images

###### &#x20; - Test: 9 identities, 1,997 query images and 4,677 gallery images

###### 

###### The footage, images and dataset construction scripts are not included in this repository.

###### 

###### \## Models and losses

###### 

###### Backbones (all ImageNet-1K pretrained): ResNet-50, EfficientNetV2-S and Swin-Base.

###### 

###### Loss configurations:

###### \- Triplet: cross-entropy with label smoothing + triplet loss

###### \- ArcFace: ArcFace in place of cross-entropy, + triplet loss

###### \- Circle: cross-entropy with label smoothing + Circle Loss

###### 

###### Evaluation uses Rank-1 accuracy and mean Average Precision (mAP).

###### 

###### \## Results

###### 

###### Rank-1 / mAP (%) on the 9 test identities.

###### 

###### | Backbone | Loss | Batch 16 | Batch 32 |

###### |---|---|---|---|

###### | ResNet-50 | Triplet | 84.3 / 65.9 | 80.0 / 65.2 |

###### | ResNet-50 | ArcFace | 70.2 / 43.8 | 77.8 / 50.9 |

###### | ResNet-50 | Circle | 77.6 / 58.9 | 74.9 / 53.1 |

###### | EfficientNetV2-S | Triplet | 81.5 / 59.5 | 81.5 / 66.9 |

###### | EfficientNetV2-S | ArcFace | 77.0 / 43.0 | 77.1 / 43.7 |

###### | EfficientNetV2-S | Circle | 73.6 / 50.2 | 76.7 / 50.1 |

###### | Swin-Base | Triplet | 84.0 / 62.6 | 86.7 / 66.4 |

###### | Swin-Base | ArcFace | 84.6 / 53.2 | 85.7 / 59.2 |

###### | Swin-Base | Circle | 87.2 / 65.1 | 86.1 / 64.8 |

###### 

###### Best Rank-1: Swin-Base + Circle at batch size 16 (87.2%). Best mAP: EfficientNetV2-S + Triplet at batch size 32 (66.9%).

###### 

###### \## Key findings

###### 

###### \- Cross-entropy with triplet loss was the most consistent choice on this small dataset; ArcFace was weaker for the two CNN backbones.

###### \- Swin-Base was the strongest backbone overall.

###### \- Background removal (rembg, U2Net-P) lowered Rank-1 for all three backbones: ResNet-50 80.0 to 78.0, EfficientNetV2-S 81.5 to 78.6, Swin-Base 86.7 to 81.7. This suggests background context carries useful identity cues in racing footage.

###### \- Random Erasing reduced retrieval performance, consistent with coat markings being identity cues.

###### \- Grad-CAM on ResNet-50 and EfficientNetV2-S focused on the jockey (silks and helmet) rather than the horse's coat, suggesting the models use the jockey as a shortcut cue. This may not hold when the same horse appears with a different jockey.

###### 

###### \## Repository contents

###### 

###### \- main.py, args.py: training and evaluation entry point and options

###### \- src/: datasets, models, losses, samplers, evaluation metrics and utilities

###### \- src\_losses\_arcface.py, src\_losses\_circle.py: ArcFace and Circle Loss

###### \- src\_models\_efficientnet\_reid.py, src\_models\_swin\_reid.py: EfficientNetV2-S and Swin-Base models

###### \- segment\_horses\_v3.py: background removal used for the segmentation experiment

###### \- rerank\_test.py: re-ranking experiment

###### \- add\_imbalance.py, imbalance\_circle\_loss.py: exploratory imbalance-aware Circle Loss, not part of the final reported results

###### \- experiments\_all.ps1, run\_single\_experiment.ps1, EXPERIMENTS\_GUIDE.md: commands used to run the experiments

###### 

###### \## Running the code

###### 

###### Developed with Python 3.11. Install the requirements with:

###### 

###### &#x20;   pip install -r requirements.txt

###### 

###### The dataset is not included, so you would need your own re-identification dataset in the folder layout expected by src/datasets/horse.py. See EXPERIMENTS\_GUIDE.md for the exact commands used for each configuration.

###### 

###### \## Acknowledgements

###### 

###### This code is adapted from the University of Surrey EEEM071 (Advanced Computer Vision) coursework codebase.

