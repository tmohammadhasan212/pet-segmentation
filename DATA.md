# Oxford-IIIT Pet data contract

Source: [Oxford Visual Geometry Group dataset page](https://www.robots.ox.ac.uk/~vgg/data/pets/).
Citation: Omkar M. Parkhi, Andrea Vedaldi, Andrew Zisserman and C. V. Jawahar,
“Cats and Dogs,” IEEE Conference on Computer Vision and Pattern Recognition, 2012.
The source states **Creative Commons Attribution-ShareAlike 4.0 International** for
commercial/research use; original image owners retain copyright. See the
[source license](https://www.robots.ox.ac.uk/~vgg/data/pets/) and
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Code licensing
is separate. Attribute the dataset and preserve the applicable share-alike terms
when distributing derived example images. Do not put the full dataset in Git.

The legacy archive README also states research-only use and respect for original
website terms. The current dataset page states CC BY-SA 4.0; both source statements
are recorded here rather than treating the code's MIT license as data permission.

Extract the official `images.tar.gz` and `annotations.tar.gz` into:

```text
data/oxford-pets/
  images/<image_id>.jpg
  annotations/README
  annotations/trainval.txt
  annotations/test.txt
  annotations/trimaps/<image_id>.png
```

`petseg prepare --download` obtains the official archives. Manual placement is
supported if Oxford's server is unavailable. The official archive MD5s, also
published in [torchvision's dataset implementation](https://docs.pytorch.org/vision/0.24/_modules/torchvision/datasets/oxford_iiit_pet.html),
are `5c4f3ee8e5d25df40f4fd59a7f44e54c` (images) and
`95a8c909bbe2e81eed6a22bccdf3f68f` (annotations).
The downloader tries Oxford, its alternate endpoint, then a Hugging Face archive
mirror; every downloaded archive must match the official checksum before extraction.

The preparation command verifies the label descriptions from the downloaded
`annotations/README` before creating manifests:

| Trimap value | Source meaning | Binary target | Valid pixel |
|---|---|---:|---|
| 1 | Foreground | 1 | yes |
| 2 | Background | 0 | yes |
| 3 | Not classified / ambiguous boundary | arbitrary | no |

Losses and region metrics exclude value 3 from every numerator and denominator.
Padding and rotation fill use 3 too. All-ignored images produce zero loss and
undefined metrics (JSON null), excluded from macro averages. Both-empty masks
score IoU/Dice 1; precision or recall with a zero denominator is 1 by convention.
This explicit convention avoids NaN and should be considered when interpreting
empty-mask subsets.

The official `trainval.txt` is split **within each breed**, using seed 42 and
20% validation by default. The official `test.txt` remains untouched. Sorted
image IDs are written to `data/manifests/{train,val,test}.txt`. Commit those
files along with `stats.json`, `audit.json` and `annotation-source.txt` after
preparation. Existing complete manifests are verified, never silently replaced.
The checkpoint records a SHA-256 of all three manifests and statistics;
evaluation rejects a different contract. No test labels choose thresholds,
checkpoints, learning rates or normalization. Threshold is preregistered at 0.5;
interactive demo adjustment is for individual cutouts, not final test reporting.

Normalization mean/std are computed only from training images (RGB pixels after
aspect-preserving thumbnails up to 256 pixels). Both encoders use these same
statistics, including the ImageNet-initialized ResNet18. This deliberately follows
the project's training-only normalization requirement; it differs from the
ImageNet normalization normally used with torchvision pretrained weights.

Images are fitted to a square canvas with aspect ratio preserved. Labels use
nearest-neighbor interpolation, images bilinear. Paired horizontal/vertical flips
and ±15° rotations share one deterministic random stream (seed, epoch, index).
Brightness, contrast and saturation change only the image. Validation and test
use deterministic letterboxing with ignored padding. Inference crops that
padding and interpolates probabilities back to the EXIF-oriented original size.

The audit decodes every official image/mask, checks dimensions and requires label
values to belong to {1,2,3}; missing and corrupt files stop training. Annotation
coverage for multiple pets must be checked visually per image: never assume all
animals in the photograph are labeled. Record the answer in the generated
`gallery_tags.csv` before interpreting such failures.
