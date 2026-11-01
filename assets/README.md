# Small demo artifacts

Store only reviewed, small cutout examples here. Real dataset-derived examples
must preserve Oxford-IIIT Pet attribution and applicable CC BY-SA terms; see
../DATA.md. Do not store full datasets or model checkpoints.

Synthetic examples, if present, illustrate output formats only and are not
measurements of segmentation quality on real pets.

`demo-screenshot.png` shows the verified local upload demo using validation image
`Abyssinian_10` from Oxford-IIIT Pet. The photo is displayed/resized and shown with
derived probability/mask/cutout panels. Dataset attribution: Omkar M. Parkhi,
Andrea Vedaldi, Andrew Zisserman and C. V. Jawahar, *Cats and Dogs*, CVPR 2012;
https://www.robots.ox.ac.uk/~vgg/data/pets/. Dataset imagery and derived panels
are subject to CC BY-SA 4.0 and original image-owner rights; see ../DATA.md.

`example-original.png`, `example-heatmap.png`, `example-mask.png`,
`example-cutout.png` and `example-comparison.png` use the same held-out validation
photo `Abyssinian_10` with the one-epoch pretrained-combined pilot at threshold
0.5. Cutout and mask retain 375×500 dimensions. The comparison is resized for
display. The detached floor patch is a real false positive, retained to show a
limitation. Dataset image and derived panels: the attribution and CC BY-SA terms
above apply. The binary alpha does not recover translucent fur.
