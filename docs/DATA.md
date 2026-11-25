# Data description and availability

## Private FMI cohort

The paper uses 554 normalized FMI images from five independent wells in the
Maokou and Longwangmiao carbonate formations of the Sichuan Basin. Three wells
(398 images, 30 acquisition sessions) form the training split, one well (86
images, 6 sessions) forms the validation split, and one previously unseen well
(70 images, 7 sessions) forms the test split. No well or acquisition session
crosses a split.

Three geologists independently annotated fracture pixels, and disagreements
were adjudicated by a senior interpreter. The FMI images, masks, well
identifiers, and private well-to-session key are covered by an industry
confidentiality agreement and are not distributed in this repository. The
public manifests therefore document the required schema without exposing field
identifiers.

## Structural proxy datasets

The structural-prior stage uses CFD, Crack500, CrackTree200, DeepCrack, HRF,
and STARE. These datasets are not repackaged here. Obtain each dataset from its
original provider and respect its license and terms. The user guide describes
the expected local directory layout.

## Public verification data

`demo/` contains ten deterministically generated grayscale images and binary
masks. They exercise manifest loading, training, validation-only threshold
selection, test evaluation, metric export, and prediction saving without using
field data. `examples/` contains three deidentified qualitative panels for
visual inspection; they are not used to calculate manuscript metrics.
