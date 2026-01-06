"""
@ARTICLE{11029450,
  author={Ribeiro, Jean A. and Carmo, Diedre S. Do and Reis, Fabiano and Magalhães, Ricardo S. and Dertkigil, Sergio S. J. and Appenzeller, Simone and Rittner, Leticia},
  journal={IEEE Data Descriptions}, 
  title={Descriptor: Manually Annotated CT Dataset of Lung Lobes in COVID-19 and Cancer Patients (LOCCA)}, 
  year={2025},
  volume={2},
  number={},
  pages={239-246},
  keywords={Lungs;Computed tomography;Annotations;Lung cancer;Biomedical imaging;Lesions;Image segmentation;Manuals;COVID-19;Three-dimensional displays;Cancer;computed tomography (CT) images;COVID-19;dataset;manual annotation for lung lobes},
  doi={10.1109/IEEEDATA.2025.3577999}}
"""
import os
import argparse
import nibabel as nib
import nrrd
import matplotlib.pyplot as plt


def load_nifti(path):
	img = nib.load(path)
	return img.get_fdata(), img.affine


def load_nrrd(path):
	data, header = nrrd.read(path)
	return data, header


def visualize_slice(image, mask=None, slice_index=None, title=None):
	if slice_index is None:
		slice_index = image.shape[2] // 2

	plt.figure(figsize=(10, 5))

	plt.subplot(1, 2, 1)
	plt.imshow(image[:, :, slice_index], cmap='gray')
	plt.title('CT Image' if title is None else title)
	plt.axis('off')

	if mask is not None:
		plt.subplot(1, 2, 2)
		plt.imshow(image[:, :, slice_index], cmap='gray')
		plt.imshow(mask[:, :, slice_index], cmap='jet', alpha=0.5)
		plt.title('Segmentation Mask')
		plt.axis('off')

	plt.tight_layout()
	plt.show()


def main(image_path, mask_path=None):
	print(f"Loading image: {image_path}")
	image, _ = load_nifti(image_path)

	mask = None
	if mask_path and os.path.exists(mask_path):
		print(f"Loading mask: {mask_path}")
		mask, _ = load_nrrd(mask_path)
		if image.shape != mask.shape:
			print("Warning: Image and mask have different dimensions!")

	visualize_slice(image, mask)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Viewer for NIfTI images and NRRD masks.")
	parser.add_argument('-i', '--image', type=str, required=True, help='Path to the NIfTI image (.nii.gz)')
	parser.add_argument('-m', '--mask', type=str, help='(Optional) Path to the NRRD mask (.nrrd)')

	args = parser.parse_args()
	main(args.image, args.mask)
