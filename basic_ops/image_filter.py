
import torch
import torch.nn.functional as F

class ImageFilter:
    def __init__(self, h, w, batch_sz):
        self.h = h
        self.w = w
        self.B = batch_sz
        self.images = torch.randint(
            low=0, high=256, size=(self.B, 1, self.h, self.w))
        self.images /= 255.0
        self.filter = torch.tensor(
            [[-1, -1, -1],[-1,  8, -1], [-1, -1, -1]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)

    def run(self):
        # Apply the filter using 2D convolution
        filtered_images = F.conv2d(self.images, self.filter, padding='same')  # B x 1 x h x w
        
        # Calculate mean and standard deviation of each filtered image
        means = torch.mean(filtered_images, dim=(1,2,3), keepdim=True)  # B x 1 x 1 x 1
        im_diffs = filtered_images - means
        stds = torch.mean(im_diffs * im_diffs, dim=(1,2,3), keepdim=False) ** 0.5

        # Find the image with the highest standard deviation
        _, most_edgy_index = torch.max(stds, dim=0)

        # Calculate the difference between original and filtered versions of the most edgy image
        original_image = self.images[most_edgy_index]  # 1 x h x w
        filtered_image = filtered_images[most_edgy_index]  # 1 x h x w
        difference = original_image - filtered_image

        self.print_results(means, stds, most_edgy_index, difference)

    def print_results(self, means, stds, most_edgy_index, difference):
        # Print results
        print("Mean of each filtered image:", means)
        print("Standard deviation of each filtered image:", stds)
        print("Index of the most edgy image:", most_edgy_index)
        print("Difference statistics:")
        print("  Min:", difference.min().item())
        print("  Max:", difference.max().item())
        print("  Mean:", difference.mean().item())