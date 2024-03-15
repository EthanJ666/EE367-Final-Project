import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torchvision.transforms.functional import to_tensor
import torchvision.utils as vutils
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  #prevent broken image loading
from torchvision.transforms.functional import to_tensor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def adjust_dimensions(input, target_shape):
    """
    Adjusts the dimensions of an input tensor to match a target shape.

    Args:
    input (torch.Tensor): The input tensor to be resized.
    target_shape (torch.Size): The target shape to match. Only spatial dimensions are considered.

    Returns:
    torch.Tensor: The resized tensor.
    """
    return F.interpolate(input, size=(target_shape[2], target_shape[3]), mode='bilinear', align_corners=False)

"""
###BUILS DATALOADER###
class CompositeRealDataset(Dataset):
    def __init__(self, composite_dir, real_dir, filenames, transform=None):
        
        self.composite_dir = composite_dir
        self.real_dir = real_dir
        self.transform = transform
        self.filenames = filenames

        #self.filenames = [file for file in os.listdir(composite_dir) if file.endswith('.png')]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        comp_filename = self.filenames[idx]
        real_filename = comp_filename.rsplit('__', 1)[0] + '.jpg'

        comp_path = os.path.join(self.composite_dir, comp_filename)
        real_path = os.path.join(self.real_dir, real_filename)

        comp_image = Image.open(comp_path).convert('RGB')
        real_image = Image.open(real_path).convert('RGB')

        if self.transform is not None:
            comp_image = self.transform(comp_image)
            real_image = self.transform(real_image)
        else:
            comp_image = to_tensor(comp_image)
            real_image = to_tensor(real_image)

        return (comp_image, real_image, comp_filename)
"""


class CompositeRealDataset(Dataset):
    def __init__(self, composite_dir, real_dir, mask_dir, filenames, transform=None):
        
        self.composite_dir = composite_dir
        self.real_dir = real_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.filenames = filenames

        #self.filenames = [file for file in os.listdir(composite_dir) if file.endswith('.png')]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        comp_filename = self.filenames[idx]
        real_filename = comp_filename.rsplit('__', 1)[0] + '.jpg'
        mask_filename = comp_filename.rsplit('__', 1)[0] + '_1' + '.png'

        comp_path = os.path.join(self.composite_dir, comp_filename)
        real_path = os.path.join(self.real_dir, real_filename)
        mask_path = os.path.join(self.mask_dir, mask_filename)

        comp_image = Image.open(comp_path).convert('RGB')
        real_image = Image.open(real_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform is not None:
            comp_image = self.transform(comp_image)
            real_image = self.transform(real_image)
            mask = self.transform(mask)
        else:
            comp_image = to_tensor(comp_image)
            real_image = to_tensor(real_image)
            mask = to_tensor(mask)

        return (comp_image, real_image, mask, comp_filename)


#Paths to composite images, real images, and masks
composite_d = './semi-harmonized_HFlickr'
real_d = './HFlickr/real_images'
mask_d = './HFlickr/masks'
all_filenames = [file for file in os.listdir(composite_d) if file.endswith('.png')]
#random.shuffle(all_filenames)
train_files = all_filenames[:8200]
test_files = all_filenames[8000:]

# Create the Dataset
dataset = CompositeRealDataset(composite_dir=composite_d, real_dir=real_d, mask_dir=mask_d, filenames=train_files, transform=None)
test_dataset = CompositeRealDataset(composite_dir=composite_d, real_dir=real_d, mask_dir=mask_d, filenames=test_files, transform=None)

# Create the DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


class ConvBlock(nn.Module):
    """Convolutional Block consisting of Conv2D -> BatchNorm -> ReLU."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class UpConvBlock(nn.Module):
    """Upsampling Block using Transpose Convolution."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1):
        super(UpConvBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.upconv(x)))

class FCNGenerator(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(FCNGenerator, self).__init__()
        # Decreasing the number of channels to reduce model size and memory usage
        self.down1 = ConvBlock(input_channels, 32)
        self.down2 = ConvBlock(32, 64)
        self.down3 = ConvBlock(64, 128)
        # Intermediate layer for depth
        self.intermediate = ConvBlock(128, 128)
        # Upsampling layers
        self.up1 = UpConvBlock(128, 64)
        self.up2 = UpConvBlock(64, 32)
        self.final = nn.Conv2d(32, output_channels, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Downsample
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        # Intermediate layer
        inter = self.intermediate(d3)
        # Upsample
        u1 = self.up1(inter)
        u2 = self.up2(u1)
        # Final layer to match output_channels
        return self.tanh(self.final(u2))


###Generator (Simple Version Used for Fast Training)###
class FCNGeneratorSimple(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(FCNGeneratorSimple, self).__init__()
        self.down1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.down2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # Downsampling
        self.down3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) # Downsampling
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1) # Upsampling
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)  # Upsampling
        self.out = nn.Conv2d(64, output_channels, kernel_size=3, padding=1)

        self.to(device)

    def forward(self, x):
        x = F.relu(self.down1(x))
        x = F.relu(self.down2(x))
        x = F.relu(self.down3(x))
        x = F.relu(self.up1(x))
        x = F.relu(self.up2(x))
        x = self.out(x)
        return x


###Diescriminator
class Discriminator(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.final_conv = nn.Conv2d(512, 1, kernel_size=1)  # Define the final convolutional layer here

        self.to(device)  # This should ideally be called after initializing the model instance

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.final_conv(x)  # Use the defined layer
        return torch.sigmoid(x.view(-1, 1))


"""
Loss Function
"""
criterion = nn.BCELoss()
criterion_l1 = nn.L1Loss()

#Calculating Mask Loss, given harmonized image, corresponding real image, and mask
def masked_loss(input, target, mask, criterion, alpha = 10):
    base_loss = criterion(input, target)
    mask_loss = criterion(input * mask, target * mask)
    return (base_loss + alpha * mask_loss)


lambda_l1 = 100  # Weight for L1 loss

# Initialize generator and discriminator
#G = FCNGenerator(input_channels=3, output_channels=3).to(device)
G = FCNGeneratorSimple(input_channels=3, output_channels=3).to(device)
D = Discriminator(input_channels=3).to(device)

optimizer_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

"""
model_save_path = './model_weights_masked_loss_30.pth'
G.load_state_dict(torch.load(model_save_path))
G.to(device)

D_save_path = './discriminator_weights_mask_loss_30.pth'
D.load_state_dict(torch.load(D_save_path))
D.to(device)
"""
#scheduler_G = ExponentialLR(optimizer_G, gamma=0.95)
#scheduler_D = ExponentialLR(optimizer_D, gamma=0.95)

"""
Training Loop
"""


alpha = 10
num_epochs = 50

for epoch in range(num_epochs):
    if epoch == 10:
        model_save_path = './model_weights_masked_loss_10.pth'
        torch.save(G.state_dict(), model_save_path)
        D_save_path = './discriminator_weights_mask_loss_10.pth'
        torch.save(D.state_dict(), D_save_path)
    if epoch == 20:
        model_save_path = './model_weights_masked_loss_20.pth'
        torch.save(G.state_dict(), model_save_path)
        D_save_path = './discriminator_weights_mask_loss_20.pth'
        torch.save(D.state_dict(), D_save_path)
    if epoch == 30:
        model_save_path = './model_weights_masked_loss_30.pth'
        torch.save(G.state_dict(), model_save_path)
        D_save_path = './discriminator_weights_mask_loss_30.pth'
        torch.save(D.state_dict(), D_save_path)
    for i, (composite_images, real_images, masks, _) in enumerate(dataloader):

        batch_size = real_images.size(0)

        # Prepare labels
        real_labels = torch.ones(batch_size, 1, device=device).to(device)
        fake_labels = torch.zeros(batch_size, 1, device=device).to(device)

        real_images, composite_images, masks = real_images.to(device), composite_images.to(device), masks.to(device)

        # ---------------------
        # Train Discriminator
        # ---------------------

        # Train Discriminator with real images
        optimizer_D.zero_grad()
        D_real_loss = criterion(D(real_images), real_labels)

        # Generated / Harmonized images
        harmonized_img = G(composite_images)
        harmonized_images = adjust_dimensions(harmonized_img, real_images.shape)
        D_fake_loss = criterion(D(harmonized_images.detach()), fake_labels)

        D_loss = D_real_loss + D_fake_loss
        D_loss.backward()
        optimizer_D.step()
        # Train Generator
        optimizer_G.zero_grad()
        #G.zero_grad()
        fake_output = D(harmonized_images)
        gen_loss_GAN = nn.BCELoss()(fake_output,
                                    torch.ones_like(fake_output))  # Generator tries to fool the discriminator
        gen_loss_L1 = masked_loss(harmonized_images, real_images, masks, criterion=nn.L1Loss(),
                                  alpha=alpha)  # L1 loss on harmonization
        gen_loss = gen_loss_GAN + lambda_l1 * gen_loss_L1
        gen_loss.backward()
        optimizer_G.step()

        #scheduler_G.step()
        #scheduler_D.step()

        # Print some loss stats
        if i % 100 == 0:  # Print every 100 mini-batches
            print(
                f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] Gen_Loss: {gen_loss.item()}')

print('Training finished.')


###Save the trained weights to local

model_save_path = './model_weights_masked_loss_50.pth'
torch.save(G.state_dict(), model_save_path)

D_save_path = './discriminator_weights_masked_loss_50.pth'
torch.save(D.state_dict(), D_save_path)



###Save the output images to be compared
output_dir = './final_output_masked_loss_HFlickr'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with torch.no_grad(): 
    for i, (composite_images, _, _, comp_names) in enumerate(test_dataloader):
        composite_images = composite_images.to(device)
        harmonized_images = G(composite_images)

        # Save images to local
        for j, image in enumerate(harmonized_images):
            vutils.save_image(image, os.path.join(output_dir, comp_names[j]))
