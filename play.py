import math
import os
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, Iterable, Mapping, Callable

import inverse_optical_flow
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet
from scipy.ndimage import map_coordinates
from tqdm import tqdm


def load_image(image_path) -> Image.Image:
    image = Image.open(image_path)
    return image


def image_to_tensor(image: Image.Image | np.ndarray, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return transform(image)


def tensor_to_image(tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> Image.Image:
    transform = transforms.Compose([
        transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist()),
        transforms.ToPILImage()
    ])
    return transform(tensor)


def flatten_values(a: Iterable | Mapping | Any):
    if isinstance(a, (dict, )):
        for x in a.values():
            yield from flatten_values(x)
    elif isinstance(a, (list, tuple, set)):
        for x in a:
            yield from flatten_values(x)
    else:
        yield a


def total_variation2d(x: torch.Tensor):
    return torch.sum(torch.abs(x[:, :, :-1] - x[:, :, 1:])) + torch.sum(torch.abs(x[:, :-1, :] - x[:, 1:, :]))


@contextmanager
def register_hooks(
        model: torch.nn.Module,
        hook: Callable,
        **kwargs
):
    handles = []
    try:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                hook: Callable = partial(hook, name=name, **kwargs)
                handle = module.register_forward_hook(hook)
                handles.append(handle)
        yield handles
    finally:
        for handle in handles:
            handle.remove()


def stat_recorder_hook(
        module: torch.nn.Module,
        input: torch.Tensor,
        output: torch.Tensor,
        name: str,
        *,
        eps: float | torch.Tensor = 1e-6,
        storage: dict[str, dict[str, torch.Tensor]]
):
    mean = output.mean(dim=[0, 2, 3])
    std = output.std(dim=[0, 2, 3], unbiased=False)
    # skewness = ((output - mean[None, :, None, None]) ** 3).mean(dim=[0, 2, 3]) / (std ** 3 + eps)
    # kurtosis = ((output - mean[None, :, None, None]) ** 4).mean(dim=[0, 2, 3]) / (std ** 4 + eps)
    # assert torch.isfinite(torch.cat([mean, std, skewness, kurtosis])).all()
    storage[name] = {
        "mean": mean,
        "std": std,
        # "skewness": skewness,
        # "kurtosis": kurtosis,
    }


def get_stats(model: torch.nn.Module, image: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
    stats = {}
    with register_hooks(model, stat_recorder_hook, storage=stats):
        _ = model(image[None])
    return stats


def alpha_composite(im1, im2, opacity1=1.0, opacity2=1.0):
    """
    Input: (4, H, W) ndarray, RGBA in 0-255
    Output: (4, H, W) ndarray, RGBA in 0-255
    """
    # Validate the opacity values
    if not 0 <= opacity1 <= 1 or not 0 <= opacity2 <= 1:
        raise ValueError('Opacity must be between 0 and 1')

    # Assuming the last channel is the alpha channel
    # Scale the alpha channels by the provided opacity values
    im1[3, :, :] = im1[3, :, :] * opacity1
    im2[3, :, :] = im2[3, :, :] * opacity2

    # Normalize the alpha channels to be between 0 and 1
    im1_alpha = im1[3, :, :] / 255.0
    im2_alpha = im2[3, :, :] / 255.0

    # Compute the composite alpha channel
    composite_alpha = im1_alpha + im2_alpha * (1 - im1_alpha)

    # Handle case where composite_alpha is 0 to avoid divide by zero error
    mask = composite_alpha > 0
    composite_alpha = np.where(mask, composite_alpha, 1)

    # Compute the composite image
    composite_image = np.empty_like(im1)
    for channel in range(3):  # Assuming the first 3 channels are RGB
        composite_image[channel, :, :] = (
            im1[channel, :, :] * im1_alpha
            + im2[channel, :, :] * im2_alpha * (1 - im1_alpha)
        ) / composite_alpha

    # Add the composite alpha channel to the image
    composite_image[3, :, :] = composite_alpha * 255

    return composite_image.astype(np.uint8)


def warp(image: np.ndarray, backward_flow: np.ndarray, order=3) -> np.ndarray:
    channels, height, width = image.shape
    index_grid = np.mgrid[0:height, 0:width].astype(float)
    # Widely, first channel is horizontal x-axis flow, the second channel is vertical y-axis flow.
    coordinates = index_grid + backward_flow[::-1]
    remapped = np.empty(image.shape, dtype=image.dtype)
    for i in range(channels):
        remapped[i] = map_coordinates(image[i], coordinates, order=order, mode='constant', cval=0)
    return remapped


class PlateuPruner:
    def __init__(self, patience: int = 5, min_delta: float = 0.0, target='minimize'):
        """
        Initializes the EarlyStopping instance.
        :param patience: Number of epochs with no improvement after which training will be stopped.
        :param min_delta: Minimum change in the monitored quantity to qualify as an improvement.
        :param target: 'minimize' for minimizing a metric (like loss), 'maximize' for maximizing a metric (like accuracy).
        """
        self.patience = patience
        self.min_delta = min_delta
        self.target = target
        self.counter = 0
        self.best_score = math.inf if target == 'minimize' else -math.inf
        self.early_stop = False
        self.is_best = True

    def report(self, score):
        """
        Reports the latest metric value and checks if it's an improvement.
        :param metric_value: The latest metric value (e.g., validation loss or accuracy).
        """
        if self.target == 'minimize':
            self.is_best = score < self.best_score
            is_improvement = score < self.best_score - self.min_delta
        else:
            self.is_best = score > self.best_score
            is_improvement = score > self.best_score + self.min_delta

        if is_improvement:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.is_best, self.early_stop


def main():
    root_path = "/Users/gimminjin/Video-tracking/Video-tracking"
    frames_path = f"{root_path}/frame"
    flow_path = f"{root_path}/flow"
    # style_filepath = "examples/style/confocal-microscopy.jpg"
    # style_filepath = "examples/style/doodle.png"
    # style_filepath = "examples/style/lego1.jpg"
    # style_filepath = "examples/style/lego6.jpg"
    # style_filepath = "examples/style/lego7.webp"
    # style_filepath = "examples/style/lego8.webp"
    # style_filepath = "examples/style/matrix.jpg"
    # style_filepath = "examples/style/mush.png"
    style_filepath = "/Users/gimminjin/Video-tracking/Video-tracking/Starry-Night-canvas-Vincent-van-Gogh-New-1889.webp"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = torchvision.models.vgg16(pretrained=True).features
    # model = torchvision.models.resnet18(pretrained=True)
    model = EfficientNet.from_pretrained('efficientnet-b0')
    # model = EfficientNet.from_pretrained('efficientnet-b4')

    # Disable grad
    for param in model.parameters():
        param.requires_grad_(False)
    # Disable running stats
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.track_running_stats = False
    # Unset "inplace"
    for module in model.modules():
        if hasattr(module, "inplace"):
            module.inplace = False

    model = model.to(device)
    model.eval()
    # print(f"model: {model}")

    named_layers = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Conv2d)]
    style_content_weights_per_layer = {
        name:
            # Constant style and content weight per layer
            (1, 1)
            # Gradually decreasing style, gradually increasing content
            # (1 - i / (len(named_layers) - 1),
            #  i / (len(named_layers) - 1))
            # First half of layers are style, second half are content
            # (1 if i < len(named_layers) // 2 else 0,
            #  0 if i < len(named_layers) // 2 else 1)
            # First quarter is none, second and third quarter of layers are style, fourth quarter is a content
            # (1 if i >= len(named_layers) // 4 and i < len(named_layers) * 3 // 4 else 0,
            #  1 if i >= len(named_layers) * 3 // 4 else 0)
            # First quarter is a style, last quarter is a content
            # (1 if i < len(named_layers) // 4 else 0,
            #  1 if i >= len(named_layers) * 3 // 4 else 0)
            # First three quarters is a style, last quarter is a content
            # (1 if i < len(named_layers) * 3 // 4 else 0,
            #  0 if i < len(named_layers) * 3 // 4 else 1)
        for i, name in enumerate(named_layers)
    }
    print(f"style_content_weights_per_layer: {style_content_weights_per_layer}")

    style_weight = 1e+1
    content_weight = 1e+0
    temporal_weight = 1e+2
    total_variation_weight = 0

    mean = torch.tensor((0.485, 0.456, 0.406)).to(device)
    std = torch.tensor((0.229, 0.224, 0.225)).to(device)

    # Clamping range for normalized image
    min_vals = (0 - mean) / std
    max_vals = (1 - mean) / std

    style_image = load_image(style_filepath)
    style_image = image_to_tensor(style_image, mean, std).to(device)
    style_stats = get_stats(model, style_image)
    # print(f"style_stats.keys(): {style_stats.keys()}")
    assert torch.isfinite(torch.cat(list(flatten_values(style_stats)))).all()

    frame_indices = sorted([int(x.name) for x in os.scandir(frames_path) if x.is_dir()])
    print(f"frames: {frame_indices}")

    styled = None
    styled_prev_warped = None

    for frame_i in frame_indices:
        frame_path = f"{frames_path}/{frame_i}"
        for filepath in Path(frame_path).glob("styled_*.png"):
            os.remove(filepath)

    for frame_i in tqdm(frame_indices, desc="Styling"):
        frame_path = f"{frames_path}/{frame_i}"

        content_filepath = f"{frame_path}/content.qoi"
        styled_filepath = f"{frame_path}/styled.pt"
        forward_flow_filepath = f"{flow_path}/flow_{frame_i-1}_to_{frame_i}.npz"
        backward_flow_filepath = f"{flow_path}/flow_{frame_i}_to_{frame_i-1}.npz"

        content = load_image(content_filepath)
        content = image_to_tensor(content, mean, std).to(device)
        tensor_to_image(content, mean, std).save(f"{frame_path}/0_content.png")

        if styled is None:
            styled = content.clone().to(device)
            disocclusion_mask = torch.ones_like(styled, device=device)
        else:
            forward_flow = np.load(forward_flow_filepath)

            with torch.no_grad():
                styled.data = styled.data.clamp_(min_vals[:, None, None], max_vals[:, None, None])
            np_styled = tensor_to_image(styled, mean, std).convert('RGBA')
            np_styled = np.array(np_styled).transpose(2, 0, 1)

            np_content = tensor_to_image(content, mean, std).convert('RGBA')
            np_content = np.array(np_content).transpose(2, 0, 1)

            # .npz 파일 읽기
            forward_flow = np.load(forward_flow_filepath)

            # 필요한 데이터 가져오기 (arr_0 키 사용)
            forward_flow_data = forward_flow['arr_0']

            # Optical Flow 계산
            forward_flow_inv, disocclusion_mask = inverse_optical_flow.max_method(forward_flow_data)
            np_styled = warp(np_styled, forward_flow_inv, order=3)
            np_styled[3, :, :] = np_styled[3, :, :] * (1 - disocclusion_mask)
            np_styled = alpha_composite(np_styled, np_content)
            np_styled = np_styled[:3].transpose(1, 2, 0)
            styled_prev_warped = image_to_tensor(np_styled, mean, std).to(device)

            styled = content.clone().to(device)
            disocclusion_mask = torch.from_numpy(disocclusion_mask).to(device).to(torch.float32)

        styled.requires_grad_(True)

        def get_opimization_strategy():
            # optimizer = torch.optim.LBFGS([styled], lr=1, max_iter=40)
            optimizer = torch.optim.Adam([styled], lr=0.1)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)
            return optimizer, scheduler

        optimizer, scheduler = get_opimization_strategy()

        epochs = 1 if isinstance(optimizer, torch.optim.LBFGS) else 40

        content_stats = get_stats(model, content)
        assert torch.isfinite(torch.cat(list(flatten_values(content_stats)))).all()

        pruner = PlateuPruner(patience=20, min_delta=0.01, target="minimize")
        best_styled = styled.clone().to(device)

        for epoch in range(epochs):
            iteration = 0

            def closure() -> float:
                nonlocal iteration
                print(f"closure(): frame_i: {frame_i}, epoch: {epoch}, iteration: {iteration}")
                optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    styled.data = styled.data.clamp_(min_vals[:, None, None], max_vals[:, None, None])
                tensor_to_image(styled, mean, std).save(f"{frame_path}/styled_{epoch}_{iteration}.png")
                styled_stats = get_stats(model, styled)

                loss = torch.zeros(1, device=device)
                for name, (style_w, content_w) in style_content_weights_per_layer.items():
                    # If requested layer weight not found in style or content stats, skip it
                    if name not in styled_stats or name not in style_stats or name not in content_stats:
                        continue
                    loss += style_weight * style_w * torch.nn.functional.mse_loss(
                        torch.cat(list(flatten_values(styled_stats[name]))),
                        torch.cat(list(flatten_values(style_stats[name]))))
                    loss += content_weight * content_w * torch.nn.functional.mse_loss(
                        torch.cat(list(flatten_values(styled_stats[name]))),
                        torch.cat(list(flatten_values(content_stats[name]))))
                loss += total_variation_weight * total_variation2d(styled)
                if styled_prev_warped is not None:
                    loss += temporal_weight * torch.nn.functional.mse_loss(styled * (1 - disocclusion_mask),
                                                                           styled_prev_warped * (1 - disocclusion_mask))
                assert not torch.isnan(loss).any()

                print(f"loss: {loss.item()}, lr: {optimizer.param_groups[0]['lr']}")
                iteration += 1
                loss.backward()
                return loss.item()

            closure_loss = optimizer.step(closure)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(closure_loss)
            else:
                scheduler.step()

            is_best, should_prune = pruner.report(closure_loss)
            if is_best:
                print(f"New best at frame {frame_i}, epoch {epoch}, iteration {iteration}")
                best_styled = styled.clone().to(device)
            if should_prune:
                print(f"Early stopping at frame {frame_i}, epoch {epoch}, iteration {iteration}")
                break

        styled = best_styled
        torch.save(styled, styled_filepath)


if __name__ == '__main__':
    main()
