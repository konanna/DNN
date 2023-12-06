from itertools import chain
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn.functional as F
import numpy as np
import pandas as pd


def gaussian_square_detector(det_size, sigma = 0.2):
    x, y = np.meshgrid(np.linspace(-1,1,det_size), np.linspace(-1,1,det_size))
    r_sqr = x**2 + y**2
    gauss = 1/(np.sqrt(2 * np.pi) * sigma)*np.exp(-r_sqr / 2 / sigma ** 2)
    detector = torch.from_numpy(gauss)
    return detector


def generate_det_row(det_size, start_pos_x, start_pos_y, det_step, N_det):
    p = []
    for i in range(N_det):
        left = start_pos_x + i * (int(det_step) + det_size)
        right = left + det_size
        up = start_pos_y
        down = start_pos_y + det_size
        p.append((up, down, left, right))
    return p


def set_det_pos(det_size=20, edge_x=10, edge_y=20, N_pixels=200):
    p = []
    det_step_x_1 = (N_pixels - 2 * edge_x - 3 * det_size) // 2
    det_step_x_2 = (N_pixels - 2 * edge_x - 4 * det_size) // 3
    det_step_y = (N_pixels - 2 * edge_y - 3 * det_size) // 2 + det_size
    p.append(generate_det_row(det_size, edge_x, edge_y, det_step_x_1, 1))
    p.append(generate_det_row(det_size, edge_x, edge_y + det_step_y, det_step_x_2, 0))
    p.append(generate_det_row(det_size, edge_x, edge_y + 2 * det_step_y, det_step_x_1, 1))
    return list(chain.from_iterable(p))


def get_detector_imgs(det_size=20, edge_x=10, edge_y=20, N_pixels=200,
                      visualize=True, is_gaussian=False):
    detector_pos = set_det_pos(det_size, edge_x, edge_y, N_pixels)
    labels_image_tensors = torch.zeros((2, N_pixels, N_pixels), dtype=torch.double)
    gauss_detector = gaussian_square_detector(det_size, sigma=0.5)
    for ind, pos in enumerate(detector_pos):
        pos_l, pos_r, pos_u, pos_d = pos
        labels_image_tensors[ind, pos_l + 1:pos_r + 1,
                             pos_u + 1:pos_d + 1] = gauss_detector[None,:,:]
        # labels_image_tensors[ind, pos_l + 1:pos_r + 1,
        #                      pos_u + 1:pos_d + 1] = 1
        # labels_image_tensors[ind] = labels_image_tensors[ind]
    if visualize:
        plt.imshow(np.zeros((N_pixels, N_pixels)))
        for det in detector_pos:
            rect = patches.Rectangle((det[2], det[0]), det_size, det_size, linewidth=1, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)
        plt.show()
    return labels_image_tensors, detector_pos


def visualize(model, example, padding=58):
    ex = F.pad(example[0], pad=(padding, padding, padding, padding))
    device = model.device
    out = model(ex.to(device))
    plt.subplot(1, 2, 1)
    plt.imshow(ex[0], interpolation='none')
    plt.title(f'Input image with label {example[1]}')
    output_image = out.detach().cpu()
    plt.subplot(1, 2, 2)
    plt.imshow(output_image, interpolation='none')
    plt.title(f'Output image')
    # plt.colorbar()
    plt.show()


def mask_visualization(model, thickness_discretization=0,
                       mode='phase', transpose=False):
    wl = model.mask_layers[0].wl.detach().cpu()
    n = np.real(model.mask_layers[0].n.detach().cpu())
    n_wl = wl.shape[0]
    n_layers = len(model.mask_layers)
    if transpose:
        plt.figure(figsize=(5 * n_layers, 4 * n_wl))
    else:
        plt.figure(figsize=(5 * n_wl, 4 * n_layers))
    for i, mask in enumerate(model.mask_layers):
        phase = torch.sigmoid(mask.phase).detach().cpu()
        if thickness_discretization != 0:
            phase = mask.sigmoid_step_function(phase, thickness_discretization)
        for j in range(n_wl):
            colorbar_label = 'Phase, deg.'
            if transpose:
                plt.subplot(n_wl, n_layers, (i + 1) + j * n_layers)
            else:
                plt.subplot(n_layers, n_wl, (j + 1) + i * n_wl)
            if mode == 'phase':
                plt.imshow(phase[j, :, :] * 360, interpolation='none')
            elif mode == 'thickness':
                plt.imshow((phase * wl * 10 ** 6 / n)[j, :, :],
                           interpolation='none')
                colorbar_label = 'Thickness, um'
            else:
                print(f'Do not support mode = "{mode}". Only "thickness" or "phase"')
            plt.title(f'Mask {j + 1} of layer {i + 1}')
            ax = plt.gca()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(label=colorbar_label, cax=cax)


def visualize_n_samples(model,
                        dataset,
                        n,
                        padding=58,
                        detector_pos=None,
                        unconstain_phase=False,
                        thickness_discretization=0,
                        seed=17):
    plt.figure(figsize=(5 * n, 8))
    np.random.seed(seed)
    rand_ind = np.random.choice(range(len(dataset)), size=n, replace=False)
    device = model.device
    for number, ind in enumerate(rand_ind):
        ex = F.pad(dataset[ind][0], pad=(padding, padding, padding, padding))
        out, _ = model(ex[None, :, :, :].to(device), unconstain_phase, thickness_discretization)
        plt.subplot(2, n, number + 1)
        plt.imshow(ex[0], interpolation='none')
        plt.title(f'Input image with label {dataset[ind][1]}')
        output_image = out[0].detach().cpu()
        plt.subplot(2, n, n + number + 1)
        plt.imshow(output_image, interpolation='none')
        if detector_pos is not None:
            det_size = detector_pos[0][1] - detector_pos[0][0]
            for det in detector_pos:
                rect = patches.Rectangle((det[2], det[0]), det_size, det_size, linewidth=1, edgecolor='r',
                                         facecolor='none')
                plt.gca().add_patch(rect)
    plt.title(f'Output image')


def save_tensor(tensor, filename, file_format="pt", float_format=None):
    if file_format == "pt":
        torch.save(tensor, filename + ".pt")
    elif file_format == "csv":
        df = pd.DataFrame(tensor.detach().cpu().numpy())
        df.to_csv(filename + ".csv", float_format=float_format, index=False)


def save_masks(model, saving_mode=0, thickness_discretization=0,
               file_format="pt", float_format=None, filename_prefix=""):
    """
    Сохранение маски в файл
    param model: Модель нейронной сети
    param saving_mode: Режим сохранения:
                       0 (default) - сохранение фазы как параметра сети,
                       1 - толщины пикселей для печати
    param thickness_discretization: Дискретизация по высоте при печати - tbd!!!
                                    (0 - default, толщина меняется непрерывно)
    param file_format: Расширение файла (pt - default)
    param float_format: Формат записи числа в файл
    param filename_prefix: Префикс в названии файла формата:
                           "{filename_prefix}_mask_{}_layer_{}.{file_format}"
    """
    # thickness_discretization - to be done

    wl = model.mask_layers[0].wl
    n = np.real(model.mask_layers[0].n)
    n_wl = wl.shape[0]
    for i, mask in enumerate(model.mask_layers):
        out = mask.phase
        if saving_mode == 1:
          out = torch.sigmoid(out) * wl * 10 ** 6 / n
        
        for j in range(n_wl):
            filename = filename_prefix + 'mask_{}_layer_{}'.format(j, i)
            save_tensor(out[j, :, :], filename, file_format, float_format)


def load_masks_from_file(model, loading_mode=0,
                         file_format="pt", filename_prefix=""):
  """
    Загрузка маски из файла
    param model: Модель нейронной сети
    param loading_mode: Режим загрузки:  - to be done
                        0 (default) - фазы как парам,
                        1 - толщины пикселей для печати
    param float_format: Формат записи числа в файл
    param filename_prefix: Префикс в названии файла формата:
                           "{filename_prefix}_mask_{}_layer_{}.{file_format}"
  """
  # loading_mode=1 - to be done

  n_masks = len(model.mask_layers)
  n_layers = model.mask_layers[0].wl.size(0)
  n = torch.real(model.mask_layers[0].n)
  wl = model.mask_layers[0].wl
  if (filename_prefix):
    filename_prefix += "_"
  for i in range(n_masks):
    filename = (filename_prefix +
                'mask_{}_layer_{}.'.format(i, 0) +
                file_format)
    phase = torch.load(filename).to(model.device)
    for j in range(1, n_layers):
      filename = (filename_prefix +
                  'mask_{}_layer_{}.'.format(i, j) +
                  file_format)
      phase = torch.cat((phase, torch.load(filename).to(model.device)), dim=0)
    if loading_mode == 1:
      phase /= wl * 10 ** 6 / n
      phase = torch.logit
    model.mask_layers[i].phase = torch.nn.Parameter(phase)


def prop_vis(model, example, padding=58, mode='abs', name_list=None):
    ex = F.pad(example, pad=(padding, padding, padding, padding))
    device = model.device
    final, imgs = model(ex.to(device))
    for ind, img in enumerate(imgs):
        if mode == 'abs':
            plt.imshow(img[0].abs().detach().cpu())
        elif mode == 'phase':
            plt.imshow(img[0].abs().detach().cpu())
        else:
            print(f'Do not support mode = "{mode}". Only "abs" or "phase"')
        if name_list is not None:
            plt.title(label=name_list[ind])
        plt.show()
    plt.imshow(final[0].abs().detach().cpu())
    if name_list is not None:
        plt.title(label=name_list[-1])
    plt.show()
