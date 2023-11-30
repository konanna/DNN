import torch
import numpy as np
import torch.nn.functional as F
from src.diffraction import DiffractiveLayer, Lens
from tqdm import tqdm
from copy import deepcopy
from torchvision import transforms

DETECTOR_POS = [
    (46, 66, 46, 66), (46, 66, 93, 113), (46, 66, 140, 160),
    (85, 105, 46, 66), (85, 105, 78, 98), (85, 105, 109, 129),
    (85, 105, 140, 160), (125, 145, 46, 66), (125, 145, 93, 113),
    (125, 145, 140, 160)
]


class Trainer:
    """
    Класс для тренировки оптической нейронной сети.
    param model: Модель нейронной сети для обучения
    param detector_pos: список позиций детекторов для классификации
    param padding: число нулевых пикселей, которое нужно добавить к изображению на вход нейронной сети.
        Если число пикселей изображения совпадает с числом пикселей в фазовой маске нейронной сети, то padding = 0
    param device: где будет проходить обучение сети 'cpu'/'cuda'
    """

    def __init__(self, model, detector_pos=DETECTOR_POS, padding=58, device='cpu'):
        self.detector_pos = detector_pos
        self.model = model
        self.padding = padding
        self.device = device

    def detector_region(self, x):
        """
        Подсчет интенсивности, которая приходится на каждый детектор в конце оптической нейронной сети.
        param x: распределение интенсивности на выходе нейронной сети
        return: тензор с суммарной интенсивностью, приходящейся на каждый детектор
        """
        detectors_list = []
        full_int = x.sum(dim=(1, 2))
        for det_x0, det_x1, det_y0, det_y1 in self.detector_pos:
            detectors_list.append(
                (x[:, det_x0: det_x1 + 1, det_y0: det_y1 + 1].sum(dim=(1, 2)) / full_int).unsqueeze(-1))
        return torch.cat(detectors_list, dim=1)

    def epoch_step(self, batch, unconstrain_phase=False, thickness_discretization=0, validation=False):
        """
        Обработка одного батча в процессе тренировки.
        param batch: (imgs, labels) батч с изображениями и их метками
        """
        images, labels = batch
        images = images.to(self.device)
        images = F.pad(images, pad=(self.padding, self.padding, self.padding, self.padding))
        labels = labels.to(self.device)

        out_img, _ = self.model(images, unconstrain_phase, thickness_discretization, validation)

        out_label = self.detector_region(out_img)
        _, predicted = torch.max(out_label.data, 1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        # if validation:
        #     loss = self.val_loss_function(out_img, labels)
        # else:
        loss = self.loss_function(out_img, labels)
        return loss, correct, total

    def train(self,
              loss_function,
              # val_loss_function,
              optimizer,
              trainloader,
              testloader,
              epochs=10,
              unconstrain_phase=False,
              thickness_discretization=0):
        """
        Функция для тренировки сети.
        """
        hist = {'train loss': [],
                'test loss': [],
                'train accuracy': [],
                'test accuracy': []}
        best_acc = 0
        self.loss_function = loss_function
        # self.val_loss_function = val_loss_function
        for epoch in range(epochs):
            ep_loss = 0
            self.model.train()
            correct = 0
            total = 0
            for batch in tqdm(trainloader):
                loss, batch_correct, batch_total = self.epoch_step(batch, unconstrain_phase,
                                                                   thickness_discretization)
                ep_loss += loss.item()
                correct += batch_correct
                total += batch_total

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            hist['train loss'].append(ep_loss / len(trainloader))
            hist['train accuracy'].append(correct / total)

            ep_loss = 0
            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in tqdm(testloader):
                    loss, batch_correct, batch_total = self.epoch_step(batch, unconstrain_phase,
                                                                       thickness_discretization, validation=True)
                    ep_loss += loss.item()
                    correct += batch_correct
                    total += batch_total
            hist['test loss'].append(ep_loss / len(testloader))
            hist['test accuracy'].append(correct / total)

            if hist['test accuracy'][-1] > best_acc:
                best_acc = hist['test accuracy'][-1]
                best_model = deepcopy(self.model)

            print(
                f"\nEpoch={epoch + 1} train loss={hist['train loss'][epoch]:.4}, test loss={hist['test loss'][epoch]:.4}")
            print(f"train acc={hist['train accuracy'][epoch]:.4}, test acc={hist['test accuracy'][epoch]:.4}")
            print("-----------------------")

        return hist, best_model

    def validate(self, dataloader, unconstrain_phase=False):
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(dataloader):
                images = images.to(self.device)
                images = F.pad(torch.squeeze(images), pad=(self.padding, self.padding, self.padding, self.padding))
                labels = labels.to(self.device)

                out_img, _ = self.model(images, unconstrain_phase)

                out_label = self.detector_region(out_img)
                _, predicted = torch.max(out_label.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        return correct / total


class MaskLayer(torch.nn.Module):
    """
    Класс для амплитудно-фазовой маски
    """

    def __init__(self, distance_before_mask=None, wl=532e-9,
                 N_pixels=1000, pixel_size=1e-6,
                 N_neurons=20, neuron_size=2e-6,
                 include_amplitude=False, n=None):
        """
        :param distance_before_mask: расстояние, которое проходит излучение до маски
        :param wl: длина волны излучения, для многоканальных изображений может быть задана списком соответствующей длины
        :param N_pixels: число пикселей в изображении
        :param pixel_size: размер одного пикселя
        :param N_neurons: число нейронов (пикселей в маске)
        :param neuron_size: размер нейронов - д.б. кратно больше размера пикселя
        :param include_amplitude: применять ли амплитудную модуляцию
        :param n: комплексный показатель преломления, для многоканальных изображений - список, len(n) = len(wl)
        """
        super(MaskLayer, self).__init__()

        self.diffractive_layer = None
        if distance_before_mask:
            self.diffractive_layer = DiffractiveLayer(wl, N_pixels, pixel_size, distance_before_mask)

        wl = torch.tensor(np.array(wl).reshape(-1, 1, 1), dtype=torch.float32)
        self.register_buffer('wl', wl)

        n = torch.tensor(np.array(n).reshape(-1, 1, 1), dtype=torch.complex64)
        if n.size(0) == wl.size(0) or n.size(0) == 1:
            self.register_buffer('n', n)
        else:
            raise Exception("Numbers of wl does not match number of n")

        self.phase = torch.nn.Parameter(torch.zeros([wl.size(0), N_neurons, N_neurons],
                                                    dtype=torch.float32))
        # self.phase = torch.nn.Parameter(torch.rand([wl.size(0), N_neurons, N_neurons],
        #                                             dtype=torch.float32))
        if include_amplitude and (n is None):
            self.amplitude = torch.nn.Parameter(torch.zeros([wl.size(0), N_neurons, N_neurons],
                                                            dtype=torch.float32) + 1)
        self.phase_amp_mod = include_amplitude
        self.N_pixels = N_pixels
        self.pixel_size = pixel_size
        self.N_neurons = N_neurons
        self.neuron_size = neuron_size

    def forward(self, E, unconstrain_phase=False, thickness_discretization=0, validation=False):
        out = E
        if self.diffractive_layer is not None:
            out = self.diffractive_layer(out)
        
        constr_phase = self.constrain_phase(unconstrain_phase,
                                            thickness_discretization,
                                            validation)

        modulation = torch.cos(constr_phase) + 1j * torch.sin(constr_phase)
        if self.phase_amp_mod:
            if self.n is None:
                constr_amp = F.relu(self.amplitude) / F.relu(self.amplitude).max()
            else:
                constr_amp = torch.exp(- self.n.imag * 2 * np.pi / self.wl
                                       * self.calc_thickness(constr_phase))
            modulation = constr_amp * modulation
        
        modulation = self.mask_resize(modulation)
        out = modulation * out
        return out

    def constrain_phase(self, unconstrain_phase=False,
                              thickness_discretization=0,
                              validation=False):
        if unconstrain_phase:
            constr_phase = self.phase
        else:
            constr_phase = 2 * np.pi * torch.sigmoid(self.phase)
            if thickness_discretization:
                if validation:
                    constr_phase = self.true_step_function(constr_phase, 
                    thickness_discretization)
                else:
                    constr_phase = self.sigmoid_step_function(constr_phase,
                    thickness_discretization)
        return constr_phase
    
    def mask_resize(self, modulation):
        """
        Тензор с весами, соответствующими толщинам пикселей фазовой маски,
        преобразуется под размер реальной фазовой маски, и дополняется
        до размера изображения в системе (по краям фазовая задержка не вносится)
        """

        mask_size_in_pixels = self.N_neurons * int(self.neuron_size //
                                                    self.pixel_size)
        mask_padding = (self.N_pixels - mask_size_in_pixels) // 2
        mask_padding = (mask_padding, mask_padding, mask_padding, mask_padding)
        modulation = transforms.functional.resize(modulation.real,
                                                  mask_size_in_pixels,
                                                  transforms.InterpolationMode.NEAREST) \
                     + transforms.functional.resize(modulation.imag,
                                                    mask_size_in_pixels,
                                                    transforms.InterpolationMode.NEAREST) * 1j
        modulation = transforms.functional.pad(modulation, mask_padding, 1)
        # modulation = ()
        return modulation

    def calc_thickness(self, phase=None):
        """
        Вычисление толщины, соответствующей набегу фаз в маске
        """
        if phase is None:
            phase = 2 * np.pi * torch.sigmoid(self.phase)
        thickness = self.wl * phase / (2 * np.pi * np.real(self.n))
        return thickness

    def sigmoid_step_function(self, phase, thickness_discretization, alpha=100):
        new_phase = torch.zeros(phase.shape, dtype=torch.float32, device=phase.device)
        phase_discr = (thickness_discretization * torch.real(self.n)/
                        self.wl).to(phase.device)
        phase_offset = 0.5 * phase_discr
        while phase_offset[0, 0, 0] < 2 * np.pi:
            new_phase = new_phase + phase_discr * torch.sigmoid(alpha * (phase - phase_offset))
            phase_offset = phase_offset + phase_discr
        return new_phase

    def true_step_function(self, phase, thickness_discretization):
        new_phase = torch.zeros(phase.shape, dtype=torch.float32, device=phase.device)
        phase_discr = thickness_discretization * np.real(self.n) / self.wl
        phase_offset = 0.5 * phase_discr
        zero_values = torch.zeros(phase.shape, dtype=torch.float32, device=phase.device)
        while phase_offset[0, 0, 0] < 2 * np.pi:
            new_phase = new_phase + phase_discr * torch.heaviside((phase - phase_offset), zero_values)
            phase_offset = phase_offset + phase_discr
        return new_phase


class new_Fourier_DNN(torch.nn.Module):
    """
    Fourier Diffractive Neural Network
    """

    def __init__(self,
                 num_layers=5,
                 wl=532e-9,
                 N_pixels=200,
                 pixel_size=1e-6,
                 N_neurons=40,
                 neuron_size=2e-6,
                 distance=5e-3,
                 lens_focus=100e-3,
                 include_amplitude_modulation=True,
                 dn=None):
        super(new_Fourier_DNN, self).__init__()
        self.lens_diffractive_layer = DiffractiveLayer(wl, N_pixels, pixel_size, lens_focus)
        self.lens = Lens(lens_focus, wl, N_pixels, pixel_size)
        self.first_diffractive_layer = DiffractiveLayer(wl, N_pixels, pixel_size, lens_focus - distance)
        self.mask_layers = torch.nn.ModuleList([MaskLayer(distance_before_mask=distance,
                                                          wl=wl,
                                                          N_pixels=N_pixels,
                                                          pixel_size=pixel_size,
                                                          N_neurons=N_neurons,
                                                          neuron_size=neuron_size,
                                                          include_amplitude=include_amplitude_modulation,
                                                          n=dn) for _ in range(0, num_layers)])

    def forward(self, E, unconsrtain_phase=False, thick_discr=0, validation=False):
        outputs = [E]
        E = self.lens_diffractive_layer(E)
        E = self.lens(E)
        E = self.first_diffractive_layer(E)
        outputs.append(E)
        for layer in self.mask_layers:
            E = layer(E, unconsrtain_phase, thick_discr, validation)
            outputs.append(E)
        E = self.lens_diffractive_layer(E)
        E = self.lens(E)
        outputs.append(E)
        E = self.lens_diffractive_layer(E)
        E_abs = torch.abs(E) ** 2
        return E_abs.sum(dim=1), outputs

    @property
    def device(self):
        return next(self.parameters()).device
