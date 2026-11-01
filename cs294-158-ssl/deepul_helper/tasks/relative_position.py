import torch
import torch.nn as nn
import torch.nn.functional as F


# Tâche prétexte de Doersch et al. 2015 (https://arxiv.org/abs/1505.05192) :
# on découpe l'image en grille 3x3, on prend le patch central + un de ses 8 voisins,
# et le réseau doit prédire la position (0..7) du voisin par rapport au centre.
class RelativePosition(nn.Module):
    metrics = ['Loss', 'Acc1']
    metrics_fmt = [':.4e', ':6.2f']

    img_size = 64    # images CIFAR/SVHN (32x32) agrandies en 64x64
    cell = 21        # taille d'une case de la grille 3x3
    patch = 16       # taille d'un patch -> 5 px de marge (gap + jitter)

    def __init__(self, dataset, n_classes):
        super().__init__()
        # Encodeur partagé (siamois) appliqué à chaque patch
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Tête qui prend les features des 2 patchs concaténées et prédit 8 positions
        self.head = nn.Sequential(
            nn.Linear(2 * 256, 512), nn.ReLU(),
            nn.Linear(512, 8)
        )
        self.latent_dim = 256 * 3 * 3
        self.dataset = dataset
        self.n_classes = n_classes

    def construct_classifier(self):
        return nn.Sequential(
            nn.BatchNorm1d(self.latent_dim, affine=False),
            nn.Linear(self.latent_dim, self.n_classes)
        )

    def _patch_features(self, patches):
        return F.adaptive_avg_pool2d(self.encoder(patches), 1).flatten(1)

    def _sample_pairs(self, images):
        # Pour chaque image : patch central + un voisin au hasard, avec un petit décalage
        # aléatoire (jitter) pour que le réseau ne puisse pas tricher avec les bords.
        n = images.shape[0]
        neighbors = [0, 1, 2, 3, 5, 6, 7, 8]  # cases de la grille 3x3 sauf le centre (4)
        labels = torch.randint(0, 8, (n,))  # tirés sur CPU pour éviter les synchros GPU
        offsets = torch.randint(0, self.cell - self.patch + 1, (n, 2, 2)).tolist()

        centers, others = [], []
        for i in range(n):
            for k, (case, out) in enumerate(((4, centers), (neighbors[labels[i]], others))):
                r = (case // 3) * self.cell + offsets[i][k][0]
                c = (case % 3) * self.cell + offsets[i][k][1]
                out.append(images[i, :, r:r + self.patch, c:c + self.patch])
        return torch.stack(centers), torch.stack(others), labels.to(images.device)

    def forward(self, images):
        images = F.interpolate(images, size=self.img_size, mode='bilinear', align_corners=False)
        center, other, labels = self._sample_pairs(images)

        feats = torch.cat([self._patch_features(center), self._patch_features(other)], dim=1)
        logits = self.head(feats)
        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=-1) == labels).float().mean() * 100.

        return dict(Loss=loss, Acc1=acc), self.encode(images, resized=True)

    def encode(self, images, resized=False):
        # Représentation de l'image entière : encodeur sur toute l'image,
        # puis moyenne par case de la grille 3x3 -> 256 x 3 x 3
        if not resized:
            images = F.interpolate(images, size=self.img_size, mode='bilinear', align_corners=False)
        return F.adaptive_avg_pool2d(self.encoder(images), 3).flatten(1)
