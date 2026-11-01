# Entraînement mono-GPU de la tâche prétexte "position relative" sur CIFAR10.
# Même principe que train_self_supervised_task.py : on entraîne l'encodeur sur la tâche
# prétexte et, en parallèle, un classifieur linéaire sur les features gelées (zs.detach()).
# Le checkpoint a le même format, donc load_model_and_data('relative_position') marche.
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from deepul_helper.data import get_datasets
from deepul_helper.tasks import RelativePosition
from deepul_helper.demos import evaluate_classifier


def main():
    epochs = 30
    out_dir = 'results/cifar10_relative_position'
    os.makedirs(out_dir, exist_ok=True)

    train_dset, test_dset, n_classes = get_datasets('cifar10', 'relative_position')
    train_loader = DataLoader(train_dset, batch_size=128, shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_dset, batch_size=128, num_workers=4)

    model = RelativePosition('cifar10', n_classes).cuda()
    classifier = model.construct_classifier().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    optimizer_linear = torch.optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    scheduler_linear = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_linear, epochs)

    for epoch in range(epochs):
        model.train()
        classifier.train()
        for images, target in train_loader:
            images, target = images.cuda(), target.cuda()
            out, zs = model(images)

            # tâche prétexte
            optimizer.zero_grad()
            out['Loss'].backward()
            optimizer.step()

            # classifieur linéaire (n'influence pas l'encodeur)
            loss_linear = F.cross_entropy(classifier(zs.detach()), target)
            optimizer_linear.zero_grad()
            loss_linear.backward()
            optimizer_linear.step()

        scheduler.step()
        scheduler_linear.step()

        model.eval()
        classifier.eval()
        test_acc1, _ = evaluate_classifier(model, classifier, test_loader)
        print(f"Epoch {epoch+1}/{epochs} - pretexte loss {out['Loss'].item():.3f} "
              f"acc {out['Acc1'].item():.1f}% - linéaire test {test_acc1:.2f}%", flush=True)

        torch.save({'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'state_dict_linear': classifier.state_dict()},
                   os.path.join(out_dir, 'model_best.pth.tar'))


if __name__ == '__main__':
    main()
