# Compare les 4 tâches prétextes (encodeurs pré-entraînés sur CIFAR10) en classification
# linéaire sur CIFAR10 et sur SVHN. Résultats : results/comparaison.csv et results/comparaison.png
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt

from deepul_helper.demos import load_model_and_data, train_linear_classifier, evaluate_classifier

tasks = ['context_encoder', 'rotation', 'simclr', 'relative_position']


def main():
    results = []
    for task in tasks:
        # CIFAR10 : classifieur linéaire déjà entraîné, stocké dans le checkpoint
        model, clf, _, test_loader = load_model_and_data(task, dataset='cifar10')
        top1, top5 = evaluate_classifier(model, clf, test_loader)
        results.append({'Task': task, 'Dataset': 'cifar10', 'Top-1': top1, 'Top-5': top5})
        print(results[-1], flush=True)

        # SVHN : même encodeur (gelé), nouveau classifieur linéaire entraîné sur SVHN
        model, _, train_loader, test_loader = load_model_and_data(task, dataset='svhn')
        clf_path = f'results/svhn_from_cifar10_{task}/linear.pth'
        if os.path.exists(clf_path):
            clf = model.construct_classifier().cuda()
            clf.load_state_dict(torch.load(clf_path))
            clf.eval()
        else:
            clf = train_linear_classifier(model, train_loader, epochs=10)
            os.makedirs(os.path.dirname(clf_path), exist_ok=True)
            torch.save(clf.state_dict(), clf_path)
        top1, top5 = evaluate_classifier(model, clf, test_loader)
        results.append({'Task': task, 'Dataset': 'svhn', 'Top-1': top1, 'Top-5': top5})
        print(results[-1], flush=True)

    df = pd.DataFrame(results)
    df.to_csv('results/comparaison.csv', index=False)
    print(df.pivot(index='Task', columns='Dataset', values='Top-1').round(2))

    df.pivot(index='Task', columns='Dataset', values='Top-1').loc[tasks].plot.bar(rot=0, figsize=(8, 4))
    plt.ylabel('Top-1 accuracy (%)')
    plt.title('Classification linéaire sur les features (encodeurs appris sur CIFAR10)')
    plt.tight_layout()
    plt.savefig('results/comparaison.png', dpi=150)


if __name__ == '__main__':
    main()
