content = open('main.py', 'r', encoding='utf-8').read()

content = content.replace(
    'from src_losses_circle import CircleLoss',
    'from src_losses_circle import CircleLoss\nfrom imbalance_circle_loss import ImbalanceAwareCircleLoss'
)

old = '    elif args.loss_type == "circle":'
new = '    elif args.loss_type == "imbalance_circle":\n        print(f"  CrossEntropy: label_smooth={args.label_smooth}")\n        print(f"  Imbalance-Aware Circle Loss: base_margin={args.circle_margin}")\n        from collections import Counter\n        from src.datasets import init_imgreid_dataset\n        _tmp_dataset = init_imgreid_dataset(root=args.root, name=args.source_names[0])\n        class_counts = Counter()\n        for img_path, pid, camid in _tmp_dataset.train:\n            class_counts[pid] += 1\n        criterion_xent = CrossEntropyLoss(\n            num_classes=dm.num_train_pids,\n            use_gpu=use_gpu,\n            label_smooth=args.label_smooth,\n        )\n        criterion_htri = ImbalanceAwareCircleLoss(\n            class_counts=dict(class_counts),\n            base_margin=args.circle_margin,\n            gamma=80,\n            power=0.5\n        )\n        use_arcface = False\n        use_circle = True\n\n    elif args.loss_type == "circle":'

content = content.replace(old, new)
open('main.py', 'w', encoding='utf-8').write(content)
print('Done')
