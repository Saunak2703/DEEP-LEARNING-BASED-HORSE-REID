import os
import os.path as osp
from .base import BaseImageDataset


class HorseReID(BaseImageDataset):

    dataset_dir = "horse"

    def __init__(self, root="datasets", verbose=True, **kwargs):
        super().__init__(root)
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.train_dir   = osp.join(self.dataset_dir, "train")
        self.test_dir    = osp.join(self.dataset_dir, "test")

        self._check_before_run()

        train          = self._process_train(self.train_dir)
        query, gallery = self._process_test(self.test_dir)

        if verbose:
            print("=> Horse Re-ID Dataset loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train   = train
        self.query   = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError(f"Dataset folder not found: {self.dataset_dir}")
        if not osp.exists(self.train_dir):
            raise RuntimeError(f"Train folder not found: {self.train_dir}")
        if not osp.exists(self.test_dir):
            raise RuntimeError(f"Test folder not found: {self.test_dir}")

    def _get_images_from_folder(self, folder_path):
        imgs = []
        for fname in sorted(os.listdir(folder_path)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                imgs.append(osp.join(folder_path, fname))
        return imgs

    def _process_train(self, train_dir):
        data = []
        pid  = 0
        for folder_name in sorted(os.listdir(train_dir)):
            folder_path = osp.join(train_dir, folder_name)
            if not osp.isdir(folder_path):
                continue
            for img_path in self._get_images_from_folder(folder_path):
                data.append((img_path, pid, 0))
            pid += 1
        return data

    def _process_test(self, test_dir):
        query   = []
        gallery = []
        pid     = 0
        for folder_name in sorted(os.listdir(test_dir)):
            folder_path = osp.join(test_dir, folder_name)
            if not osp.isdir(folder_path):
                continue
            imgs = self._get_images_from_folder(folder_path)
            if len(imgs) == 0:
                continue
            split = max(1, int(len(imgs) * 0.3))
            query_imgs   = imgs[:split]
            gallery_imgs = imgs[split:] if len(imgs) > 1 else imgs[:1]
            for img_path in query_imgs:
                query.append((img_path, pid, 0))   # query camid=0
            for img_path in gallery_imgs:
                gallery.append((img_path, pid, 1)) # gallery camid=1
            pid += 1
        return query, gallery
