"""
Script chia lai bo du lieu Zalo AI Traffic Sign 2020: Test 20% - Val 10% - Train 70%.

Thu tu cat: xao tron bang seed 42, cat 20% dau lam Hold-out Test, 10% tiep theo lam
Validation, phan con lai (~70%) lam Train.

Muc dich: dam bao ca 3 model (YOLOv8, Faster R-CNN, RT-DETR) dung CHUNG mot phep chia,
va tap Test 20% duoc giau hoan toan khoi qua trinh huan luyen.

Chay o dau cung duoc (Local / Kaggle / Colab), script tu do duong dan dataset goc.

Cach dung:
    python split_dataset.py                          # tu dong do moi thu
    python split_dataset.py --output-root ./data_v3  # chi dinh noi luu ket qua

Cau truc thu muc sinh ra:
    <output_root>/
    |-- dataset_train_val/          <- CHI thu muc nay duoc dung de train
    |   |-- train/images, train/labels
    |   |-- val/images, val/labels
    |   |-- train_annotations.json  (COCO subset - cho Faster R-CNN)
    |   |-- val_annotations.json    (COCO subset - cho Faster R-CNN)
    |   `-- data.yaml               (CHI co train + val, KHONG co test)
    |-- holdout_test/               <- TAP AN, chi mo ra khi danh gia cuoi cung
    |   |-- images/, labels/
    |   `-- holdout_test_annotations.json
    `-- split_manifest.json         <- Bien ban chia du lieu de doi chieu
"""

import argparse
import glob
import json
import os
import random
import shutil

# Seed co dinh - day la thu duy nhat dam bao tap Test khong bao gio thay doi
# qua cac lan chay va qua ca 3 mo hinh khac nhau.
RANDOM_SEED = 42

# Ty le chia: 20% Hold-out Test / 10% Validation / 70% Train.
# Thu tu cat rat quan trong: CAT TAP TEST RA TRUOC TIEN, roi moi den Val,
# phan con lai moi la Train. Xem giai thich o ham chia_ba_tap().
TY_LE_TEST = 0.2
TY_LE_VAL = 0.1

# 7 loai bien bao, thu tu giu nguyen theo dataset.yaml cua cac ban train truoc
CLASS_NAMES = [
    'No entry',
    'No parking / waiting',
    'No turning',
    'Max Speed',
    'Other prohibition signs',
    'Warning signs',
    'Mandatory signs',
]


# ==========================================================
# PHAN 1: DO TIM DU LIEU GOC
# ==========================================================

def do_tim_du_lieu_goc(thu_muc_goc=None):
    """Tim file JSON nhan va thu muc anh cua bo du lieu Zalo AI.

    Tu dong thu lan luot cac vi tri quen thuoc: Kaggle, Colab, roi den thu muc hien tai.
    """
    if thu_muc_goc:
        mau_tim = [os.path.join(thu_muc_goc, '**', 'train_traffic_sign_dataset.json')]
    else:
        mau_tim = [
            '/kaggle/input/**/train_traffic_sign_dataset.json',
            '/content/dataset/**/train_traffic_sign_dataset.json',
            '/content/**/train_traffic_sign_dataset.json',
            './**/train_traffic_sign_dataset.json',
        ]

    duong_dan_json = None
    for mau in mau_tim:
        ket_qua = glob.glob(mau, recursive=True)
        if ket_qua:
            duong_dan_json = sorted(ket_qua)[0]
            break

    if not duong_dan_json:
        raise FileNotFoundError(
            "Khong tim thay file train_traffic_sign_dataset.json.\n"
            "Neu chay tren Kaggle: nho Add Dataset 'phhasian0710/za-traffic-2020'.\n"
            "Neu chay tren Colab: nho tai va giai nen dataset truoc.\n"
            "Hoac chi dinh thu cong bang tham so --source-dir"
        )

    # Thu muc anh thuong nam ngay canh file JSON
    thu_muc_anh = os.path.join(os.path.dirname(duong_dan_json), 'images')
    if not os.path.isdir(thu_muc_anh):
        gia_tri_tim = glob.glob(
            os.path.join(os.path.dirname(os.path.dirname(duong_dan_json)), '**', 'traffic_train', 'images'),
            recursive=True,
        )
        if not gia_tri_tim:
            raise FileNotFoundError(f"Tim thay JSON tai {duong_dan_json} nhung khong thay thu muc anh di kem.")
        thu_muc_anh = gia_tri_tim[0]

    return duong_dan_json, thu_muc_anh


# ==========================================================
# PHAN 2: DOC VA CHIA DU LIEU
# ==========================================================

def doc_du_lieu_coco(duong_dan_json):
    """Doc file COCO JSON, tra ve (du_lieu_goc, thong_tin_anh, bbox_theo_anh)."""
    with open(duong_dan_json, 'r', encoding='utf-8') as f:
        du_lieu = json.load(f)

    thong_tin_anh = {img['id']: img for img in du_lieu['images']}

    # Gom bbox theo tung anh de truy xuat nhanh
    bbox_theo_anh = {}
    for ann in du_lieu['annotations']:
        img_id = ann['image_id']
        if img_id not in bbox_theo_anh:
            bbox_theo_anh[img_id] = []
        bbox_theo_anh[img_id].append(ann)

    return du_lieu, thong_tin_anh, bbox_theo_anh


def chia_ba_tap(danh_sach_id):
    """Xao tron bang seed 42 roi cat lan luot: Test 20% -> Val 10% -> Train (phan con lai).

    Vi sao phai cat tap Test ra TRUOC TIEN?
    Neu cat Train truoc (Train -> Val -> Test) thi tap Test nam o cuoi danh sach.
    Sau nay chi can chinh ty le Train mot chut, toan bo tap Test se bi xe dich theo
    va khong con so sanh duoc voi cac ket qua da chay truoc do.
    Cat Test ra dau tien thi no luon la 20% dau cua danh sach da xao tron, nen du
    co doi ty le Train/Val the nao, tap Test van dung y nguyen nhung buc anh cu.

    Seed co dinh nen chay script nay o may nao, luc nao cung ra ket qua y het.
    """
    danh_sach = list(danh_sach_id)
    random.seed(RANDOM_SEED)
    random.shuffle(danh_sach)

    tong_so = len(danh_sach)
    so_anh_test = int(tong_so * TY_LE_TEST)
    so_anh_val = int(tong_so * TY_LE_VAL)

    tap_test = danh_sach[:so_anh_test]                              # 20% dau tien
    tap_val = danh_sach[so_anh_test:so_anh_test + so_anh_val]       # 10% tiep theo
    tap_train = danh_sach[so_anh_test + so_anh_val:]                # ~70% con lai

    return tap_train, tap_val, tap_test


# ==========================================================
# PHAN 3: GHI DU LIEU RA DIA
# ==========================================================

def chuyen_bbox_coco_sang_yolo(bbox, chieu_rong, chieu_cao):
    """COCO [x_min, y_min, w, h] tinh bang pixel -> YOLO [xc, yc, w, h] chuan hoa 0-1."""
    x_min, y_min, w, h = bbox
    x_tam = (x_min + w / 2) / chieu_rong
    y_tam = (y_min + h / 2) / chieu_cao
    w_chuan = w / chieu_rong
    h_chuan = h / chieu_cao
    return x_tam, y_tam, w_chuan, h_chuan


def ghi_mot_tap(danh_sach_id, thu_muc_dich, thong_tin_anh, bbox_theo_anh, thu_muc_anh_goc):
    """Copy anh va sinh file nhan .txt chuan YOLO cho mot tap du lieu."""
    os.makedirs(os.path.join(thu_muc_dich, 'images'), exist_ok=True)
    os.makedirs(os.path.join(thu_muc_dich, 'labels'), exist_ok=True)

    so_anh_ghi = 0
    so_bbox_ghi = 0
    id_ghi_thanh_cong = []

    for img_id in danh_sach_id:
        thong_tin = thong_tin_anh[img_id]
        ten_file = thong_tin['file_name']
        duong_dan_nguon = os.path.join(thu_muc_anh_goc, ten_file)

        # Bo qua an toan neu file anh bi thieu, khong lam vang ca script
        if not os.path.exists(duong_dan_nguon):
            continue

        shutil.copy(duong_dan_nguon, os.path.join(thu_muc_dich, 'images', ten_file))

        ten_file_nhan = ten_file.rsplit('.', 1)[0] + '.txt'
        duong_dan_nhan = os.path.join(thu_muc_dich, 'labels', ten_file_nhan)

        with open(duong_dan_nhan, 'w', encoding='utf-8') as f_nhan:
            for ann in bbox_theo_anh.get(img_id, []):
                # JSON goc danh so lop tu 1 den 7, YOLO can 0 den 6 nen phai tru 1
                class_id = int(ann['category_id']) - 1
                x_tam, y_tam, w_chuan, h_chuan = chuyen_bbox_coco_sang_yolo(
                    ann['bbox'], thong_tin['width'], thong_tin['height']
                )
                f_nhan.write(f"{class_id} {x_tam:.6f} {y_tam:.6f} {w_chuan:.6f} {h_chuan:.6f}\n")
                so_bbox_ghi += 1

        so_anh_ghi += 1
        id_ghi_thanh_cong.append(img_id)

    return so_anh_ghi, so_bbox_ghi, id_ghi_thanh_cong


def ghi_json_coco_con(danh_sach_id, du_lieu_goc, thong_tin_anh, bbox_theo_anh, duong_dan_luu):
    """Xuat mot file COCO JSON chi chua anh cua tap duoc chon.

    Faster R-CNN doc truc tiep COCO JSON chu khong doc file .txt cua YOLO,
    nen phai co ban JSON rieng cho tung tap thi 3 model moi dung chung duoc phep chia.
    """
    tap_id = set(danh_sach_id)

    du_lieu_con = {
        'images': [thong_tin_anh[i] for i in danh_sach_id],
        'annotations': [ann for i in danh_sach_id for ann in bbox_theo_anh.get(i, [])],
        'categories': du_lieu_goc.get('categories', []),
    }
    if 'info' in du_lieu_goc:
        du_lieu_con['info'] = du_lieu_goc['info']

    with open(duong_dan_luu, 'w', encoding='utf-8') as f:
        json.dump(du_lieu_con, f, ensure_ascii=False)

    return len(tap_id), len(du_lieu_con['annotations'])


def ghi_data_yaml(thu_muc_train_val):
    """Sinh file data.yaml cho Ultralytics.

    CHU Y QUAN TRONG: file nay CHI khai bao train va val.
    Tuyet doi khong them khoa 'test' tro toi thu muc hold-out, vi Ultralytics
    se tu dong danh gia tren do va lam ro ri thong tin tap Test vao qua trinh chon model.
    """
    dong_ten_lop = '\n'.join(f'  {i}: {ten}' for i, ten in enumerate(CLASS_NAMES))

    noi_dung = f"""# File cau hinh dataset cho Ultralytics (YOLOv8 / RT-DETR)
# CHI CO train va val. Tap Test 20% duoc giau o thu muc holdout_test/ ben ngoai.
path: {thu_muc_train_val}
train: train/images
val: val/images

# Danh sach 7 loai bien bao (class_id chay tu 0 toi 6)
names:
{dong_ten_lop}
"""

    duong_dan_yaml = os.path.join(thu_muc_train_val, 'data.yaml')
    with open(duong_dan_yaml, 'w', encoding='utf-8') as f:
        f.write(noi_dung)

    return duong_dan_yaml


# ==========================================================
# PHAN 4: HAM CHINH
# ==========================================================

def xac_dinh_thu_muc_mac_dinh():
    """Doan noi luu ket qua dua tren moi truong dang chay."""
    if os.path.isdir('/kaggle/working'):
        return '/kaggle/working/data_v3'
    if os.path.isdir('/content'):
        return '/content/data_v3'
    return os.path.join(os.getcwd(), 'data_v3')


def main():
    parser = argparse.ArgumentParser(
        description='Chia dataset Zalo AI Traffic Sign theo ty le 70-10-20 voi seed co dinh 42.'
    )
    parser.add_argument('--source-dir', default=None,
                        help='Thu muc chua dataset goc. Bo trong de script tu do tim.')
    parser.add_argument('--output-root', default=None,
                        help='Noi luu ket qua. Bo trong de script tu chon theo moi truong.')
    parser.add_argument('--force', action='store_true',
                        help='Xoa thu muc ket qua cu truoc khi chia lai tu dau.')
    args = parser.parse_args()

    thu_muc_ket_qua = args.output_root or xac_dinh_thu_muc_mac_dinh()

    print('=' * 70)
    print('CHIA DATASET ZALO AI TRAFFIC SIGN 2020 THEO TY LE 70-10-20')
    print('=' * 70)

    # --- Buoc 1: Do tim du lieu goc ---
    duong_dan_json, thu_muc_anh = do_tim_du_lieu_goc(args.source_dir)
    print(f"File nhan goc : {duong_dan_json}")
    print(f"Thu muc anh   : {thu_muc_anh}")
    print(f"Noi luu ket qua: {thu_muc_ket_qua}\n")

    if args.force and os.path.isdir(thu_muc_ket_qua):
        print('Dang xoa thu muc ket qua cu...')
        shutil.rmtree(thu_muc_ket_qua)

    # --- Buoc 2: Doc va chia ---
    du_lieu_goc, thong_tin_anh, bbox_theo_anh = doc_du_lieu_coco(duong_dan_json)
    tap_train, tap_val, tap_test = chia_ba_tap(thong_tin_anh.keys())

    tong_so = len(thong_tin_anh)
    print(f"Tong so anh co nhan: {tong_so}")
    print("Thu tu cat: Test -> Val -> Train")
    print(f"  Test  : {len(tap_test)} anh ({len(tap_test) / tong_so:.1%})  <- TAP AN, cat ra dau tien")
    print(f"  Val   : {len(tap_val)} anh ({len(tap_val) / tong_so:.1%})")
    print(f"  Train : {len(tap_train)} anh ({len(tap_train) / tong_so:.1%})\n")

    # --- Buoc 3: Ghi tap Train va Val ---
    thu_muc_train_val = os.path.join(thu_muc_ket_qua, 'dataset_train_val')
    thu_muc_test = os.path.join(thu_muc_ket_qua, 'holdout_test')

    ket_qua_thong_ke = {}
    for ten_tap, danh_sach in [('train', tap_train), ('val', tap_val)]:
        print(f"Dang ghi tap {ten_tap}...")
        so_anh, so_bbox, id_thanh_cong = ghi_mot_tap(
            danh_sach, os.path.join(thu_muc_train_val, ten_tap),
            thong_tin_anh, bbox_theo_anh, thu_muc_anh
        )
        ghi_json_coco_con(
            id_thanh_cong, du_lieu_goc, thong_tin_anh, bbox_theo_anh,
            os.path.join(thu_muc_train_val, f'{ten_tap}_annotations.json')
        )
        ket_qua_thong_ke[ten_tap] = {'so_anh': so_anh, 'so_bbox': so_bbox, 'image_ids': id_thanh_cong}
        print(f"  -> {so_anh} anh, {so_bbox} bounding box")

    # --- Buoc 4: Ghi tap Hold-out Test (tach rieng hoan toan) ---
    print("Dang ghi tap Hold-out Test (giau khoi qua trinh train)...")
    so_anh, so_bbox, id_thanh_cong = ghi_mot_tap(
        tap_test, thu_muc_test, thong_tin_anh, bbox_theo_anh, thu_muc_anh
    )
    ghi_json_coco_con(
        id_thanh_cong, du_lieu_goc, thong_tin_anh, bbox_theo_anh,
        os.path.join(thu_muc_test, 'holdout_test_annotations.json')
    )
    ket_qua_thong_ke['holdout_test'] = {'so_anh': so_anh, 'so_bbox': so_bbox, 'image_ids': id_thanh_cong}
    print(f"  -> {so_anh} anh, {so_bbox} bounding box\n")

    # --- Buoc 5: Sinh data.yaml (chi train + val) ---
    duong_dan_yaml = ghi_data_yaml(thu_muc_train_val)
    print(f"Da tao {duong_dan_yaml} (chi khai bao train va val)")

    # --- Buoc 6: Ghi bien ban chia du lieu de doi chieu ve sau ---
    bien_ban = {
        'random_seed': RANDOM_SEED,
        'thu_tu_cat': ['holdout_test', 'val', 'train'],
        'ty_le': {
            'holdout_test': TY_LE_TEST,
            'val': TY_LE_VAL,
            'train': round(1 - TY_LE_TEST - TY_LE_VAL, 2),
        },
        'nguon_du_lieu': duong_dan_json,
        'tong_so_anh': tong_so,
        'thong_ke': {
            ten: {'so_anh': gt['so_anh'], 'so_bbox': gt['so_bbox']}
            for ten, gt in ket_qua_thong_ke.items()
        },
        'image_ids': {ten: gt['image_ids'] for ten, gt in ket_qua_thong_ke.items()},
    }

    duong_dan_bien_ban = os.path.join(thu_muc_ket_qua, 'split_manifest.json')
    with open(duong_dan_bien_ban, 'w', encoding='utf-8') as f:
        json.dump(bien_ban, f, ensure_ascii=False, indent=2)
    print(f"Da tao bien ban chia du lieu: {duong_dan_bien_ban}\n")

    # --- Buoc 7: Kiem tra cheo, chac chan 3 tap khong dinh nhau ---
    tap_train_set = set(ket_qua_thong_ke['train']['image_ids'])
    tap_val_set = set(ket_qua_thong_ke['val']['image_ids'])
    tap_test_set = set(ket_qua_thong_ke['holdout_test']['image_ids'])

    assert not (tap_train_set & tap_test_set), 'LOI NGHIEM TRONG: Train va Test bi trung anh!'
    assert not (tap_val_set & tap_test_set), 'LOI NGHIEM TRONG: Val va Test bi trung anh!'
    assert not (tap_train_set & tap_val_set), 'LOI NGHIEM TRONG: Train va Val bi trung anh!'

    print('=' * 70)
    print('HOAN TAT. Kiem tra cheo: 3 tap khong co bat ky anh nao trung nhau.')
    print(f"Dung file nay de train : {duong_dan_yaml}")
    print(f"TUYET DOI KHONG dung   : {thu_muc_test}  (chi mo khi danh gia cuoi cung)")
    print('=' * 70)


if __name__ == '__main__':
    main()
