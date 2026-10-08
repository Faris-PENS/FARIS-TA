# Menjalankan model di Proxmox

Jalankan inferensi di dalam VM Linux Proxmox. Contoh ini memakai Ubuntu Server dengan CPU; sebagai titik awal uji, alokasikan 2 vCPU, RAM 4 GB, dan disk 20 GB. Ukur kebutuhan aktual pada VM tersebut. Model ONNX tidak memerlukan PyTorch atau GPU untuk inferensi.

## 1. Siapkan VM

Di antarmuka Proxmox, buat VM Ubuntu Server, sambungkan ke bridge jaringan yang dapat diakses dari komputer Windows, lalu pasang Ubuntu. Di terminal VM, jalankan:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip openssh-server
mkdir -p ~/melon
```

Catat alamat IP VM dan nama pengguna Ubuntu. Pastikan SSH dari komputer Windows ke VM bisa terhubung. Contoh di bawah menggunakan `faris@192.168.1.50`; ganti dengan alamat Anda.

## 2. Salin model dan program

Jalankan di PowerShell Windows:

```powershell
Set-Location "C:\Users\User\Documents\TA-FARIS\TA-FARIS-2"
scp .\densenet121.onnx .\yolo_melon.onnx .\predict_onnx.py .\predict_full_image.py .\requirements-inference.txt faris@192.168.1.50:~/melon/
```

Salin juga gambar yang akan diuji ke VM dengan `scp`, atau gunakan gambar yang sudah tersedia di VM. File dataset training dan checkpoint `.pth` tidak diperlukan untuk inferensi ONNX.

## 3. Instal dan jalankan di VM

```bash
cd ~/melon
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-inference.txt
python predict_full_image.py /path/ke/foto_penuh.jpg
```

Program tersebut mendeteksi setiap kotak melon dengan YOLO, menambah padding 5%, kemudian mengklasifikasikan setiap crop dengan DenseNet. Jika gambar sudah berupa crop satu melon, gunakan:

```bash
python predict_onnx.py /path/ke/crop_melon.jpg
```

`predict_onnx.py` juga menerima `--bbox x_center y_center width height` atau `--label /path/ke/label.txt` bila kotak sudah diketahui. Jika foto berada dalam direktori `images/` dengan pasangan anotasi di direktori `labels/`, skrip memakai label secara otomatis. Tanpa kotak, seluruh gambar dipakai sebagai crop; model dilatih dengan crop melon.

Program menampilkan JSON berisi kotak, kelas, confidence, dan probabilitas setiap kelas. Ini adalah program terminal satu foto per perintah. Jika perlu diakses dari browser, HP, atau sistem lain melalui jaringan, diperlukan layanan web/API tambahan di dalam VM.

Dataset evaluasi memiliki satu melon berlabel per gambar. Sebelum menerima foto lapangan yang lebih lebar, tanpa melon, atau berisi beberapa melon, uji alur ini dengan contoh foto dari kamera pengguna.
