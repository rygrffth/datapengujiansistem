import os
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

PARENT_FOLDER = "hasil_pengujian_50ws"
OUTPUT_FOLDER = "output_pengujian"

FOLDER_KONDISI = [
    "Arc_ke_Normal",
    "Arc_ke_Off",
    "Normal_ke_Arc",
    "Off_ke_Arc"
]

KOLOM_AKTUAL = "Output Sistem Aktual"
KOLOM_DIHARAPKAN = "Output Sistem yang Diharapkan"
# Menggunakan 'Waktu Relatif (detik)' jika diperlukan, tapi untuk delay dalam baris, ini tidak dipakai
# KOLOM_TIMESTAMP = "Waktu Relatif (detik)"

# Mapping untuk mencocokkan nama label di data dengan key transisi
LABEL_MAP = {
    'Status: Normal': 'Normal',
    'Status: Arc Flash': 'Arc',
    'Status: Off Contact': 'Off'
}

def analisis_waktu_tunda(df):
    """
    Menganalisis waktu tunda deteksi (dalam jumlah baris data) untuk setiap transisi.
    """
    delays = {}
    
    transisi_diharapkan_index = df[df[KOLOM_DIHARAPKAN].ne(df[KOLOM_DIHARAPKAN].shift())].index.tolist()
    
    for idx in transisi_diharapkan_index:
        if idx == 0:
            continue
        
        label_sebelum_raw = df.loc[idx - 1, KOLOM_DIHARAPKAN].strip()
        label_sekarang_raw = df.loc[idx, KOLOM_DIHARAPKAN].strip()
        
        if label_sebelum_raw == label_sekarang_raw:
            continue

        label_sebelum = LABEL_MAP.get(label_sebelum_raw, label_sebelum_raw)
        label_sekarang = LABEL_MAP.get(label_sekarang_raw, label_sekarang_raw)
        
        transisi_key = f"Dari {label_sebelum_raw} ke {label_sekarang_raw}"
        
        prediksi_index_pertama = df.index[(df.index >= idx) & (df[KOLOM_AKTUAL].str.strip() == label_sekarang_raw)]
        
        if not prediksi_index_pertama.empty:
            prediksi_idx = prediksi_index_pertama[0]
            
            # Menghitung delay sebagai jumlah baris
            delay_rows = prediksi_idx - idx
            
            if transisi_key not in delays:
                delays[transisi_key] = []
            delays[transisi_key].append(delay_rows)
            
    return delays

def analisis_data():
    """
    Fungsi utama untuk membaca semua file, menghitung metrik lengkap,
    dan menghasilkan laporan serta confusion matrix.
    """
    hasil_analisis = []
    semua_aktual = []
    semua_diharapkan = []
    semua_delays = {}
    
    print("Memulai analisis data...")
    
    path_parent_folder = os.path.join(os.getcwd(), PARENT_FOLDER)
    if not os.path.exists(path_parent_folder):
        print(f"KESALAHAN: Folder utama '{PARENT_FOLDER}' tidak ditemukan.")
        return

    path_output_folder = os.path.join(path_parent_folder, OUTPUT_FOLDER)
    os.makedirs(path_output_folder, exist_ok=True)
    print(f"Folder output '{path_output_folder}' berhasil dibuat.")

    file_processed = False
    for nama_folder in FOLDER_KONDISI:
        path_folder = os.path.join(path_parent_folder, nama_folder)
        if not os.path.exists(path_folder):
            print(f"\nPERINGATAN: Folder '{nama_folder}' tidak ditemukan. Melewati...")
            continue

        print(f"\nMenganalisis folder: {nama_folder}")
        
        total_data_kondisi, benar_kondisi = 0, 0

        for i in range(1, 11):
            nama_file = f"percobaan_{i}.csv"
            path_file = os.path.join(path_folder, nama_file)
            if not os.path.exists(path_file):
                print(f"  - File '{nama_file}' tidak ditemukan. Melewati...")
                continue
            
            try:
                df = pd.read_csv(path_file)
                if not file_processed:
                    print("\n--- Nama-nama Kolom yang Ditemukan di File Pertama ---")
                    print(df.columns.tolist())
                    print("-----------------------------------------------------")
                    file_processed = True

                if KOLOM_AKTUAL not in df.columns or KOLOM_DIHARAPKAN not in df.columns:
                    print(f"  - PERINGATAN: Kolom penting ('{KOLOM_AKTUAL}' atau '{KOLOM_DIHARAPKAN}') tidak ada di '{nama_file}'.")
                    continue
                
                prediksi_benar = (df[KOLOM_AKTUAL] == df[KOLOM_DIHARAPKAN]).sum()
                total_data_kondisi += len(df)
                benar_kondisi += prediksi_benar
                
                semua_aktual.extend(df[KOLOM_AKTUAL].tolist())
                semua_diharapkan.extend(df[KOLOM_DIHARAPKAN].tolist())

                delays_per_file = analisis_waktu_tunda(df)
                for key, values in delays_per_file.items():
                    if key not in semua_delays:
                        semua_delays[key] = []
                    semua_delays[key].extend(values)
                    
            except Exception as e:
                print(f"  - Gagal memproses file '{nama_file}': {e}")

        salah_kondisi = total_data_kondisi - benar_kondisi
        akurasi_persen = (benar_kondisi / total_data_kondisi * 100) if total_data_kondisi > 0 else 0
        
        hasil_analisis.append({
            "Nama Kondisi": nama_folder,
            "Total Data": total_data_kondisi,
            "Prediksi Benar (Sesuai)": benar_kondisi,
            "Prediksi Salah (Tidak Sesuai)": salah_kondisi,
            "Akurasi (%)": round(akurasi_persen, 2)
        })

    if hasil_analisis and all(h['Total Data'] > 0 for h in hasil_analisis):
        laporan_df = pd.DataFrame(hasil_analisis)
        total_data_keseluruhan = laporan_df['Total Data'].sum()
        total_benar_keseluruhan = laporan_df['Prediksi Benar (Sesuai)'].sum()
        total_salah_keseluruhan = laporan_df['Prediksi Salah (Tidak Sesuai)'].sum()
        akurasi_total_gabungan = (total_benar_keseluruhan / total_data_keseluruhan * 100) if total_data_keseluruhan > 0 else 0
        
        summary_row = pd.DataFrame([{"Nama Kondisi": "--- TOTAL KESELURUHAN ---", "Total Data": total_data_keseluruhan,
                                    "Prediksi Benar (Sesuai)": total_benar_keseluruhan, "Prediksi Salah (Tidak Sesuai)": total_salah_keseluruhan,
                                    "Akurasi (%)": round(akurasi_total_gabungan, 2)}])
        
        laporan_sederhana_df = pd.concat([laporan_df, summary_row], ignore_index=True)
        laporan_sederhana_df.to_csv(os.path.join(path_output_folder, "laporan_akurasi.csv"), index=False)
        print("\n\n--- Laporan Akurasi Sederhana ---")
        print(laporan_sederhana_df.to_string())
        print(f"\nLaporan 'laporan_akurasi.csv' berhasil dibuat di folder '{path_output_folder}'!")


        print("\n--- Laporan Klasifikasi Lengkap (per Kelas) ---")
        try:
            report_string = classification_report(semua_diharapkan, semua_aktual)
            print(report_string)
            report_dict = classification_report(semua_diharapkan, semua_aktual, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()
            report_df.reset_index(inplace=True)
            report_df = report_df.rename(columns={'index': 'Kelas'})
            report_df.to_csv(os.path.join(path_output_folder, "laporan_metrik_lengkap.csv"), index=False)
            print(f"Laporan 'laporan_metrik_lengkap.csv' berhasil dibuat di folder '{path_output_folder}'!")
        except Exception as e:
            print(f"Gagal membuat laporan klasifikasi: {e}")

    else:
        print("\nTidak ada data yang diproses. Laporan tidak dibuat.")


    if semua_aktual and semua_diharapkan:
        labels = sorted(list(set(semua_aktual) | set(semua_diharapkan)))
        cm = confusion_matrix(semua_diharapkan, semua_aktual, labels=labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        
        plt.title('Confusion Matrix Gabungan dari Semua Kondisi', fontsize=16)
        plt.ylabel('Output yang Diharapkan (Label Sebenarnya)', fontsize=12)
        plt.xlabel('Output Aktual (Prediksi Sistem)', fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(os.path.join(path_output_folder, "confusion_matrix.png"))
        print(f"\nGambar 'confusion_matrix.png' berhasil dibuat di folder '{path_output_folder}'!")
    else:
        print("\nTidak ada data untuk membuat confusion matrix.")

    print("\n--- Ringkasan Analisis Waktu Tunda ---")
    delay_summary = []
    
    for key, values in semua_delays.items():
        if values:
            avg = np.mean(values)
            min_val = np.min(values)
            max_val = np.max(values)
            delay_summary.append({
                "Jenis Transisi": key,
                "Rata-rata (baris data)": round(avg, 2),
                "Minimum (baris data)": int(min_val),
                "Maksimum (baris data)": int(max_val)
            })
    
    if delay_summary:
        delay_df = pd.DataFrame(delay_summary)
        print(delay_df.to_string(index=False))
        delay_df.to_csv(os.path.join(path_output_folder, "analisis_waktu_tunda.csv"), index=False)
        print(f"\nLaporan 'analisis_waktu_tunda.csv' berhasil dibuat di folder '{path_output_folder}'!")
    else:
        print("Tidak ada data transisi untuk dianalisis.")

if __name__ == "__main__":
    analisis_data()