import os
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# --- KONFIGURASI ---
# Nama folder utama yang berisi semua folder kondisi
PARENT_FOLDER = "hasil_pengujian_final"

# Nama folder-folder kondisi yang ada di dalam PARENT_FOLDER
FOLDER_KONDISI = [
    "Arc_ke_Normal",
    "Arc_ke_Off",
    "Normal_ke_Arc",
    "Off_ke_Arc"
]

# Nama kolom yang akan dianalisis
KOLOM_AKTUAL = "Output Sistem Aktual"
KOLOM_DIHARAPKAN = "Output Sistem yang Diharapkan"

def analisis_data():
    """
    Fungsi utama untuk membaca semua file, menghitung metrik lengkap,
    dan menghasilkan laporan serta confusion matrix.
    """
    hasil_analisis = []
    semua_aktual = []
    semua_diharapkan = []

    print("Memulai analisis data...")
    
    path_parent_folder = os.path.join(os.getcwd(), PARENT_FOLDER)
    if not os.path.exists(path_parent_folder):
        print(f"KESALAHAN: Folder utama '{PARENT_FOLDER}' tidak ditemukan.")
        return

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
                if KOLOM_AKTUAL not in df.columns or KOLOM_DIHARAPKAN not in df.columns:
                    print(f"  - PERINGATAN: Kolom penting tidak ada di '{nama_file}'.")
                    continue
                
                prediksi_benar = (df[KOLOM_AKTUAL] == df[KOLOM_DIHARAPKAN]).sum()
                total_data_kondisi += len(df)
                benar_kondisi += prediksi_benar
                
                semua_aktual.extend(df[KOLOM_AKTUAL].tolist())
                semua_diharapkan.extend(df[KOLOM_DIHARAPKAN].tolist())
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

    # --- Membuat Laporan dan Validasi ---
    if hasil_analisis:
        # Laporan Akurasi Sederhana
        laporan_df = pd.DataFrame(hasil_analisis)
        total_data_keseluruhan = laporan_df['Total Data'].sum()
        total_benar_keseluruhan = laporan_df['Prediksi Benar (Sesuai)'].sum()
        total_salah_keseluruhan = laporan_df['Prediksi Salah (Tidak Sesuai)'].sum()
        akurasi_total_gabungan = (total_benar_keseluruhan / total_data_keseluruhan * 100) if total_data_keseluruhan > 0 else 0
        
        summary_row = pd.DataFrame([{"Nama Kondisi": "--- TOTAL KESELURUHAN ---", "Total Data": total_data_keseluruhan,
                                     "Prediksi Benar (Sesuai)": total_benar_keseluruhan, "Prediksi Salah (Tidak Sesuai)": total_salah_keseluruhan,
                                     "Akurasi (%)": round(akurasi_total_gabungan, 2)}])
        
        laporan_sederhana_df = pd.concat([laporan_df, summary_row], ignore_index=True)
        laporan_sederhana_df.to_csv("laporan_akurasi.csv", index=False)
        print("\n\n--- Laporan Akurasi Sederhana ---")
        print(laporan_sederhana_df.to_string())
        print("\nLaporan 'laporan_akurasi.csv' berhasil dibuat!")

        # --- Membuat Laporan Metrik Lengkap ---
        print("\n--- Laporan Klasifikasi Lengkap (per Kelas) ---")
        # Menghasilkan laporan sebagai string untuk ditampilkan di terminal
        report_string = classification_report(semua_diharapkan, semua_aktual)
        print(report_string)

        # Menghasilkan laporan sebagai dictionary untuk disimpan ke CSV
        report_dict = classification_report(semua_diharapkan, semua_aktual, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        report_df.reset_index(inplace=True)
        report_df = report_df.rename(columns={'index': 'Kelas'})
        report_df.to_csv("laporan_metrik_lengkap.csv", index=False)
        print("Laporan 'laporan_metrik_lengkap.csv' berhasil dibuat!")

    else:
        print("\nTidak ada data yang diproses. Laporan tidak dibuat.")

    # --- Membuat Confusion Matrix ---
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
        
        plt.savefig("confusion_matrix.png")
        print("\nGambar 'confusion_matrix.png' berhasil dibuat!")
    else:
        print("\nTidak ada data untuk membuat confusion matrix.")

if __name__ == "__main__":
    analisis_data()
