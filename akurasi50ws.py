import os
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

PARENT_FOLDER = "hasil_pengujian_50ws"

FOLDER_KONDISI = [
    "Arc_ke_Normal",
    "Arc_ke_Off",
    "Normal_ke_Arc",
    "Off_ke_Arc"
]

KOLOM_AKTUAL = "Output Sistem Aktual"
KOLOM_DIHARAPKAN = "Output Sistem yang Diharapkan"

def hitung_delay_per_kondisi(df):
    delay_info = {}
    
    perubahan_kondisi_diharapkan = df[KOLOM_DIHARAPKAN].ne(df[KOLOM_DIHARAPKAN].shift()).index
    
    if len(perubahan_kondisi_diharapkan) > 1:
        for i in range(1, len(perubahan_kondisi_diharapkan)):
            indeks_awal_perubahan = perubahan_kondisi_diharapkan[i]
            kondisi_baru_diharapkan = df.loc[indeks_awal_perubahan, KOLOM_DIHARAPKAN]
            
            try:
                indeks_deteksi_awal = df.loc[indeks_awal_perubahan:][KOLOM_AKTUAL].eq(kondisi_baru_diharapkan).idxmax()
                delay = indeks_deteksi_awal - indeks_awal_perubahan
            except ValueError:
                delay = -1
            
            nama_perubahan = f"Dari {df.loc[indeks_awal_perubahan - 1, KOLOM_DIHARAPKAN]} ke {kondisi_baru_diharapkan}"
            
            if nama_perubahan not in delay_info:
                delay_info[nama_perubahan] = []
            delay_info[nama_perubahan].append(delay)
            
    return delay_info

def analisis_data():
    hasil_analisis = []
    semua_aktual = []
    semua_diharapkan = []
    semua_delay = {}

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
                print(f"  - File '{nama_file}' tidak ditemukan. Melewati...")
                continue
            
            try:
                df = pd.read_csv(path_file)
                if KOLOM_AKTUAL not in df.columns or KOLOM_DIHARAPKAN not in df.columns:
                    print(f"  - PERINGATAN: Kolom penting tidak ada di '{nama_file}'.")
                    continue
                
                delay_data = hitung_delay_per_kondisi(df)
                for kondisi, delays in delay_data.items():
                    if kondisi not in semua_delay:
                        semua_delay[kondisi] = []
                    semua_delay[kondisi].extend(delays)
                
                prediksi_benar = (df[KOLOM_AKTUAL] == df[KOLOM_DIHARAPKAN]).sum()
                total_data_kondisi += len(df)
                benar_kondisi += prediksi_benar
                
                semua_aktual.extend(df[KOLOM_AKTUAL].tolist())
                semua_diharapkan.extend(df[KOLOM_DIHARAPKAN].tolist())
            except Exception as e:
                print(f"  - Gagal memproses file '{nama_file}': {e}")

        salah_kondisi = total_data_kondisi - benar_kondisi
        akurasi_persen = (benar_kondisi / total_data_kondisi * 100) if total_data_kondisi > 0 else 0
        
        hasil_analisis.append({
            "Nama Kondisi": nama_folder,
            "Total Data": total_data_kondisi,
            "Prediksi Benar (Sesuai)": benar_kondisi,
            "Prediksi Salah (Tidak Sesuai)": salah_kondisi,
            "Akurasi (%)": round(akurasi_persen, 2)
        })

    if hasil_analisis:
        laporan_df = pd.DataFrame(hasil_analisis)
        total_data_keseluruhan = laporan_df['Total Data'].sum()
        total_benar_keseluruhan = laporan_df['Prediksi Benar (Sesuai)'].sum()
        total_salah_keseluruhan = laporan_df['Prediksi Salah (Tidak Sesuai)'].sum()
        akurasi_total_gabungan = (total_benar_keseluruhan / total_data_keseluruhan * 100) if total_data_keseluruhan > 0 else 0
        
        summary_row = pd.DataFrame([{"Nama Kondisi": "--- TOTAL KESELURUHAN ---", "Total Data": total_data_keseluruhan,
                                     "Prediksi Benar (Sesuai)": total_benar_keseluruhan, "Prediksi Salah (Tidak Sesuai)": total_salah_keseluruhan,
                                     "Akurasi (%)": round(akurasi_total_gabungan, 2)}])
        
        laporan_sederhana_df = pd.concat([laporan_df, summary_row], ignore_index=True)
        laporan_sederhana_df.to_csv(os.path.join(path_parent_folder, "laporan_akurasi.csv"), index=False)
        print("\n\n--- Laporan Akurasi Sederhana ---")
        print(laporan_sederhana_df.to_string())
        print(f"\nLaporan 'laporan_akurasi.csv' berhasil dibuat di folder '{PARENT_FOLDER}'!")

        print("\n--- Laporan Klasifikasi Lengkap (per Kelas) ---")
        try:
            report_string = classification_report(semua_diharapkan, semua_aktual)
            print(report_string)

            report_dict = classification_report(semua_diharapkan, semua_aktual, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()
            report_df.reset_index(inplace=True)
            report_df = report_df.rename(columns={'index': 'Kelas'})
            report_df.to_csv(os.path.join(path_parent_folder, "laporan_metrik_lengkap.csv"), index=False)
            print(f"Laporan 'laporan_metrik_lengkap.csv' berhasil dibuat di folder '{PARENT_FOLDER}'!")
        except ValueError as e:
            print(f"PERINGATAN: Gagal membuat laporan klasifikasi. Mungkin ada label yang tidak ada di data: {e}")

        print("\n--- Laporan Delay per Kondisi ---")
        if semua_delay:
            delay_data_final = []
            
            target_transitions = [
                "Dari Status: Arc Flash ke Status: Normal",
                "Dari Status: Arc Flash ke Status: Off Contact",
                "Dari Status: Normal ke Status: Arc Flash",
                "Dari Status: Off Contact ke Status: Arc Flash"
            ]
            
            for kondisi in target_transitions:
                if kondisi in semua_delay and semua_delay[kondisi]:
                    valid_delays = [d for d in semua_delay[kondisi] if d != -1]
                    avg_delay = sum(valid_delays) / len(valid_delays) if valid_delays else 'N/A'
                    delay_data_final.append({
                        "Perubahan Kondisi": kondisi,
                        "Rata-rata Delay (baris data)": round(avg_delay, 2) if isinstance(avg_delay, (int, float)) else avg_delay
                    })
            
            if delay_data_final:
                delay_df = pd.DataFrame(delay_data_final)
                print(delay_df.to_string())
                delay_df.to_csv(os.path.join(path_parent_folder, "laporan_delay.csv"), index=False)
                print(f"\nLaporan 'laporan_delay.csv' berhasil dibuat di folder '{PARENT_FOLDER}'!")
            else:
                print("Tidak ada data delay untuk transisi yang diminta.")
        else:
            print("Tidak ada data delay yang dapat dihitung.")

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
        
        plt.savefig(os.path.join(path_parent_folder, "confusion_matrix.png"))
        print(f"\nGambar 'confusion_matrix.png' berhasil dibuat di folder '{PARENT_FOLDER}'!")
    else:
        print("\nTidak ada data untuk membuat confusion matrix.")

if __name__ == "__main__":
    analisis_data()