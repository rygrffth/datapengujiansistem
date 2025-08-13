import pandas as pd
import os
import re
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

output_folder = os.path.join('hasil_pengujian_25ws', 'output pengujian')
os.makedirs(output_folder, exist_ok=True)

file_pairs_info = [
    {
        "truth_folder": os.path.join('hasil_pengujian_25ws', 'akurasi 100 normal ke arc flash ke off contact'),
        "truth_prefix": 'akurasi 100 normal ke arc flash ke off contact',
        "pred_folder": os.path.join('hasil_pengujian_25ws', 'normal ke arc flash ke off contact'),
        "pred_prefix": 'normal ke arc flash ke off contact'
    },
    {
        "truth_folder": os.path.join('hasil_pengujian_25ws', 'akurasi100 off contact ke arc ke normal'),
        "truth_prefix": 'akurasi100 off contact ke arc ke normal',
        "pred_folder": os.path.join('hasil_pengujian_25ws', 'mentah fix off contact ke arc ke normal'),
        "pred_prefix": 'mentah fix off contact ke arc ke normal'
    }
]

def save_df_as_png(df, filename, title, footer_text=None):
    try:
        fig_height = 2 + len(df) * 0.5 + (1 if footer_text else 0)
        fig, ax = plt.subplots(figsize=(10, fig_height), dpi=200)
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#40466e')
        
        plt.title(title, fontsize=16, pad=20, weight='bold')
        
        if footer_text:
            plt.figtext(0.5, 0.05, footer_text, ha="center", fontsize=12, weight='bold')

        output_path = os.path.join(output_folder, filename)
        plt.tight_layout(pad=1)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.5)
        plt.close(fig)
        print(f"-> Gambar Tabel '{output_path}' berhasil disimpan.")
    except Exception as e:
        print(f"Gagal membuat gambar tabel '{filename}': {e}")

def find_transition_delays(truth_df, pred_df):
    delays = {}
    truth_df['Timestamp_dt'] = pd.to_datetime(truth_df['Timestamp'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    pred_df['Timestamp_dt'] = pd.to_datetime(pred_df['Timestamp'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    truth_transitions = truth_df[truth_df['Hasil_Prediksi'].ne(truth_df['Hasil_Prediksi'].shift())].index.tolist()
    for idx in truth_transitions:
        if idx == 0: continue
        label_from = truth_df.loc[idx - 1, 'Hasil_Prediksi'].strip().replace(' ⚠', '')
        label_to = truth_df.loc[idx, 'Hasil_Prediksi'].strip().replace(' ⚠', '')
        transition_key = f"{label_from} ke {label_to}"
        pred_transition_indices = pred_df.index[(pred_df.index >= idx) & (pred_df['Hasil_Prediksi'].str.contains(label_to))]
        if not pred_transition_indices.empty:
            pred_idx = pred_transition_indices[0]
            truth_time = truth_df.loc[idx, 'Timestamp_dt']
            pred_time = pred_df.loc[pred_idx, 'Timestamp_dt']
            delay_seconds = (pred_time - truth_time).total_seconds()
            if transition_key not in delays: delays[transition_key] = []
            delays[transition_key].append(delay_seconds)
    return {k: v[0] for k, v in delays.items() if v}

num_synthetic_files = 4
all_true_labels = []
all_pred_labels = []
mismatched_rows_list = []
transition_data = {}
delay_results = {'NORMAL ke ARC FLASH': [], 'ARC FLASH ke NO CONTACT': [], 'NO CONTACT ke ARC FLASH': [], 'ARC FLASH ke NORMAL': []}

print("--- Memulai Pengujian Akurasi Gabungan ---")

for pair in file_pairs_info:
    truth_folder = pair["truth_folder"]
    pred_folder = pair["pred_folder"]
    truth_prefix = pair["truth_prefix"]
    pred_prefix = pair["pred_prefix"]
    print(f"\nMenguji Skenario:\n1. '{truth_folder}'\n2. '{pred_folder}'\n" + "-"*60)
    
    file_indices = [''] + list(range(1, num_synthetic_files + 1))
    
    for i in file_indices:
        truth_filename = os.path.join(truth_folder, f"{truth_prefix}{i}.csv")
        pred_filename = os.path.join(pred_folder, f"{pred_prefix}{i}.csv")
        try:
            truth_df = pd.read_csv(truth_filename)
            pred_df = pd.read_csv(pred_filename)
            truth_df.columns = truth_df.columns.str.strip()
            pred_df.columns = pred_df.columns.str.strip()
            y_true = truth_df['Hasil_Prediksi'].str.strip()
            y_pred = pred_df['Hasil_Prediksi'].str.strip()
            
            truth_df['transition_group'] = (truth_df['Hasil_Prediksi'].ne(truth_df['Hasil_Prediksi'].shift())).cumsum()
            for group_num, group_df in truth_df.groupby('transition_group'):
                if len(group_df) > 1 and group_df.index[0] > 0:
                    label_from = truth_df.loc[group_df.index[0]-1, 'Hasil_Prediksi'].strip().replace(' ⚠', '')
                    label_to = group_df['Hasil_Prediksi'].iloc[0].strip().replace(' ⚠', '')
                    key_map = {'ARC FLASH': 'Arc', 'NORMAL': 'Normal', 'NO CONTACT': 'Off'}
                    transition_key = f"{key_map.get(label_from, label_from)} ke {key_map.get(label_to, label_to)}"
                    if transition_key not in transition_data: transition_data[transition_key] = {'true': [], 'pred': []}
                    transition_data[transition_key]['true'].extend(y_true[group_df.index])
                    transition_data[transition_key]['pred'].extend(y_pred[group_df.index])

            all_true_labels.extend(y_true)
            all_pred_labels.extend(y_pred)
            
            file_delays = find_transition_delays(truth_df, pred_df)
            for key, delay in file_delays.items():
                if key in delay_results: delay_results[key].append(delay)

            mismatch_mask = y_true != y_pred
            if mismatch_mask.any():
                mismatched_data = truth_df[mismatch_mask].copy()
                mismatched_data['Prediksi_Model'] = y_pred[mismatch_mask]
                mismatched_data.rename(columns={'Hasil_Prediksi': 'Label_Seharusnya'}, inplace=True)
                mismatched_data['Sumber_File'] = os.path.basename(truth_filename)
                mismatched_rows_list.append(mismatched_data)
        except FileNotFoundError:
            print(f"File tidak ditemukan: {truth_filename} atau {pred_filename}")
            continue
        except Exception as e:
            print(f"Error saat memproses file '{truth_filename}': {e}")
            continue

if all_true_labels:
    print("\n\n" + "="*50)
    print("---                                HASIL AKHIR PENGUJIAN                                ---")
    print("="*50)

    print("\n--- Hasil Akurasi per Skenario Transisi ---")
    transition_summary = []
    ordered_keys = ['Arc ke Normal', 'Arc ke Off', 'Normal ke Arc', 'Off ke Arc']
    for key in ordered_keys:
        if key in transition_data:
            data = transition_data[key]
            num_tests_for_key = len(delay_results.get(key.replace('Arc', 'ARC FLASH').replace('Normal', 'NORMAL').replace('Off', 'NO CONTACT'), []))
            total_data = len(data['true'])
            benar = accuracy_score(data['true'], data['pred'], normalize=False)
            salah = total_data - benar
            akurasi = accuracy_score(data['true'], data['pred'])
            transition_summary.append({'Nama Kondisi': key, 'Jumlah Pengujian': num_tests_for_key, 'Total Data': total_data, 'Prediksi Benar': benar, 'Prediksi Salah': salah, 'Akurasi (%)': f"{akurasi*100:,.2f}".replace('.', ',')})
    
    df_transition = pd.DataFrame()
    if transition_summary:
        df_transition = pd.DataFrame(transition_summary)
        total_row = df_transition[['Total Data', 'Prediksi Benar', 'Prediksi Salah']].sum()
        total_row['Nama Kondisi'] = 'Total'
        total_row['Jumlah Pengujian'] = df_transition['Jumlah Pengujian'].sum()
        total_accuracy = total_row['Prediksi Benar'] / total_row['Total Data']
        total_row['Akurasi (%)'] = f"{total_accuracy*100:,.2f}".replace('.', ',')
        df_transition = pd.concat([df_transition, pd.DataFrame(total_row).T], ignore_index=True)
        print(df_transition.to_string(index=False))
        save_df_as_png(df_transition, 'hasil_akurasi_per_transisi.png', 'Tabel 4.9 Hasil Akurasi per Skenario Transisi')

    print("\n--- Laporan Metrik Klasifikasi per Kelas ---")
    report_dict = classification_report(all_true_labels, all_pred_labels, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report_dict).transpose()
    df_report.drop('support', axis=1, inplace=True)
    df_report.rename(columns={'precision': 'Precision', 'recall': 'Recall', 'f1-score': 'F1-Score'}, inplace=True)
    unique_labels_report = sorted(list(set(all_true_labels)))
    cm_report = confusion_matrix(all_true_labels, all_pred_labels, labels=unique_labels_report)
    support_values = cm_report.sum(axis=1)
    df_report.loc[unique_labels_report, 'Support'] = support_values
    df_report['Support'] = df_report['Support'].fillna(df_report.loc[unique_labels_report, 'Support'].sum())
    df_report['Support'] = df_report['Support'].astype(int).astype(str)
    df_report.loc['accuracy', 'Support'] = ''
    df_report.index.name = 'Kelas'
    df_report.reset_index(inplace=True)
    df_report['Kelas'] = df_report['Kelas'].replace({'NO CONTACT': 'Off Contact', 'ARC FLASH ⚠': 'Arc Flash', 'NORMAL': 'Normal'})
    print(df_report.to_string(index=False))
    save_df_as_png(df_report, 'laporan_metrik_klasifikasi.png', 'Tabel 4.10 Laporan Metrik Klasifikasi per Kelas')

    print("\n--- Ringkasan Gabungan dari Semua Skenario ---")
    cm_all = confusion_matrix(all_true_labels, all_pred_labels, labels=unique_labels_report)
    summary_data = []
    for i, label in enumerate(unique_labels_report):
        total_data = cm_all[i, :].sum()
        benar = cm_all[i, i]
        salah = total_data - benar
        akurasi = (benar / total_data * 100) if total_data > 0 else 0
        summary_data.append({'Kondisi': label.replace(' ⚠', ''), 'Total Data': total_data, 'Prediksi Benar': benar, 'Prediksi Salah': salah, 'Akurasi (%)': f"{akurasi:.2f}%"})
    
    df_summary = pd.DataFrame(summary_data)
    total_row_summary = df_summary[['Total Data', 'Prediksi Benar', 'Prediksi Salah']].sum()
    total_row_summary['Kondisi'] = 'TOTAL KESELURUHAN'
    overall_accuracy = accuracy_score(all_true_labels, all_pred_labels)
    total_row_summary['Akurasi (%)'] = f"{overall_accuracy*100:.2f}%"
    df_summary = pd.concat([df_summary, pd.DataFrame(total_row_summary).T], ignore_index=True)
    
    print(df_summary.to_string(index=False))
    save_df_as_png(df_summary, 'ringkasan_hasil_akhir.png', 'Ringkasan Hasil Akhir Pengujian')

    print("\n--- Analisis Waktu Tunda Deteksi ---")
    print(f"{'Jenis Transisi':<25} | {'Rata-rata (detik)':<20} | {'Minimum (detik)':<20} | {'Maksimum (detik)':<20}")
    print("-" * 95)
    
    delay_summary_list = []
    for transition, delays in delay_results.items():
        if delays:
            avg_delay, min_delay, max_delay = np.mean(delays), np.min(delays), np.max(delays)
            print(f"{transition:<25} | {avg_delay:<20.3f} | {min_delay:<20.3f} | {max_delay:<20.3f}")
            delay_summary_list.append({
                'Jenis Transisi': transition,
                'Rata-rata (detik)': f"{avg_delay:.3f}",
                'Minimum (detik)': f"{min_delay:.3f}",
                'Maksimum (detik)': f"{max_delay:.3f}"
            })
        else:
            print(f"{transition:<25} | {'(tidak ada data)':<20} | {'(tidak ada data)':<20} | {'(tidak ada data)':<20}")
            delay_summary_list.append({
                'Jenis Transisi': transition,
                'Rata-rata (detik)': '(tidak ada data)',
                'Minimum (detik)': '(tidak ada data)',
                'Maksimum (detik)': '(tidak ada data)'
            })
    print("-" * 95)
    
    if delay_summary_list:
        df_delay_summary = pd.DataFrame(delay_summary_list)
        save_df_as_png(df_delay_summary, 'analisis_waktu_tunda_deteksi.png', 'Tabel Hasil Analisis Waktu Tunda Deteksi')
    
    output_excel_path = os.path.join(output_folder, 'laporan_pengujian_lengkap.xlsx')
    print(f"\nMenyimpan semua laporan ke file Excel: {output_excel_path}")
    try:
        with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
            if not df_transition.empty:
                df_transition.to_excel(writer, sheet_name='Akurasi per Transisi', index=False)
            if not df_report.empty:
                df_report.to_excel(writer, sheet_name='Metrik Klasifikasi', index=False)
            if not df_summary.empty:
                df_summary.to_excel(writer, sheet_name='Ringkasan Gabungan', index=False)
            if 'df_delay_summary' in locals() and not df_delay_summary.empty:
                df_delay_summary.to_excel(writer, sheet_name='Waktu Tunda Deteksi', index=False)
            if mismatched_rows_list:
                full_error_report = pd.concat(mismatched_rows_list, ignore_index=True)
                kolom_laporan = ['Sumber_File', 'Timestamp', 'Tegangan_V', 'Arus_A', 'Mean_V', 'Std_Dev_V', 'Mean_I', 'Std_Dev_I', 'Label_Seharusnya', 'Prediksi_Model', 'Label_Numerik']
                full_error_report[kolom_laporan].to_excel(writer, sheet_name='Detail Kesalahan', index=False)

        print("-> Berhasil menyimpan file Excel.")
    except Exception as e:
        print(f"Gagal menyimpan file Excel: {e}")

    try:
        print("\nMembuat Confusion Matrix...")
        unique_labels = sorted(list(set(all_true_labels) | set(all_pred_labels)))
        cm = confusion_matrix(all_true_labels, all_pred_labels, labels=unique_labels)
        plt.figure(figsize=(12, 9))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels, annot_kws={"size": 14})
        plt.title('Confusion Matrix Gabungan dari Semua Data', fontsize=16)
        plt.ylabel('Label Aktual (Seharusnya)', fontsize=12)
        plt.xlabel('Label Prediksi (Hasil Model)', fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        output_cm_path = os.path.join(output_folder, 'confusion_matrix_gabungan.png')
        plt.savefig(output_cm_path)
        print(f"-> Gambar Confusion Matrix berhasil disimpan.")
    except Exception as e:
        print(f"Gagal membuat gambar Confusion Matrix: {e}")
else:
    print("\nTidak ada file yang diproses.")