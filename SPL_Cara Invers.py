import tkinter as tk
from tkinter import messagebox, scrolledtext
import numpy as np

class SPLSolverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Penyelesaian SPL (Dengan Langkah-Langkah)")
        self.root.geometry("1000x750")
        self.root.configure(bg="#e09898")

        frame_top = tk.Frame(root, bg="#e6c3c3", pady=10)
        frame_top.pack(fill=tk.X)

        tk.Label(frame_top, text="Ordo Matriks (n):", bg="#e6c3c3", font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=10)
        self.entry_ordo = tk.Entry(frame_top, width=8, font=("Arial", 12))
        self.entry_ordo.pack(side=tk.LEFT, padx=5)
        self.entry_ordo.insert(0, "3") 

        tk.Label(frame_top, text="(Mendukung Copy-Paste | Steps Otomatis)", bg="#e6c3c3", font=("Arial", 10, "italic")).pack(side=tk.LEFT, padx=10)

        # --- Frame Input ---
        frame_main = tk.Frame(root, bg="#e09898")
        frame_main.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        frame_main.columnconfigure(0, weight=3)
        frame_main.columnconfigure(1, weight=1)
        frame_main.rowconfigure(1, weight=1)

        # Input A
        tk.Label(frame_main, text="Matriks A (Koefisien):", bg="#e09898", font=("Arial", 11, "bold")).grid(row=0, column=0, sticky="w")
        self.text_A = scrolledtext.ScrolledText(frame_main, height=8, font=("Consolas", 10))
        self.text_A.grid(row=1, column=0, sticky="nsew", padx=5)
        self.text_A.insert(tk.END, "1 2 3\n2 5 3\n1 0 8") 

        # Input B
        tk.Label(frame_main, text="Matriks B (Konstanta):", bg="#e09898", font=("Arial", 11, "bold")).grid(row=0, column=1, sticky="w")
        self.text_B = scrolledtext.ScrolledText(frame_main, height=8, font=("Consolas", 10))
        self.text_B.grid(row=1, column=1, sticky="nsew", padx=5)
        self.text_B.insert(tk.END, "5\n3\n17")

        # Tombol
        btn_solve = tk.Button(root, text="Hitung & Tampilkan Langkah", bg="#fefdfd", font=("Arial", 12, "bold"), command=self.solve_spl)
        btn_solve.pack(pady=5)

        # --- Output Langkah & Hasil ---
        tk.Label(root, text="Langkah Penyelesaian & Hasil:", bg="#e09898", font=("Arial", 11, "bold")).pack(anchor="w", padx=15)
        self.text_result = scrolledtext.ScrolledText(root, height=15, bg="#f0f0f0", font=("Consolas", 10))
        self.text_result.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    def parse_matrix(self, text_widget, rows, cols, is_vector=False):
        content = text_widget.get("1.0", tk.END).strip()
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if len(lines) != rows:
            raise ValueError(f"Jumlah baris input ({len(lines)}) tidak sesuai ordo ({rows}).")
        
        data = []
        for line in lines:
            line = line.replace(',', ' ') 
            parts = line.split()
            if not is_vector and len(parts) != cols:
                raise ValueError(f"Baris '{line}' harus memiliki {cols} kolom.")
            data.append([float(x) for x in parts])
        
        np_data = np.array(data)
        if is_vector and np_data.shape == (rows,): 
             np_data = np_data.reshape(rows, 1)
        return np_data

    def format_matrix_str(self, matrix, name="M"):
        """Mengubah array numpy menjadi string rapi untuk ditampilkan"""
        return f"{name} =\n{np.array2string(matrix, precision=2, suppress_small=True, separator=' ')}\n"

    def solve_spl(self):
        try:
            # 1. Persiapan Data
            try:
                n = int(self.entry_ordo.get())
            except:
                messagebox.showerror("Error", "Ordo harus integer.")
                return

            try:
                A = self.parse_matrix(self.text_A, n, n)
                B = self.parse_matrix(self.text_B, n, 1, is_vector=True)
            except ValueError as ve:
                messagebox.showerror("Input Error", str(ve))
                return

            # 2. Mulai Perhitungan & Pencatatan Langkah
            steps = f"=== PENYELESAIAN SPL METODE INVERS (Ordo {n}x{n}) ===\n\n"
            
            detA = np.linalg.det(A)
            steps += f"LANGKAH 1: Hitung Determinan Matriks A\n"
            steps += f"det(A) = {detA:.4f}\n\n"

            if np.isclose(detA, 0):
                steps += "HASIL: Matriks Singular (Determinan 0).\nTidak memiliki invers. Tidak ada solusi unik."
                self.text_result.delete("1.0", tk.END)
                self.text_result.insert(tk.END, steps)
                return
            
            if n <= 5: 
                steps += f"LANGKAH 2: Hitung Matriks Kofaktor\n"
                cofaktor_matrix = np.zeros((n, n))
                for i in range(n):
                    for j in range(n):
                        minor_mat = np.delete(np.delete(A, i, axis=0), j, axis=1)
                        minor_val = np.linalg.det(minor_mat)
                        cofaktor_matrix[i][j] = ((-1) ** (i + j)) * minor_val
                
                steps += self.format_matrix_str(cofaktor_matrix, "Matriks Kofaktor") + "\n"

                # Adjugate
                steps += f"LANGKAH 3: Hitung Adjugate (Transpose Kofaktor)\n"
                adjugate_matrix = cofaktor_matrix.T
                steps += self.format_matrix_str(adjugate_matrix, "Adj(A)") + "\n"

                # Invers
                steps += f"LANGKAH 4: Hitung Invers (Adj(A) / det(A))\n"
                inverse_A = adjugate_matrix / detA
                steps += self.format_matrix_str(inverse_A, "A^(-1)") + "\n"

            else:
                steps += f"LANGKAH 2 - 4: Menghitung Invers Matriks\n"
                steps += f"(Detail matriks Kofaktor & Adjoin disembunyikan karena ukuran {n}x{n} terlalu besar untuk ditampilkan)\n"
                inverse_A = np.linalg.inv(A) 
                steps += "Matriks Invers berhasil dihitung.\n\n"

            steps += f"LANGKAH 5: Kalikan Invers dengan Matriks B (X = A^(-1) . B)\n"
            X = np.dot(inverse_A, B)
            
            steps += "\n===== HASIL AKHIR (Nilai Variabel) =====\n"
            for i in range(n):
                steps += f"x{i+1} = {X[i][0]:.6f}\n"

            # Tampilkan ke layar
            self.text_result.delete("1.0", tk.END)
            self.text_result.insert(tk.END, steps)

        except Exception as e:
            messagebox.showerror("Error", f"Terjadi kesalahan: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SPLSolverApp(root)
    root.mainloop()