"""
lotaria_ml_tkinter.py

Interface desktop (Tkinter). Coloquei toda a lógica de dados/ML em
lotaria_core.py; neste ficheiro só cuido da apresentação.

Execução: python lotaria_ml_tkinter.py
"""

import threading
import tkinter as tk
from collections import Counter
from datetime import datetime
from tkinter import messagebox, scrolledtext, ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from lotaria_core import SORTEIOS, COLUNAS_NUMEROS, AnalisadorLotaria, ResultadoInvalidoError


def montar_relatorio(analyzer: AnalisadorLotaria) -> str:
    previsoes = analyzer.prever_todos_sorteios()
    return f"""RELATORIO ESTATISTICO + MACHINE LEARNING
Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M')}
Total de sorteios na base: {len(analyzer.df)}

Sorteios de lotaria sao eventos aleatorios e independentes; este
relatorio e uma curiosidade estatistica, nao uma garantia de acerto.

==================================================
RANKING PARA OS PROXIMOS SORTEIOS
==================================================
Ranking Geral ...........: {previsoes['Geral']}

Fezada  (manha) .........: {previsoes['Fezada']}
Aqueceu (meio-dia) ......: {previsoes['Aqueceu']}
Kazola  (tarde) .........: {previsoes['Kazola']}
Eskebra (noite) .........: {previsoes['Eskebra']}

==================================================
INFORMACOES DO MODELO
==================================================
- Algoritmo: Random Forest Regressor
- Features: frequencia total, frequencia recente,
  atraso, peso temporal, posicao media e par/impar
- O modelo e retreinado automaticamente quando novos
  resultados sao adicionados.
- Combinacao: 65% Machine Learning + 35% Estatistica
"""


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Loto - Análise Estatística")
        self.root.geometry("1080x740")
        self.analyzer = AnalisadorLotaria()
        self._criar_interface()

    def _criar_interface(self) -> None:
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=8, pady=8)

        self._criar_aba_previsoes(notebook)
        self._criar_aba_adicionar(notebook)
        self._criar_aba_grafico(notebook)

        self.atualizar()

    def _criar_aba_previsoes(self, notebook: ttk.Notebook) -> None:
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Ranking / Previsoes")

        self.txt = scrolledtext.ScrolledText(frame, width=118, height=32, font=("Consolas", 10))
        self.txt.pack(padx=8, pady=8)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(pady=6)
        ttk.Button(
            btn_frame, text="Gerar Ranking / Previsoes", command=self.atualizar
        ).pack(side="left", padx=6)

    def _criar_aba_adicionar(self, notebook: ttk.Notebook) -> None:
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Adicionar Resultado")

        ttk.Label(frame, text="Data (AAAA-MM-DD):").grid(row=0, column=0, padx=6, pady=6, sticky="e")
        self.entry_data = ttk.Entry(frame, width=16)
        self.entry_data.insert(0, datetime.now().strftime("%Y-%m-%d"))
        self.entry_data.grid(row=0, column=1, padx=6, pady=6)

        ttk.Label(frame, text="Sorteio:").grid(row=1, column=0, padx=6, pady=6, sticky="e")
        self.combo = ttk.Combobox(frame, values=SORTEIOS, width=14, state="readonly")
        self.combo.current(0)
        self.combo.grid(row=1, column=1, padx=6, pady=6)

        self.entries: list[ttk.Entry] = []
        for i in range(len(COLUNAS_NUMEROS)):
            ttk.Label(frame, text=f"Numero {i + 1}:").grid(row=2 + i, column=0, padx=6, pady=4, sticky="e")
            entry = ttk.Entry(frame, width=10)
            entry.grid(row=2 + i, column=1, padx=6, pady=4)
            self.entries.append(entry)

        ttk.Button(
            frame, text="Salvar Resultado e Retreinar", command=self.adicionar
        ).grid(row=2 + len(COLUNAS_NUMEROS), column=0, columnspan=2, pady=16)

    def _criar_aba_grafico(self, notebook: ttk.Notebook) -> None:
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Grafico de Frequencia")
        self.frame_grafico = ttk.Frame(frame)
        self.frame_grafico.pack(fill="both", expand=True)
        ttk.Button(frame, text="Atualizar Grafico", command=self.mostrar_grafico).pack(pady=6)

    def atualizar(self) -> None:
        """Gero o relatório numa thread separada para não travar a UI
        enquanto o modelo treina / a previsão é calculada."""
        self.txt.delete(1.0, tk.END)
        self.txt.insert(tk.END, "Calculando ranking...\nAguarde alguns segundos...\n")

        def tarefa() -> None:
            relatorio = montar_relatorio(self.analyzer)
            self.root.after(0, self._exibir_relatorio, relatorio)

        threading.Thread(target=tarefa, daemon=True).start()

    def _exibir_relatorio(self, relatorio: str) -> None:
        self.txt.delete(1.0, tk.END)
        self.txt.insert(tk.END, relatorio)

    def adicionar(self) -> None:
        try:
            numeros = [int(entry.get()) for entry in self.entries]
        except ValueError:
            messagebox.showerror("Erro", "Todos os numeros devem ser inteiros.")
            return

        try:
            self.analyzer.adicionar_resultado(
                self.entry_data.get(), self.combo.get(), numeros
            )
        except ResultadoInvalidoError as erro:
            messagebox.showerror("Erro", str(erro))
            return

        messagebox.showinfo("Sucesso", "Resultado salvo. Gerando novo ranking...")
        self.atualizar()

    def mostrar_grafico(self) -> None:
        for widget in self.frame_grafico.winfo_children():
            widget.destroy()

        contagem = Counter(
            n for col in COLUNAS_NUMEROS for n in self.analyzer.df[col].tolist()
        ).most_common(20)
        numeros = [str(n) for n, _ in contagem]
        frequencias = [f for _, f in contagem]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(numeros, frequencias, color="steelblue")
        ax.set_title("Frequencia dos 20 Numeros Mais Sorteados")
        ax.set_xlabel("Numero")
        ax.set_ylabel("Quantidade")
        plt.xticks(rotation=45)

        canvas = FigureCanvasTkAgg(fig, master=self.frame_grafico)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
