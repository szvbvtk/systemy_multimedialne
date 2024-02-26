from docx import Document
from docx.shared import Inches
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

def generate_report(files, fsize_values):
    document = Document()
    document.add_heading('Analiza sinusoidalnych sygnałów', 0)

    for file in files:
        document.add_heading(f"Plik {file}", level=2)

        for fsize in fsize_values:
            document.add_heaqding(f"Rozmiar okna FFT: {fsize}", level=3)

            fig, axs = plt.subplots(2, 1, figsize=(10, 7))

            # Tu wykonujesz jakieś funkcje i rysujesz wykresy

            fig.suptitle(f"Rozmiar okna FFT: {fsize}")
            fig.tight_layout(pad=1.5)
            memfile = BytesIO()
            fig.savefig(memfile)

            document.add_picture(memfile, width=Inches(6))

            memfile.close()

            # Tu dodajesz dane tekstowe - wartości, wyjście funkcji etc.
            document.add_paragraph(f"Wartość losowa = {np.random.rand(1)}")

    document.save("report.docx")