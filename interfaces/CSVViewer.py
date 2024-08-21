import warnings
from pandasgui import show
import pandas as pd

class CSVViewer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.dataset = None

    def load_data(self, array=None):
        if array is not None:
            self.dataset = pd.DataFrame(array)
            return
        try:
            self.dataset = pd.read_csv(self.file_path)
            print("Data loaded successfully.")
        except Exception as e:
            print(f"Error loading data: {e}")

    def show_data(self):
        if self.dataset is not None:
            if not self.dataset.empty:
                print(self.dataset.iloc[0])  # Acceso seguro por posición
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                gui = show(self.dataset)
        else:
            print("No data to show. Please load the data first.")

# Ejemplo de uso
if __name__ == "__main__":
    file_path = r"C:/machine-learning/machine-learning-az/data/modulo1/Data.csv"
    viewer = CSVViewer(file_path)
    viewer.load_data()
    viewer.show_data()
