import os
import re
import pandas as pd
import pkg_resources
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.Chem.Draw import rdMolDraw2D
from PyQt6.QtCore import (
    Qt,
    QRunnable,
    pyqtSlot,
    pyqtSignal,
    QObject,
    QThreadPool,
    QAbstractTableModel,
    QProcess,
    QTemporaryDir,
    QSize
)
from PyQt6.QtGui import QPixmap, QIcon, QImage, QPainter
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QPushButton,
    QTableView,
    QPlainTextEdit,
    QProgressBar,
    QTextEdit
)
from utils import work_in_tmp_dir
from molecule import Molecule

VEHICLE_MOL2_FILENAME = pkg_resources.resource_filename("hcie", "Data/vehicle_dft.mol2")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HCIE")
        self.smiles = None
        self.process = None
        self.mol = None

        # Set up a temporary directory for the images to be stored in; this will be deleted when the application is shut
        self.image_directory = QTemporaryDir()

        self.results_window = ResultsWindow()

        self.threadpool = QThreadPool()

        # Define holders for the query image, and its molecular formula
        self.molecule_image = QLabel()
        self.molecular_formula = QLabel()

        # The box for inputting the SMILES string of the query molecule
        self.smiles_input = QLineEdit("SMILES")

        # Define the action buttons
        self.visualise_button = QPushButton("Visualise")
        self.run_button = QPushButton("Run")
        self.visualise_button.clicked.connect(self.visualise_smiles)
        self.run_button.clicked.connect(self.run_hcie)
        self.run_button.setEnabled(False)
        self.results_button = QPushButton("View Results")
        self.results_button.clicked.connect(self.show_results_window)
        if not os.path.exists('query_output.csv'):
            self.results_button.setEnabled(False)

        # The display that shows the progress of ShaEP
        self.progress_text_display = QPlainTextEdit()
        self.progress_text_display.setMaximumHeight(40)
        self.progress_text_display.setReadOnly(True)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)

        # Set the layout of the horizontal buttons
        horizontal_buttons_layout = QHBoxLayout()
        horizontal_buttons_layout.addWidget(self.visualise_button)
        horizontal_buttons_layout.addWidget(self.run_button)

        # Set the layout of the input and visualisation of SMILES display
        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.molecule_image)
        vertical_layout.addWidget(self.molecular_formula)
        vertical_layout.addWidget(self.progress_bar)
        vertical_layout.addWidget(self.progress_text_display)
        vertical_layout.addWidget(self.smiles_input)
        vertical_layout.addLayout(horizontal_buttons_layout)
        vertical_layout.addWidget(self.results_button)
        self.molecule_image.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.molecular_formula.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        container = QWidget()
        container.setLayout(vertical_layout)

        self.setCentralWidget(container)

    def visualise_smiles(self, path):
        """
        Code to generate an image, using draw_molecule, of the input SMILES string and visualise it in the display window.
        :return: None
        """
        self.smiles = self.smiles_input.text()
        rd_mol = Chem.MolFromSmiles(self.smiles)
        formula = CalcMolFormula(rd_mol)
        path_to_query_image = os.path.join(self.image_directory.path(), 'smiles_input.png')
        draw_molecule(path_to_query_image, rd_mol)
        self.molecule_image.setPixmap(QPixmap(path_to_query_image))
        self.molecular_formula.setText(formula)
        self.run_button.setEnabled(True)

        return None

    def run_hcie(self):
        """
        Runs a HCIE search on the input SMILES string, by generating a HCIE molecule, pre-processing it to generate
        co-ordinates, charges, and a MOL2 file. The ShaEP search is then executed externally using a QProcess.
        :return: None
        """
        if not self.smiles:
            raise TypeError(f'SMILES must be a string, not {type(self.smiles)}')
        else:
            self.mol = Molecule(self.smiles, max_hits=250)
            self.pre_shaep_processing(self.mol)
            self.shaep_search(self.mol)

        return None

    @staticmethod
    def pre_shaep_processing(molecule):
        """
        Optimises the geometry, generates the charges and writes a mol2 file, in preparation for a
        shaep search
        :param molecule: an instance of the HCIE Molecule class
        :return: None
        """
        molecule.do_geometry_optimisation_and_set_charges_and_coordinates(
            optimise=molecule.optimise
        )
        molecule.write_mol2_file()

        return None

    def post_shape_processing(self):
        """
        Collects the ShaEP outputs, interprets them, and outputs a HCIE file
        :param molecule: an instance of the HCIE Molecule class
        :return: None
        """
        self.mol.get_scores_from_similarity_file()
        self.mol.print_output_file()

        return None

    def shaep_search(self, mol):
        """
        Searches ShaEP for the query molecule against the VEHICLe library. This is run as an external process using
        QProcess.
        :param mol: an instance of the HCIE Molecule class
        :return: None
        """
        if self.process is None:
            self.process = QProcess()
            self.process.readyReadStandardOutput.connect(self.handle_stdout)
            self.process.readyReadStandardError.connect(self.handle_stderr)
            self.process.finished.connect(self.search_finished)
            self.process.start(
                "shaep",
                [
                    "--maxhits",
                    str(mol.max_hits),
                    "-v",
                    "1",
                    "-q",
                    mol.mol2_filename,
                    "--output-file",
                    "similarity.txt",
                    "--structures",
                    "overlay.sdf",
                    "--outputQuery",
                    VEHICLE_MOL2_FILENAME,
                ]
            )

        return None

    def execute_hcie_threads(self):
        """
        Executes HCIE using the Thread worker, so that the GUI does not crash
        """
        worker = Worker(self.run_hcie)
        self.threadpool.start(worker)
        self.show_results_window()
        return None

    def progress_update(self, message):
        progress = self.progress_percentage_parser(message)
        self.progress_text_display.appendPlainText(message)
        if progress:
            self.progress_bar.setValue(progress)

    def handle_stderr(self):
        """
        Code to read the standard error stream of an external process, receiving diagnostic and error messages
        from the process.
        :return: None
        """
        data = self.process.readAllStandardError()
        stderr = bytes(data).decode("utf8")
        self.progress_update(stderr)

        return None

    def handle_stdout(self):
        """
        Code to read the standard output stream of an external process, receiving result data
        :return: None
        """
        data = self.process.readAllStandardOutput()
        stdout = bytes(data).decode("utf8")
        self.progress_update(stdout)

        return None

    def search_finished(self):
        self.progress_update("Search Complete")
        self.process = None

        self.post_shape_processing()
        self.results_button.setEnabled(True)
        self.visualise_button.setEnabled(False)
        self.run_button.setEnabled(False)

        return None

    @staticmethod
    def progress_percentage_parser(output):
        """
        Simple method to determine how far through the search is.
        :param output: The line to search for the regex
        :return: an integer between 0 and 100
        """
        progress_re = re.compile(r"(\d+) structures processed.")
        match = progress_re.search(output)
        if match:
            progress = int(match.group(1))
            percentage = int(100 * progress / 24500)
        else:
            percentage = 0

        return percentage

    def show_results_window(self):
        self.results_window.mol = self.mol
        self.results_window.image_directory = self.image_directory
        self.results_window.show()
        self.results_window.get_results_table()


class ResultsWindow(QMainWindow):
    """
    Window to display results of HCIE search
    """

    def __init__(self):
        super().__init__()
        self.mol = None
        self.model = None
        self.image_directory = None
        self.table = QTableView()
        self.data = None

        self.resize(650, 650)

        self.setCentralWidget(self.table)

    def get_results_table(self):
        """
        Take the results csv file output from HCIE and interpolate it into a pandas dataframe for visualisation within
        the GUI.
        :return:
        """
        self.data = pd.read_csv(f'{self.mol.name}_output.csv',
                                sep=',',
                                header=0
                                )

        # Convert SMILES strings to RDKit Mol Objects
        PandasTools.AddMoleculeColumnToFrame(self.data,
                                             smilesCol='smiles',
                                             molCol='Molecule',
                                             includeFingerprints=True
                                             )
        self.draw_results_images()
        image_paths = [os.path.join(self.image_directory.path(), f'{regid}.png') for regid in self.data['regid']]
        self.data['Structure'] = image_paths

        # Reorder columns so that structures are displayed first
        self.data = self.data[['Structure', 'average_similarity', 'shape_similarity', 'esp_similarity', 'smiles', 'regid']]
        self.data = self.data.iloc[::-1]

        # Set the data table to be the model.
        self.model = TableModel(self.data)
        self.table.setModel(self.model)

        # Resize the table columns and rows to contain the images, and the window to display all the table columns
        self.table.resizeRowsToContents()
        self.table.resizeColumnsToContents()
        self.resize_window_to_contents()

    def draw_results_images(self):
        """
        draw images of each of the molecules in the dataframe, and store these in the temporary directory
        :return: None
        """
        for idx, rd_mol in enumerate(self.data['Molecule']):
            path = os.path.join(self.image_directory.path(), f'{self.data["regid"][idx]}.png')
            draw_molecule(path, rd_mol)

        return None

    def resize_window_to_contents(self):
        vh = self.table.verticalHeader()
        hh = self.table.horizontalHeader()
        size = QSize(hh.length(), vh.length())  # Get the length of the headers along each axis.
        size += QSize(vh.size().width(), hh.size().height())  # Add on the lengths from the *other* header
        size += QSize(20, 20)  # Extend further so scrollbars aren't shown.
        self.resize(size)


class Worker(QRunnable):
    """
    Worker Thread.
    This is needed to prevent the GUI from becoming unresponsive when executing a long calculation.
    It multi-threads the handling of the GUI and the task, so prevents the wheel of death.
    """

    def __init__(self, funct, *args, **kwargs):
        super(Worker, self).__init__()

        self.function = funct
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        """
        initialise the runner function with passed args and kwargs
        :return: None
        """
        try:
            result = self.function(*self.args, **self.kwargs)
        finally:
            self.signals.finished.emit()


class WorkerSignals(QObject):
    """
    Defines the signals to indicate to the thread, and the user, that the long-running calculation is complete
    """
    finished = pyqtSignal()


class TableModel(QAbstractTableModel):

    def __init__(self, data):
        super(TableModel, self).__init__()
        self._data = data

    def data(self, index, role):
        if role == Qt.ItemDataRole.DisplayRole:
            if index.column() != 0:
                value = self._data.iloc[index.row(), index.column()]
                return str(value)

        if role == Qt.ItemDataRole.DecorationRole and index.column() == 0:
            value = self._data.iloc[index.row(), index.column()]
            pixmap = QPixmap(value)
            return pixmap

    def rowCount(self, index):
        return self._data.shape[0]

    def columnCount(self, index):
        return self._data.shape[1]

    def headerData(self, section, orientation, role):
        # section is the index of the column/row.
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._data.columns[section])

            if orientation == Qt.Orientation.Vertical:
                return str(self._data.index[section])


def draw_molecule(path, molecule):
    """
    Code to generate an image, using RDKit, of the input SMILES string
    :param path: Path (including filename) of desired molecule
    :param molecule: an instance of an RDKit mol object
    :return: None
    """
    d = rdMolDraw2D.MolDraw2DCairo(250, 200)
    d.drawOptions().addStereoAnnotation = True
    d.drawOptions().clearBackground = False
    d.DrawMolecule(molecule)
    d.FinishDrawing()
    d.WriteDrawingText(path)

    return None


def main():
    app = QApplication([])
    window = MainWindow()
    window.resize(400, 450)
    window.show()
    app.exec()
    exit(app.exec())


if __name__ == '__main__':
    main()
