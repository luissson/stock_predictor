import sys
import os
sys.path.append(os.getcwd())
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import pdb

class predictor_gui(QDialog):


   def __init__(self, parent = None):
      super(predictor_gui, self).__init__(parent)
      self.buttons = QDialogButtonBox(QDialogButtonBox.Ok, Qt.Horizontal, self)
      self.buttons.move(260, 250)
      self.buttons.accepted.connect(self.accept)
      self.init_gui()


   def init_gui(self):   
      '''Main window display'''
      self.btn_traindates = QPushButton('Launch Train Dates Dialog', self)
      self.btn_traindates.move(10, 100)
      self.btn_traindates.clicked.connect(self.train_enddate)

      self.btn_intervals = QPushButton('Launch Prediction Interval Dialog', self)
      self.btn_intervals.move(10, 60)
      self.btn_intervals.clicked.connect(self.pred_intervals) 

      self.btn_symbols = QPushButton('Launch Stock Symbols Dialog', self)
      self.btn_symbols.move(10, 20)
      self.btn_symbols.clicked.connect(self.stock_symbol)

      self.intervals = QLineEdit(self)
      self.intervals.move(200, 60)    

      self.symbols = QLineEdit(self)
      self.symbols.move(200, 20)

      self.setGeometry(150, 150, 350, 300)
      self.setWindowTitle('Stock Price Predictor')
      self.show()


   def pred_intervals(self):
      '''List containing the days after the last training date to predict'''
      text, ok = QInputDialog.getText(self, 'Prediction Intervals Dialog', 
         'Enter number of days from last training date (e.g. 7, 14, 28, ...)')
      if ok:
         self.intervals.setText(str(text))


   def train_enddate(self):
      '''Calender popup allowing user to select date to end training'''
      layout = QVBoxLayout(self)
      self.text = QDateEdit(self)
      layout.addWidget(self.text)
      self.text.setCalendarPopup(True)


   def stock_symbol(self):
      '''Text input allowing user to input prediction stock symbols.'''
      text, ok = QInputDialog.getText(self, 'Stock Symbol Dialog', 
         'Enter the prediction symbols (e.g. AAPL, GOOG, ...):')
      if ok:
         self.symbols.setText(str(text))


   def get_intervals(self):
      '''Retrieve prediction intervals '''
      return str(self.intervals.text())


   def get_train_dates(self):
      '''Retrieve last train date '''
      end_date = self.text.dateTime().toPyDateTime()
      return end_date.strftime("%Y-%m-%d")


   def get_stocks(self):
      '''Retrieve stock symbols input '''
      return str(self.symbols.text())


   @staticmethod
   def get_inputs( parent = None):
      '''Method to retrieve user inputs as variables'''
      dialog = predictor_gui(parent)
      result = dialog.exec_()
      dates = dialog.get_train_dates()
      stocks = dialog.get_stocks()
      intervals = dialog.get_intervals()      
      return (dates, stocks, intervals, result == QDialog.Accepted)


def launch_gui():
      app = QApplication(sys.argv)
      dates, stocks, intervals, ok = predictor_gui.get_inputs()
      if ok:
         return dates, stocks, intervals
      sys.exit(app.exec_()) 
      exit()
