from UI.ui_main import *
from Services.Repository import Repository
from Services.MachineLearning import MachineLearning
from Services.Model.Dataset import Dataset
from Services.Model.PredictedValues import PredictedValues
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from pyspark.sql.functions import split, regexp_replace, col
import numpy

WINDOW_SIZE = 0
ROW = 0
MENU_SELECTED_STYLESHEET = """
    border-left: 22px solid qlineargradient(spread:pad, x1:0.034, y1:0, x2:0.216, y2:0, stop:0.499 rgba(0, 170, 255, 255), stop:0.5 rgba(85, 170, 255, 0));
    background-color: rgb(40, 44, 52);
    """
SCORE_STYLESHEET = """
    QFrame {
	    border-radius: 92px;
	    background-color: qconicalgradient(cx:0.5, cy:0.5, angle:180, stop:{STOP_1} rgba(255, 85, 255, 0), stop:{STOP_2} rgba(85, 255, 127, 255));
    }
    """


class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.first = True
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowFlag(Qt.FramelessWindowHint)

        self.repository = Repository()
        self.ml = MachineLearning()

        self.df = self.repository.readFromDatabase()
        self.df_copy = self.df

        self.initialize()

        self.showDataSet(self.df)

        self.show()

    def initialize(self):
        self.ui.stackedWidget.setCurrentWidget(self.ui.page_1)
        self.ui.btnClose.clicked.connect(lambda: self.closeApp())
        self.ui.btnMin.clicked.connect(lambda: self.showMinimized())
        self.ui.btnMinMax.clicked.connect(lambda: self.minmaxWindowSize())
        self.ui.windowBar.mouseMoveEvent = self.moveWindow
        self.ui.btnHide.clicked.connect(lambda: self.toogleMenu(250, True))
        self.ui.btnMenuLoadCSV.clicked.connect(lambda: self.buttonClick())
        self.ui.btnMenuTrain.clicked.connect(lambda: self.buttonClick())
        self.ui.btnMenuPredict.clicked.connect(lambda: self.buttonClick())
        self.ui.btnMenuDisplay.clicked.connect(lambda: self.buttonClick())
        self.ui.pushButton_14.clicked.connect(lambda: self.buttonClick())
        self.ui.btnTable.clicked.connect(lambda: self.buttonClick())
        self.ui.btnChart.clicked.connect(lambda: self.buttonClick())

        self.ui.btnOpenCSV.clicked.connect(lambda: self.openCSVFile())
        self.ui.btnGoTrain.clicked.connect(lambda: self.trainModel())
        self.ui.btnGoPredict.clicked.connect(lambda: self.testModel())

        cmb_items = ["90/10", "85/15", "80/20", "75/25"]
        self.ui.cmbPercentageTrain.addItems(cmb_items)
        cmb_items2 = ["Select what to sort...", "Predisted price", "Actual price"]
        self.ui.cmb1WhatSortPredict.addItems(cmb_items2)
        cmb_items3 = ["Select how to sort...", "Ascending", "Descending"]
        self.ui.cmb2HowSortPredict.addItems(cmb_items3)
        self.ui.btnPredictSort.clicked.connect(lambda: self.sortPredictedValues())

        cmb_items4 = ["Select what to sort...", "Car name", "Horsepower", "Price"]
        self.ui.cmbWhatSortDS.addItems(cmb_items4)
        cmb_items5 = ["Select how to sort...", "Ascending", "Descending"]
        self.ui.cmbHowSortDS.addItems(cmb_items5)
        self.ui.btnSortDS.clicked.connect(lambda: self.sortDataset())
        self.ui.btnFilterPredict.clicked.connect(lambda: self.filterPredictedValues())

        cmb_items6 = self.df.dropDuplicates(["CompanyName"]).select('CompanyName').rdd.flatMap(lambda x: x).collect()
        cmb_items6.insert(0, "")
        self.ui.cmbCarNameDS.addItems(cmb_items6)

        cmb_items7 = ["", "gas", "diesel"]
        self.ui.cmbFueltypeDS.addItems(cmb_items7)
        self.ui.btnFilterDS.clicked.connect(lambda: self.filterDataset())

        self.ui.circularProgressBarBaseAccuracy_2.setHidden(True)
        self.ui.labelOpenCSVMessage.setHidden(True)
        self.ui.labelTrainMessage.setHidden(True)

        self.ui.horizontalLayout_31 = QtWidgets.QHBoxLayout(self.ui.frameChart)
        self.ui.horizontalLayout_31.setObjectName("horizontalLayout_31")
        self.ui.figure = plt.figure()
        self.ui.figure.patch.set_facecolor('#52527A')
        self.ui.canvas = FigureCanvas(self.ui.figure)
        self.ui.horizontalLayout_31.addWidget(self.ui.canvas)

    def closeApp(self):
        self.close()


    def minmaxWindowSize(self):
        global WINDOW_SIZE
        win_status = WINDOW_SIZE
        if win_status == 0:
            WINDOW_SIZE = 1
            self.showMaximized()
            self.ui.btnMinMax.setIcon(QtGui.QIcon(':/icons/icons/minimize.svg'))
        else:
            WINDOW_SIZE = 0
            self.showNormal()
            self.ui.btnMinMax.setIcon(QtGui.QIcon(':/icons/icons/maximize.svg'))

    def mousePressEvent(self, event):
        self.dragPos = event.globalPos()

    def moveWindow(self, event):
        global WINDOW_SIZE
        if event.buttons() == Qt.LeftButton:
            if WINDOW_SIZE == 1:
                self.minmaxWindowSize()
            self.move(self.pos() + event.globalPos() - self.dragPos)
            self.dragPos = event.globalPos()
            event.accept()

    def toogleMenu(self, maxWidth, enable):
        if enable:
            width = self.ui.leftMenu.width()
            maxEx = maxWidth
            standard = 65

            if width == 65:
                widthEx = maxEx
            else:
                widthEx = standard

            self.animation = QPropertyAnimation(self.ui.leftMenu, b"minimumWidth")
            self.animation.setDuration(300)
            self.animation.setStartValue(width)
            self.animation.setEndValue(widthEx)
            self.animation.start()

    def buttonClick(self):
        btn = self.sender()
        btnName = btn.objectName()

        if btnName == "btnMenuLoadCSV":
            self.ui.stackedWidget.setCurrentWidget(self.ui.page_2)
            self.resetStyle(btnName)
            btn.setStyleSheet(self.selectMenu(btn.styleSheet()))
        elif btnName == "btnMenuTrain":
            self.ui.stackedWidget.setCurrentWidget(self.ui.page_3)
            self.resetStyle(btnName)
            btn.setStyleSheet(self.selectMenu(btn.styleSheet()))
        elif btnName == "btnMenuPredict":
            self.ui.stackedWidget.setCurrentWidget(self.ui.page_4)
            self.resetStyle(btnName)
            btn.setStyleSheet(self.selectMenu(btn.styleSheet()))
        elif btnName == "btnMenuDisplay":
            self.ui.stackedWidget.setCurrentWidget(self.ui.page_5)
            self.resetStyle(btnName)
            btn.setStyleSheet(self.selectMenu(btn.styleSheet()))
        elif btnName == "btnTable":
            self.ui.stackedWidget_2.setCurrentWidget(self.ui.display_page1)
        elif btnName == "btnChart":
            self.ui.stackedWidget_2.setCurrentWidget(self.ui.display_page2)
        elif btnName == "pushButton_14":
            self.ui.stackedWidget.setCurrentWidget(self.ui.page_6)
            self.resetStyle(btnName)


    def resetStyle(self, widget):
        for w in self.ui.leftMenu.findChildren(QPushButton):
            if w.objectName() != widget:
                w.setStyleSheet(self.deselectMenu(w.styleSheet()))

    def selectMenu(self, style):
        select = style + MENU_SELECTED_STYLESHEET
        return select

    def deselectMenu(self, style):
        deselect = style.replace(MENU_SELECTED_STYLESHEET, "")
        return deselect

    def openCSVFile(self):
        file = QFileDialog.getOpenFileName(self, 'Open file', 'E:\milijevic\Desktop\bigdata\projekat')
        success = self.repository.writeCSVInDatabase(file[0])
        if(success):
            self.ui.labelOpenCSVMessage.setHidden(False)
            self.df = self.repository.readFromDatabase()
            self.df_copy = self.df
            self.showDataSet(self.df)

    def showDataSet(self, df):
        list = []
        listH = df.select('horsepower').rdd.flatMap(lambda x: x).collect()
        listCN = df.select('CompanyName').rdd.flatMap(lambda x: x).collect()
        listCM = df.select('CarModel').rdd.flatMap(lambda x: x).collect()
        listFT = df.select('fueltype').rdd.flatMap(lambda x: x).collect()
        listP = df.select('price').rdd.flatMap(lambda x: x).collect()
        for i in range(len(listH)):
            list.append(Dataset(listCN[i] + ", " + listCM[i], listH[i], listFT[i], listP[i]))
        self.ui.tableDataset.setRowCount(len(list))
        row = 0
        for obj in list:
            self.ui.tableDataset.setItem(row, 0, QtWidgets.QTableWidgetItem(obj.companyName))
            self.ui.tableDataset.setItem(row, 1, QtWidgets.QTableWidgetItem(str(obj.horsepower)))
            self.ui.tableDataset.setItem(row, 2, QtWidgets.QTableWidgetItem(obj.fueltype))
            self.ui.tableDataset.setItem(row, 3, QtWidgets.QTableWidgetItem(str(obj.price)))
            row+=1

    def trainModel(self):
        ratio = self.ui.cmbPercentageTrain.currentText()
        self.ml.TrainModel(ratio)
        self.ui.labelTrainMessage.setHidden(False)

    def testModel(self):
        y_test, y_pred, score = self.ml.TestModel()
        self.y_test = y_test.copy()
        self.y_test_copy = y_test.copy()
        self.y_pred = y_pred.copy()
        self.y_pred_copy = y_pred.copy()
        progress = (1 - score) * 0.5 / 1 + 0.5
        stop_1 = str(progress - 0.001)
        stop_2 = str(progress)
        new_stylesheet = SCORE_STYLESHEET
        new_stylesheet = new_stylesheet.replace("{STOP_1}", stop_1).replace("{STOP_2}", stop_2)
        self.ui.circularProgressAccuracy.setStyleSheet(new_stylesheet)
        self.ui.labPercentAccuracy.setText(str(round(score, 2) * 100) + "%")
        self.showPredictedValues()
        # df = self.repository.readFromDatabase()
        # list = []
        # listH2 = df.select('horsepower').rdd.flatMap(lambda x: x).collect()
        # listCN2 = df.select('CompanyName').rdd.flatMap(lambda x: x).collect()
        # listAV2 = df.select('price').rdd.flatMap(lambda x: x).collect()
        # listAV = pred.select('price').rdd.flatMap(lambda x: x).collect()
        # listPV = pred.select('prediction').rdd.flatMap(lambda x: x).collect()
        # listH = pred.select('xs[5]').rdd.flatMap(lambda x: x).collect()
        #
        # print(len(listH))
        # print(len(listH2))
        # print(len(listCN2))
        # print(len(listAV))
        # print(len(listAV2))
        # print(len(listPV))
        #
        # cmb_items = []
        # listCNR = []
        # for i in range(len(listH)):
        #     for j in range(len(listH2)):
        #         if(listAV[i] == listAV2[j] and listH[i] == listH2[j]):
        #             list.append(PredictedValues(listCN2[j], listH[i], listAV[i], listPV[i]))
        #             listCNR.append(listCN2[j])
        #             result = list.count(listCN2[j])
        #             if result == 0:
        #                 cmb_items.append(listCN2[j])
        #             break
        #
        # for i in range(len(listH)):
        #     for j in range(len(listH2)):
        #         if(listAV[i] == listAV2[j] and listH[i] == listH2[j]):
        #             list.append(PredictedValues(listCN2[j], listH[i], listAV[i], listPV[i]))
        #             listCNR.append(listCN2[j])
        #             result = list.count(listCN2[j])
        #             if result == 0:
        #                 cmb_items.append(listCN2[j])
        #             break
        #
        # self.ui.cmbCarNamePredict.addItems(cmb_items)
        #
        # self.ui.tableDisplay.setRowCount(len(y_test))
        # row = 0
        # for i in range(len(y_test)):
        #     self.ui.tableDisplay.setItem(row, 0, QtWidgets.QTableWidgetItem(str(y_pred[i])))
        #     self.ui.tableDisplay.setItem(row, 1, QtWidgets.QTableWidgetItem(str(y_test[i])))
        #     row += 1

        # self.ui.horizontalLayout_31 = QtWidgets.QHBoxLayout(self.ui.frameChart)
        # self.ui.horizontalLayout_31.setObjectName("horizontalLayout_31")
        #
        # self.ui.figure = plt.figure()
        # self.ui.figure.patch.set_facecolor('#52527A')
        # self.ui.canvas = FigureCanvas(self.ui.figure)
        # self.ui.horizontalLayout_31.addWidget(self.ui.canvas)
        # ax = plt.axes()
        # ax.set_facecolor("#3D3D4B")
        # plt.scatter(y_test, y_pred)
        # f = lambda x: x
        # plt.plot(y_test, f(y_test), lw=2.5, c="orange")
        # plt.xlabel("Test prices", fontweight='bold')
        # plt.ylabel("Predicted prices", fontweight='bold')
        # plt.title("Test values v/s Predicted values")
        # fig, ax = plt.subplots(nrows=1, ncols=1)
        # self.ui.canvas.draw()

    def showPredictedValues(self):
        self.ui.tableDisplay.setRowCount(len(self.y_test_copy))
        row = 0
        for i in range(len(self.y_test_copy)):
            self.ui.tableDisplay.setItem(row, 0, QtWidgets.QTableWidgetItem(str(self.y_pred_copy[i])))
            self.ui.tableDisplay.setItem(row, 1, QtWidgets.QTableWidgetItem(str(self.y_test_copy[i])))
            row += 1

        plt.clf()
        ax = plt.axes()
        ax.set_facecolor("#3D3D4B")
        plt.scatter(self.y_test_copy, self.y_pred_copy)
        f = lambda x: x
        plt.plot(self.y_test, f(self.y_test), lw=2.5, c="orange")
        plt.xlabel("Test prices", fontweight='bold')
        plt.ylabel("Predicted prices", fontweight='bold')
        plt.title("Test values v/s Predicted values")
        # fig, ax = plt.subplots(nrows=1, ncols=1)
        if(self.first):
            self.first = False

        self.ui.canvas.draw()


    def sortPredictedValues(self):
        what_sort = self.ui.cmb1WhatSortPredict.currentText()
        how_sort = self.ui.cmb2HowSortPredict.currentText()
        if what_sort == "Select what to sort..." or how_sort == "Select how to sort...":
            self.y_test_copy = self.y_test.copy()
            self.y_pred_copy = self.y_pred.copy()
            self.showPredictedValues()
        else:
            if how_sort == "Ascending":
                if what_sort == "Actual price":
                    indexes = numpy.argsort(self.y_test_copy)
                    self.y_test_copy = [self.y_test_copy[i] for i in indexes]
                    self.y_pred_copy = [self.y_pred_copy[i] for i in indexes]
                else:
                    indexes = numpy.argsort(self.y_pred_copy)
                    self.y_test_copy = [self.y_test_copy[i] for i in indexes]
                    self.y_pred_copy = [self.y_pred_copy[i] for i in indexes]
            else:
                if what_sort == "Actual price":
                    indexes = numpy.argsort(self.y_test_copy)
                    indexes = numpy.flip(indexes)
                    self.y_test_copy = [self.y_test_copy[i] for i in indexes]
                    self.y_pred_copy = [self.y_pred_copy[i] for i in indexes]
                else:
                    indexes = numpy.argsort(self.y_pred_copy)
                    indexes = numpy.flip(indexes)
                    self.y_test_copy = [self.y_test_copy[i] for i in indexes]
                    self.y_pred_copy = [self.y_pred_copy[i] for i in indexes]
            self.showPredictedValues()

    def sortDataset(self):
        what_sort = self.ui.cmbWhatSortDS.currentText()
        how_sort = self.ui.cmbHowSortDS.currentText()
        if what_sort == "Select what to sort..." or how_sort == "Select how to sort...":
            self.df_copy = self.df
            self.showDataSet(self.df_copy)
        else:
            if how_sort == "Ascending":
                if what_sort == "Car name":
                    df = self.df_copy.sort(col("CompanyName").asc(), col("CarModel").asc())
                elif what_sort == "Horsepower":
                    df = self.df_copy.sort(col("horsepower").asc())
                else:
                    df = self.df_copy.sort(col("price").asc())
            else:
                if what_sort == "Car name":
                    df = self.df_copy.sort(col("CompanyName").desc(), col("CarModel").desc())
                elif what_sort == "Horsepower":
                    df = self.df_copy.sort(col("horsepower").desc())
                else:
                    df = self.df_copy.sort(col("price").desc())
            self.df_copy = df
            self.showDataSet(self.df_copy)

    def filterPredictedValues(self):
        include_price = True
        if self.ui.lineEPriceMin.text() == "" and self.ui.lineEPriceMax.text() == "":
            include_price = False

        if include_price == True:
            price_min = 0
            price_max = 10000000
            price_min_str = self.ui.lineEPriceMin.text()
            price_max_str = self.ui.lineEPriceMax.text()
            if price_min_str == "":
                price_max = float(price_max_str)
                price_min = 0
            elif price_max_str == "":
                price_min = float(price_min_str)
                price_max = 10000000
            else:
                price_min = float(price_min_str)
                price_max = float(price_max_str)
            if price_min <= price_max:
                indexes = []
                for i in range(len(self.y_test_copy)):
                    if price_min > self.y_test_copy[i] or self.y_test_copy[i] > price_max:
                        indexes.append(i)
                for i in reversed(indexes):
                    self.y_test_copy.pop(i)
                    self.y_pred_copy.pop(i)
                self.showPredictedValues()
        # try:
        #     include_price = True
        #     if self.ui.lineEPriceMin.text() == "" and self.ui.lineEPriceMax.text() == "":
        #         include_price = False
        #
        #     if include_price == True:
        #         price_min = 0
        #         price_max = 10000000
        #         price_min_str = self.ui.lineEPriceMin.text()
        #         price_max_str = self.ui.lineEPriceMax.text()
        #         if price_min_str == "":
        #             price_max = float(price_max_str)
        #             price_min = 0
        #         elif price_max_str == "":
        #             price_min = float(price_min_str)
        #             price_max = 10000000
        #         else:
        #             price_min = float(price_min_str)
        #             price_max = float(price_max_str)
        #         if price_min <= price_max:
        #             indexes = []
        #             print(len(self.y_test_copy))
        #             for i in range(len(self.y_test_copy)):
        #                 if price_min > self.y_test_copy[i] or self.y_test_copy[i] > price_max:
        #                     indexes.append(i)
        #             for i in indexes:
        #                 self.y_test_copy.pop(i)
        #                 self.y_pred_copy.pop(i)
        #             self.showPredictedValues()
        # except:
        #     print("An exception occurred")

    def filterDataset(self):
        try:
            include_price = True
            include_horsepower = True
            include_company_name = True
            include_fuel_type = True
            if self.ui.lEPriceMinDS.text() == "" and self.ui.lEPriceMaxDS.text() == "":
                include_price = False
            if self.ui.lEHpMinDS.text() == "" and self.ui.lEHpMaxDS.text() == "":
                include_horsepower = False
            if self.ui.cmbCarNameDS.currentText() == "":
                include_company_name = False
            if self.ui.cmbFueltypeDS.currentText() == "":
                include_fuel_type = False

            df = self.df_copy

            if include_price == True:
                price_min = 0
                price_max = 10000000
                price_min_str = self.ui.lEPriceMinDS.text()
                price_max_str = self.ui.lEPriceMaxDS.text()
                if price_min_str == "":
                    price_max = float(price_max_str)
                    price_min = 0
                elif price_max_str == "":
                    price_min = float(price_min_str)
                    price_max = 10000000
                else:
                    price_min = float(price_min_str)
                    price_max = float(price_max_str)
                if price_min <= price_max:
                    df = df.filter(df.price >= price_min)
                    df = df.filter(df.price <= price_max)
            if include_company_name == True:
                company_name = self.ui.cmbCarNameDS.currentText()
                df = df.filter(df.CompanyName == company_name)
            if include_fuel_type == True:
                fuel_type = self.ui.cmbFueltypeDS.currentText()
                df = df.filter(df.fueltype == fuel_type)
            if include_horsepower == True:
                horsepower_min = 0
                horsepower_max = 10000000
                horsepower_min_str = self.ui.lEHpMinDS.text()
                horsepower_max_str = self.ui.lEHpMaxDS.text()
                if horsepower_min_str == "":
                    horsepower_max = int(horsepower_max_str)
                    horsepower_min = 0
                elif horsepower_max_str == "":
                    horsepower_min = int(horsepower_min_str)
                    horsepower_max = 10000000
                else:
                    horsepower_min = int(horsepower_min_str)
                    horsepower_max = int(horsepower_max_str)
                if horsepower_min <= horsepower_max:
                    df = df.filter(df.horsepower >= horsepower_min)
                    df = df.filter(df.horsepower <= horsepower_max)

            if df.count() > 0:
                self.df_copy = df
                self.showDataSet(self.df_copy)
        except:
            print("An exception occurred")
