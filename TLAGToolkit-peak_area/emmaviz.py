from tlagtk import visualizer as vz

app = vz.QApplication(vz.sys.argv)
window = vz.Window()
window.show()
vz.sys.exit(app.exec_())